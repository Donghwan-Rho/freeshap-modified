# Iteration 01 — Label-aware Eigenvector Selection for LRFShap

*Reading copy, 5 페이지 분량. plan_v2 + critique_v2 의 발췌 요약.*

## §1 Motivation & Question

LRFShap (lrfshap.pdf, Algorithm 1, p.4) 은 K(X_N, X_N) 의 best rank-r
approximation, 즉 **top-r by eigenvalue λ_i** 로 feature Φ_tr =
U_r Λ_r^{1/2} 를 만든 뒤 모든 prefix 의 ridge solve 를 r-차원 공간에서
수행한다. §5.5 (pp.7-8) 의 cross-dataset 실험은 이 단순한 truncation 의
한계를 드러낸다 — 같은 rank budget 에서도 SST-2 / MNLI 는 LC(r) 가
낮고 ER 이 1 근처에서 흔들리는 반면, AG News / MR 은 LC(r) 가 높고 ER 이
안정적으로 1 에 붙는다. Cross-dataset 으로만 보면 FC(r) 는 ER 을 거의
설명하지 못하고 (Fig. 6(b), Pearson(ER, FC) 가 r = 5–15% 에서 음수),
LC(r) 와 LC/FC 만 일관되게 0.7 이상의 상관을 보인다.

이 관찰을 그대로 받으면 가설은 자연스럽다: **rank budget 이 같다면, LC(r)
또는 LC/FC 를 더 끌어올리는 방향이 ER 을 직접 끌어올린다**. 핵심 질문은
세 가지 — (i) top-r by λ 가 아니라 LC 또는 LC/FC 를 maximize 하도록 r 개의
eigenvector 를 *재선택* 하면 ER 이 올라가는가, (ii) 그 류의 prior work 가
존재하는가, (iii) maximizer 를 어떻게 효율적으로 계산하는가. 이 reading
copy 는 (i)–(iii) 에 대한 알고리즘 3 개와 4 트랙 prior work mapping 을
요약한다.

## §2 Theoretical Backing — eq. (9) 가 직접 주는 supervised score

LRFShap 본문 §5.5 (lrfshap.pdf p.7, eq. (9), 같은 식이 appendix §A
p.12 eq. (8)–(9) 에서 재기술) 은 in-sample predictor gap 을 정확한
spectral identity 로 적어둔다:

  ‖f_ρ^FNTK(X_N) − f_ρ^LRNTK(X_N)‖² = Σ_{i>r} (λ_i / (λ_i + ρ))² · (u_i⊤ Y_N)²
                                    ≤ (1 − LC(r)) · ‖Y_N‖²

오른쪽 (1 − LC) bound 가 paper 가 명시적으로 사용한 형태지만, **진짜
동등식은 왼쪽** 이고 이쪽은 인덱스 집합 I ⊆ {1, …, n} 의 함수다 (단, basis
는 K(X_N) 의 eigendecomposition 이 미리 고정해 둔 U Λ U⊤ 안에 한정 — fixed
basis selection only). 따라서 임의의 I 에 대해

  E(I) := Σ_{i∉I} s_i,    s_i := (λ_i / (λ_i + ρ))² · (u_i⊤ Y_N)²

를 정의하면, |I| = r budget 하에서 E(I) 의 minimizer 는

  **I\* = top-r by s_i = (λ_i/(λ_i+ρ))² · (u_i⊤ Y_N)²**

이다. LRFShap 이 쓰는 top-r by λ_i 와의 차이는 ρ 의 자리에서 나온다.
ρ → ∞ 면 (λ/(λ+ρ))² 가 λ² 에 비례해 수축하므로 큰 λ 가 그대로 이긴다 —
그래서 top-r by λ 가 ρ 가 클 때 합리적이다. ρ → 0 면 필터가 1 로 saturate
하고 (u_i⊤Y)² 만 남아서 score 가 *순수 label alignment* 가 된다. **LRFShap
은 ρ = 1e-3 같은 작은 값에서 돌리니 후자에 가깝고, supervised selection
의 이득이 가장 클 가능성이 있다**.

수식 단위로 정리하면 — top-r by λ 는 식 (9) 의 *상한* (1 − LC) 만 줄이는
반면, supervised top-r 은 *equality* 인 LHS 자체를 줄인다. 이게 본 plan
의 이론적 anchor 다.

**S = N 한정성**. 위 식은 전체 학습셋 위의 predictor gap 이라, TMC 가
거치는 임의 subset S ⊊ N 에서는 ‖f_S^FNTK(X_S) − f_S^LRNTK(X_S)‖² 같은
*coalition-내부* gap 이 진짜 신경 써야 할 양이다. 본 plan 은 score s_i 를
매 coalition 마다 재계산하지 않고 **K(X_N) 에서 한 번 계산한 뒤 모든
permutation 에 동일하게 사용** 한다 — "S = N 의 minimizer 가 모든 |S| < n
coalition 에 대해서도 average-case 로 유의한 개선을 준다" 는 가정을
짊어진다. 이 가정은 식으로 보장되지 않으니 다음 iter 의 실험에서 verify
해야 한다.

## §3 Algorithmic Proposals (단순 → 정교)

세 알고리즘 모두 LRFShap Algorithm 1 의 line 3-4 (eigendecomp + Φ
construction) 만 교체하고 line 5-14 의 TMC loop 는 그대로 둔다. 즉
**preparation 단계에서만 추가 비용**, TMC iteration 안의 cost
O(M(n²r² + nr³)) 는 동일.

| Algo | Score 사용 | Decomp cost | Theory 정당화 강도 | Novelty 위치 | 실패 가능 |
|------|-----------|-------------|-------------------|-------------|----------|
| **A1** | s_i (full) | O(n³) | eq. (9) **exact min** | **main**: ρ-aware spectral filter | small-data noise overfit |
| **A2** | α_i = (u_i⊤Y)²/λ_i | O(n³) | §5.5 cross-dataset Pearson | **ablation**: Canatar α_i 를 selection 으로 | small-λ degenerate (falsifier #1) |
| **A3** | s_i (partial) | O(μrn²) flop | eq. (9) + power-law tail bound | **main**: partial decomp + supervised refinement | spectrum plateau, randomized SVD wall-clock |

**A1 (Greedy supervised top-r by s_i)**. Input: full eigendecomp K =
U Λ U⊤, labels Y_N, ρ, budget r, **null-mode floor ε (default ε = 10·ρ)**.
Procedure: (1) c_i := u_i⊤ Y_N (cost O(n²C)). (2) s_i = (λ_i/(λ_i+ρ))² ·
‖c_i‖_F² (multi-class Frobenius extension, §5.5 binary 정의의 자연스러운
일반화). (3) λ_i < ε 인 인덱스 제외 — ρ → 0 한계에서 (λ/(λ+ρ))² 가 1 로
saturate 되어 null direction 이 selection 에 들어와 ridge solve 의
conditioning 을 망가뜨리는 것을 방지. (4) I\* := top-r of surviving s_i.
(5) Φ_tr, Φ_te 빌드 → Algorithm 1.

**A2 (α_i ablation)**. r_i = (u_i⊤Y)² / (λ_i ‖Y‖²) 는 Canatar–Bordelon–
Pehlevan 2021 [B9] 의 task-model alignment α_i = (u_i⊤Y)²/λ_i 와 같은 양.
A2 의 실질적 contribution 은 "α_i 를 LRFShap 의 selection criterion 으로
옮긴 것" 이라 novelty 가 제한적이고, 본 plan 은 **A1 ↔ A2 분리 ablation**
으로만 사용한다. 분모의 λ_i 가 small-λ degenerate mode (falsifier #1) 를
1순위로 끌어올리는 약점이 있다.

**A3 (Partial decomp + supervised refinement)**. A1 의 약점은 full O(n³)
eigendecomp 를 전제로 한다는 것. n = 10000 (lrfshap §5.4 의 SST-2/MNLI
Pareto 실험 영역) 에서 의미 있는 비용이다. A3 는 **top-2r by λ 까지만
partial Lanczos / randomized SVD 로 뽑은 뒤 (Halko–Martinsson–Tropp 2011
[B13]), 그 안에서 supervised top-r 을 재선택**. Cost: preparation
O(μrn² + μ²r²n) — μ = 2, r = 0.1n 이면 ≈ 0.24n³ 로 *flop 기준* 약 4× 빠름.
다만 GPU 위 wall-clock 은 dense eigh 가 매우 잘 최적화돼 있어 randomized
방식이 항상 빠르다고 단정할 수 없으니, crossover 는 다음 iter 실험에서
직접 측정. Truncation error 분석 — λ 가 power-law decay (lrfshap Remark
E.3, p.21: λ_j ≤ C·j^{−α}) 하면 i > μr 의 supervised score mass 는
(μr+1)^{−2α} 로 빠르게 떨어져 충분히 작다.

## §4 Related Work Mapping (4 트랙)

**Track A — kernel-side leverage scores (label-agnostic)**. Alaoui–
Mahoney 2015 [B3] 의 ridge leverage ℓ_i(ρ) = [K(K+ρI)⁻¹]_{ii},
Bach 2013 [B4] 의 effective dimension d_eff(ρ) = Σ λ_i/(λ_i+ρ).
**우리 A1–A3 와 차이 — supervised refinement of ridge leverage**:
ridge leverage 의 mode-wise 형태는 λ_i/(λ_i+ρ) 로, 우리 score 와

  s_i = (λ_i / (λ_i + ρ))² · (u_i⊤Y)² = (mode-i ridge leverage)² · (u_i⊤Y)²

로 깔끔하게 factorize 된다. 즉 **본 plan 의 핵심은 "ridge leverage
score 의 supervised refinement"** — leverage 가 column sampling 에 쓰는
*label-agnostic* 가중을, 우리는 *label-aligned* term 과 곱해 eigenvector
selection 에 쓴다 (critique R1 §1 의 framing).

**Track B — kernel-target alignment**. Cristianini et al. 2001 [B5] 의
KTA, Cortes–Mohri–Rostamizadeh 2012 [B6] 의 centered KTA. **차이**: KTA
는 *kernel 자체* 를 (Y Y⊤ 와 align 되도록) 재구성. A1–A3 는 kernel 은
고정하고 그 spectral basis 안에서 *coordinate selection*. Wang et al.
AAAI 2014 / Neurocomputing 2016 [B7, B8] 의 label-aware base kernels 는
ideal kernel eigenfunction 을 unlabeled data 위로 extrapolate 해 새 base
kernel 을 만드는 방향 — 우리와 정반대 (kernel 고정 vs kernel 재구성).

**Track C — eigenlearning / spectrum-dependent generalization**.
Canatar–Bordelon–Pehlevan 2021 (Nature Comm) [B9] 가 task-model alignment
α_i 와 mode-wise learnability 도입, lrfshap §5.5 가 이미 인용. Simon–
Dickens–Karkada–DeWeese TMLR 2023 [B10] 이 후속 — KRR generalization 의
closed-form 이 정확히 mode-별 (λ_i/(λ_i+ρ))² · (u_i⊤y)² 항의 합. **즉
우리 score s_i 가 eigenlearning framework 의 mode-wise generalization
contribution 과 본질적으로 같은 양**. **차이**: 이 라인은 generalization
*분석* 도구이지, low-rank Shapley 안의 budget r 에서 어떤 mode 를 keep
할지를 다루지 않는다. A2 는 α_i 를 selection 으로 옮긴 것이라 novelty
제한적이고, A1 (s_i 가 ρ-aware), A3 (partial decomp 로 cost 까지 잡음)
가 본 plan 의 main contribution.

**Track D — data Shapley + low-rank / influence-based attribution**.
FreeShap [B1] 이 baseline. TRAK [B11] 은 random projection +
ensembling 으로 kernel spectrum 을 직접 쓰지 않음. DataInf [B12] 는
LoRA Hessian 의 closed-form inverse — Shapley 가 아니고 influence
function 계열. **차이**: A1–A3 는 LRFShap 안의 *low-rank basis 선택*
자체를 supervise 하는 직교적 contribution, prior work 에 직접 선례 없음.

## §5 Verification Plan (다음 iter 의 실험 설계)

본 plan 이 짊어진 검증 항목 네 가지 — 다음 iter 의 executor obligatory
실험.

**(V1) ER 향상의 cross-dataset 측정**. SST-2 / MNLI (LC 가 낮은 dataset)
+ AG News (LC 가 높아 saturated 인 baseline) 세 개로 ER vs r 곡선을
A1 / A3 / top-r-by-λ (LRFShap 원본) 비교. 가설: SST-2 / MNLI 에서 A1, A3
가 baseline 을 일관되게 위로 올리고, AG News 에서는 saturated 라 격차가
작다.

**(V2) S = N → S ⊊ N transfer 가정 검증**. Synthetic spectrum 에서
|S| = 0.1n, 0.3n, 0.7n, n 별로 supervised vs unsupervised selection 의
ER gap 측정. K(X_N) 의 score 가 작은 |S| 까지도 transfer 되는지 확인.

**(V3) Falsifier #1 — degenerate small-λ mode 분리 실험**. Synthetic
spectrum (power-law + 인위적 tail spike) 에서 A1, A2, top-r-by-λ 의 ER
측정. tail spike 가 강해질수록 A2 의 ER 만 baseline 아래로 떨어지면
A1 ↔ A2 가 분리됨을 confirm. 동시에 A1 의 ε floor sensitivity 도
ε ∈ {ρ, 10ρ, 100ρ} grid 로 측정.

**(V4) Falsifier #2 — label noise 시나리오**. SST-2 의 train label 5–20%
random flip + control (flip rate = 0%) 로 A1 vs top-r-by-λ 의 ER 비교.
가설: lrfshap Fig. 2 (p.6) 의 SST-2 1% selection 에서 LRFShap 이
baseline 을 *초과* 한 것이 implicit denoiser 효과였다면, supervised
selection 은 그 효과를 잃는다 (flip rate 가 높을수록 A1 의 ER 이
baseline 아래로 떨어지면 partial confirm).

추가로, A3 의 randomized SVD wall-clock crossover 측정 (n = 1000, 2000,
5000, 10000 + GPU dense eigh 와의 비교) 과, Prop 4.2 (lrfshap p.5) 의
원문 norm (operator vs nuclear) 한 줄 인용 — 이 두 가지는 다음 iter
의 reading 단계에서 보강.

## 남은 risk (critique R2 final sign-off 발췌)

- A3 의 partial decomp truncation error 분석에 (u_i⊤Y)² 의 spectrum 위
  분포 가정이 추가로 필요 (label 이 high-frequency mode 와 align 된
  adversarial case 에서 prefactor 가 ‖Y‖² 와 비례해 커질 가능성).
- §6 falsifier #2 의 flip rate 5–20% 가 SST-2 baseline (~91%) 를 흔들 수
  있으니 control = 0% 를 같은 plot 에 포함 권장.

두 항목 모두 plan_v2 본문 수정 없이 다음 iter 의 reading / 실험 설계 시
한 줄씩 보강하면 충분. 본 reading copy 의 결론에 영향 없음.

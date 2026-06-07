# Iteration 01 — Label-aware eigenvector selection 탐색 (planner v1)

## 1. Motivation

LRFShap 의 Algorithm 1 은 K = k(X_N, X_N) 의 best rank-r approximation,
즉 top-r by eigenvalue λ_i 로 feature Φ_tr = U_r Λ_r^{1/2} 를 만든 뒤
모든 prefix 의 ridge solve 를 r-dimensional space 에서 수행한다 (lrfshap.pdf,
Algorithm 1, p.4). §5.5 (pp.7–8) 가 보여주는 그림은 흥미롭다. 같은 rank
budget 에서도 SST-2 / MNLI 는 LC(r) 가 낮고 ER 이 1 근처에서 흔들리는 반면,
AG News / MR 은 LC(r) 가 높고 ER 이 안정적으로 1 에 붙는다. 그리고 cross-dataset
correlation 만 놓고 보면 FC(r) 는 ER 을 거의 설명하지 못하고 (Fig. 6(b),
Pearson(ER, FC) 가 r = 5–15% 에서 음수로 떨어짐), LC(r) 와 LC/FC 만 일관되게
0.7 이상으로 올라간다. 이 관찰을 그대로 받아들이면, **rank 가 같다면 같은
budget 으로 LC(r) 또는 LC/FC 를 더 끌어올리는 방향이 ER 을 직접 끌어올린다**
는 가설이 자연스럽다.

사용자가 던진 핵심 질문은 그래서 이렇다. (i) top-r by λ 가 아니라 LC 또는
LC/FC 를 maximize 하도록 r 개의 eigenvector 를 *재선택* 하면 ER 이 올라가는가,
(ii) 그 류의 prior work 가 있는가, (iii) maximizer 를 어떻게 효율적으로
계산하는가. 이 plan 은 (i)–(iii) 에 대해 알고리즘 3 개와 4 트랙 prior work
mapping 을 제시한다.

## 2. Theoretical backing — eq. (9) 가 직접 주는 supervised score

LRFShap 의 §A (lrfshap.pdf, p.12, eq. (8)–(9)) 는 in-sample predictor gap 을
정확한 spectral identity 로 적어둔다:

  ‖f_ρ^FNTK(X_N) − f_ρ^LRNTK(X_N)‖² = Σ_{i>r} (λ_i / (λ_i + ρ))² · (u_i⊤ Y_N)²
                                    ≤ (1 − LC(r)) · ‖Y_N‖².

오른쪽 상한이 (1 − LC) 로 나오긴 하지만 진짜 동등식은 왼쪽이다. 그리고 왼쪽은
**선택된 인덱스 집합 I ⊆ {1, …, n} 의 함수** 다 (basis 가 best rank-r 이라
강제할 이유는 증명에 없다). 즉 임의의 I 에 대해

  E(I) := Σ_{i∉I} s_i,    s_i := (λ_i / (λ_i + ρ))² · (u_i⊤ Y_N)²

를 정의하면, |I| = r 이라는 budget 하에서 E(I) 를 최소화하는 I 는

  **I\* = top-r by s_i, i.e., supervised score s_i = (λ_i/(λ_i+ρ))² · (u_i⊤ Y_N)²**

이다. 반면 LRFShap 이 쓰는 것은 top-r by λ_i. 두 선택의 차이는 ρ 의 자리에서
나온다. ρ → ∞ 면 (λ/(λ+ρ))² 가 λ² 에 비례해 수축하므로 큰 λ 가 그대로
이긴다 — 그래서 top-r by λ 가 ρ 가 클 때 합리적이다. ρ → 0 면 필터가 1 로
saturate 하고 (u_i⊤ Y)² 만 남아서 score 가 *순수 label alignment* 가 된다.
LRFShap 은 ρ = 1e-3 같은 작은 값에서 돌리니 후자에 가깝고, 그래서 supervised
selection 의 이득이 가장 클 가능성이 있다.

수식 단위로 차이는 명료하다. top-r by λ 는 식 (9) 의 *상한* (1 − LC) 만
줄이는 반면, supervised top-r 은 *equality* 인 LHS 자체를 줄인다. Prop 4.2
(p.5) 의 Shapley value 안정성 bound 가 ‖K̃ − K‖₂ 에 의존하긴 하지만, 거기서
‖K̃ − K‖₂ 자리는 사실 in-sample predictor gap 의 proxy 일 뿐이고 (Cortes et
al. 2010 [B14] 의 원래 bound 도 predictor gap 을 직접 쓴다), 우리가 minimize
하는 양과 알고리즘이 신경 쓰는 양이 일치하는 쪽은 supervised selection 이다.

## 3. Algorithmic proposals (단순 → 정교)

### Algorithm A1 — Greedy supervised top-r by s_i (baseline 변형)

Input: full eigendecomp K = U Λ U⊤, labels Y_N, ridge ρ, budget r.

Procedure:
1. Compute c_i := u_i⊤ Y_N for i = 1, …, n (cost O(n²C) 한 번).
2. Compute s_i = (λ_i / (λ_i + ρ))² · c_i² (binary case) or Σ_c c_{i,c}²
   (multi-class, §5).
3. I\* := indices of top-r values of s_i.
4. Build Φ_tr := U_{I\*} Λ_{I\*}^{1/2}, Φ_te := K_te U_{I\*} Λ_{I\*}^{−1/2}.
5. Run LRFShap Algorithm 1 line 5–14 unchanged with these Φ_tr, Φ_te.

Cost: 한 번의 추가 비용은 O(n²C) for c_i + O(n) for sorting. 본 LRFShap 의
preparation cost 가 이미 O(n³) (eigen decomp, lrfshap.pdf Table 2, p.23) 이므로
**preparation 단계에서는 dominated**. TMC iteration 안의 cost 는 r 이 동일하다면
완전히 동일 (line 11, O(n²r² + nr³) per permutation).

Eq. (9) 와의 관계: §2 에서 도출한 minimizer 를 그대로 사용. predictor gap 의
exact spectral identity 를 minimize.

위험 / 한계: (i) eigenvalue 가 매우 작은 i 가 c_i² 만으로 뽑히면 ridge solve
의 conditioning 이 나빠진다 — Φ_S⊤ Φ_S + ρI 의 smallest eigenvalue 가 ρ
바닥에서 결정되긴 하지만 line 11 의 numerical stability 는 점검 필요. (ii)
TMC 의 marginal gain 분포가 변해 early-stop ratio 가 달라지고 wall-clock
speedup 이 LRFShap 의 figure 4 (p.6) 와 다른 trend 를 보일 수 있다. (iii)
SST-2 / MNLI 의 1% selection 에서 LRFShap 이 baseline 을 *초과* 한 것 (Fig. 2,
p.6) 은 unsupervised low-rank truncation 이 우연히 noise filter 역할을 했을
가능성이 있는데, supervised selection 은 noise 까지 같이 fitting 해버려서 이
"regularizer 효과" 를 잃을 수 있다.

### Algorithm A2 — LC/FC-balanced score (LC/FC trade-off 직접 반영)

Input: 위와 동일.

Procedure:
1. c_i² = (u_i⊤ Y_N)², 각 i 에 대해 ratio r_i := c_i² / (λ_i · ‖Y_N‖²) 계산.
2. (i) Sort by r_i descending → top-r 선택 (LC/FC ratio greedy).
   또는 (ii) constrained: maximize Σ_{i∈I} c_i² / ‖Y‖² (LC) subject to
   Σ_{i∈I} λ_i / Σ_j λ_j ≤ τ (FC budget). τ ∈ {0.5, 0.7, 0.9} 로 grid.
3. 이후는 A1 과 동일 (Φ_tr, Φ_te 빌드 → Algorithm 1).

Cost: A1 과 동일 O(n²C) preparation 추가. Constrained 변형 (ii) 는 fractional
knapsack 이라 O(n log n).

Eq. (9) 와의 관계: §5.5 의 cross-dataset 분석이 직접 motivate. Fig. 6(c)
(p.8) 가 LC/FC 와 ER 이 LC 단독보다 cross-dataset Pearson 이 일관되게 높다고
보고하므로, dataset 별 LC/FC 가 다른 origin 을 *동일 budget 안에서* 끌어올릴
수 있다. r_i 점수는 사실 ρ → 0 에서의 s_i 를 λ_i 로 한 번 더 나눈 것과
같으므로 — eq. (9) 의 spectral filter (λ/(λ+ρ))² 가 ρ ≪ λ 영역에서 1 로
saturate 한다는 사실을 활용한 small-ρ heuristic.

위험 / 한계: 큰 λ 의 eigenvector 를 제외하면 line 11 ridge regression 의
features 가 작은 norm 에 몰려 numerical error 가 키워질 수 있다. 또 r_i 는
ρ-independent 라 ρ 가 크지 않은 영역에서만 정당화된다. RTE 같이 n 이 작고
LC/FC 가 들쭉날쭉한 dataset (Fig. 12, p.13) 에서는 ratio 가 unstable.

### Algorithm A3 — Partial-decomp + supervised refinement (정교)

가장 큰 우려는 A1, A2 가 모두 "full eigendecomp 이미 있다" 를 전제로
한다는 점이다. n = 10000 일 때 O(n³) eigen decomp 자체가 비싸진다 (lrfshap.pdf
Table 2, p.23: preparation O(n³) 가 그대로 살아있음). A3 는 **top-2r by λ
까지만 partial Lanczos / randomized SVD 로 뽑은 뒤, 그 안에서 supervised
top-r 을 재선택** 한다.

Input: K, Y_N, ρ, r, oversample factor μ (default μ = 2).

Procedure:
1. Partial decomp: top-(μr) eigenpairs (U_{μr}, Λ_{μr}) via randomized SVD
   (Halko–Martinsson–Tropp 2011 류). Cost O(n²(μr) + n(μr)²).
2. c_i = U_{μr}^⊤ Y_N (μr × C). s_i = (λ_i / (λ_i + ρ))² · ‖c_i‖² (multi-class
   합성, §5).
3. I\* := top-r of s_i within the μr pool.
4. Φ_tr, Φ_te 빌드 후 Algorithm 1.

Cost: preparation O(n²(μr) + n(μr)²) ≈ O(μrn² + μ²r²n) — μ = 2, r = 0.1n 이면
0.2n³ + 0.04n³ = 0.24n³ 로 O(n³) 보다 약 4× 빠르다. n = 10000 에서 의미
있음. TMC step 은 동일.

Eq. (9) 와의 관계: 진짜 supervised optimum I\* 는 *all-n* score 위에서
정의되는데, eq. (9) 의 score s_i 가 i > μr 영역에서 (λ_i/(λ_i+ρ))² 로 강하게
suppress 된다는 사실을 사용. λ 가 power-law decay 하면 (lrfshap.pdf Remark
E.3, p.21: λ_j ≤ C j^{−α}) i > μr 의 잔여가 (μr+1)^{−2α} 로 빠르게 떨어지고,
μ = 2, α ≥ 1 이면 약 25% 미만의 score mass 만 buried — 충분히 작은 truncation
error.

위험 / 한계: power-law decay 가 깨지면 (예: spectrum 이 plateau + tail)
truncation 이 큰 supervised score 를 놓칠 수 있다. 그리고 randomized SVD 의
정확도는 λ 의 spectral gap 에 의존 — eNTK 가 BERT 같은 사전훈련 모델에서
flat tail 을 갖는다는 보고 (Wei–Hu–Steinhardt 2022, lrfshap.pdf [B 참조]) 가
있어 실제로 spectral gap 이 약하면 μ 를 키워야 한다 (μ = 4–8). 또
Lanczos / randomized SVD 가 GPU 위에서 dense O(n³) eigendecomp 보다 느릴 수
있는 wall-clock crossover 가 n 에 따라 다름 — 실험에서 measure.

3 개를 한눈에:

| Algo | Score 사용 | Decomp cost | Theory 정당화 강도 | 실패 가능 |
|------|-----------|-------------|-------------------|----------|
| A1   | s_i (full) | O(n³)      | eq. (9) exact min | small-data noise overfit |
| A2   | LC/FC ratio | O(n³)     | §5.5 cross-dataset Pearson | numerical stability, ρ-independent |
| A3   | s_i (partial) | O(μrn²) | eq. (9) + power-law tail bound | spectrum plateau, randomized SVD 정확도 |

## 4. Related work mapping (4 트랙)

**Track A — kernel-side leverage scores (label 무관)**: Alaoui–Mahoney NeurIPS
2015, "Fast randomized kernel ridge regression with statistical guarantees"
(arxiv.org/abs/1411.0306) 이 ridge leverage score ℓ_i(ρ) = [K(K+ρI)⁻¹]_{ii}
를 column 선택 기준으로 사용. Bach JMLR 2013 ([B] in lrfshap §2) 의
effective dimension d_eff(ρ) = tr(K(K+ρI)⁻¹) = Σ λ_i/(λ_i+ρ) 도 같은 줄.
**우리 A1–A3 와 차이**: leverage score 는 label 을 안 보고 row/column 을 뽑는
Nyström-style sampling — eq. (9) 의 (u_i⊤ Y)² 항을 무시. A3 의 partial
decomp + supervised refine 은 leverage 가 아니라 *eigenvector* 를 뽑는 것.

**Track B — kernel-target alignment**: Cristianini et al. NeurIPS 2001 의 KTA
정의 후, Cortes–Mohri–Rostamizadeh JMLR 2012 "Algorithms for learning
kernels based on centered alignment" 가 centered KTA 로 base kernel 들의
combination 을 학습. **차이**: KTA 는 *kernel 자체* 를 (Y Y⊤ 와 align 하도록)
재구성. A1–A3 는 kernel 은 고정하고 그 spectral basis 안에서 *coordinate
selection*. Liu et al. AAAI 2014 / Neurocomputing 2016 "label-aware base
kernels" 가 비슷하게 ideal kernel 의 eigenfunction 을 extrapolate 하는데,
이쪽도 새 base kernel 을 만드는 거라서 우리와 다르다
(sciencedirect.com/science/article/abs/pii/S0925231215010796).

**Track C — eigenlearning / spectrum-dependent generalization**:
Canatar–Bordelon–Pehlevan, Nature Comm 2021, "Spectral bias and task-model
alignment explain generalization in kernel regression" (lrfshap §5.5 가 이미
인용) 이 task-model alignment α_i := (u_i⊤ Y)² / λ_i 와 mode-wise learnability
를 제시. Simon et al. Nature Comm 2023, "The eigenlearning framework: A
conservation law perspective" 가 후속 — 같은 score s_i 를 *예측 시점* 에서
mode-wise generalization 에 연결. **차이**: 이쪽은 generalization 분석 도구
이지, low-rank Shapley 안의 budget r 에서 어떤 mode 를 keep 할지를 다루지
않음. 우리는 그들의 score 를 *알고리즘적 selection criterion* 으로 변환.

**Track D — data Shapley + low-rank**: FreeShap (Wang et al. 2024a, lrfshap
baseline) 이 eNTK 로 retraining 을 우회한 것이 가장 가깝고, LRFShap 자체가
그 위에 low-rank 를 얹은 것. TRAK (Park et al. ICML 2023, arxiv.org/abs/2303.14186)
은 attribution 을 random projection + ensembling 으로 estimate — kernel
spectrum 을 직접 쓰지 않는다는 점에서 다름. DataInf (Kwon–Wu–Wang–Mohri
ICLR 2024, arxiv.org/abs/2310.00902) 는 LoRA layer 의 Hessian 에 closed-form
inverse — Shapley 가 아니고 influence function 계열. **차이**: A1–A3 는
LRFShap 안의 *low-rank basis 선택* 자체를 supervise 하는 직교적 contribution.

## 5. Practical issues

Multi-class y 처리부터 정리해야 한다. lrfshap §5.5 (p.7) 의 LC 정의는
binary 라 ‖P_r y‖² / ‖y‖² 인데, multi-class 에서는 Y_N ∈ {0, 1}^{n×C} 라
class 별 ‖P_r y_c‖² 를 합치는 방법이 두 가지다 — sum: Σ_c (u_i⊤ y_c)² (Frobenius
inner product), 또는 normalize-then-sum. lrfshap.pdf §D.2 (eq. (12)–(19))
의 scalar single-logit eNTK 를 쓰는 한 prediction 도 column 별 독립적이라
sum 형태가 자연스럽고, 이게 LRFShap 의 Algorithm 1 line 11 (Φ_te Φ_S⊤
... Y_S, Y_S 가 |S|×C) 과 호환된다. 따라서 본 plan 은 모든 algorithm 에서
s_i = (λ_i/(λ_i+ρ))² · ‖U_{:i}^⊤ Y_N‖_F² 로 통일.

Partial decomposition 의 cost 는 위 A3 에서 분석했지만, 실용적으로 PyTorch
의 torch.lobpcg 또는 SciPy 의 sparse.linalg.eigsh 가 GPU dense O(n³) 보다
빠른 crossover 가 n ≈ 5000 부근에 있다는 것이 일반적 경험치 — n = 10000 까지
가는 lrfshap §5.4 의 SST-2 / MNLI Pareto 실험 (Fig. 5, p.6) 에서 의미가 살아남.

마지막으로 TMC 안의 비용은 본질적으로 변하지 않는다는 점을 분명히 해두면,
이 plan 이 제안하는 변경은 모두 *preparation 단계* — Algorithm 1 line 3–4
에서 U_r 을 어떻게 고르냐의 문제고, line 5–14 의 TMC loop 와 그 cost
O(M(n²r² + nr³)) 는 그대로다. 즉 본 제안의 speedup 손익은 preparation 에 한정,
**accuracy 측 (ER) 손익은 모든 TMC iteration 에 누적**. 실험은 다음 iter 에서
SST-2, MNLI (LC 가 낮은 dataset) + AG News (LC 가 높아 saturated 인 baseline)
세 개로 ER vs r 곡선을 그려 직접 비교하는 것이 핵심.

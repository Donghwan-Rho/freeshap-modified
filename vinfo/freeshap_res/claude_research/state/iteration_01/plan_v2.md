# Iteration 01 — Label-aware eigenvector selection 탐색 (planner v2)

> v2 변경 요약: critique round 1 의 6 가지 발견 — (a) eq. (9) 페이지 인용
> 충돌, (b) Simon et al. venue 오기, (c) Liu et al. 저자명, (d) S = N
> 한정 가정 미명시, (e) ρ → 0 null mode 안전장치, (f) A2 novelty 약함 +
> falsifier #1 취약 — 을 본문에 직접 반영. 알고리즘 A1–A3 골격, 4-track
> mapping, 실험 후속 일정은 v1 그대로 유지.

## 1. Motivation

LRFShap 의 Algorithm 1 은 K = k(X_N, X_N) 의 best rank-r approximation,
즉 top-r by eigenvalue λ_i 로 feature Φ_tr = U_r Λ_r^{1/2} 를 만든 뒤
모든 prefix 의 ridge solve 를 r-dimensional space 에서 수행합니다
(lrfshap.pdf, Algorithm 1, p.4). §5.5 (pp.7–8) 가 보여주는 그림은
흥미로운데, 같은 rank budget 에서도 SST-2 / MNLI 는 LC(r) 가 낮고 ER 이
1 근처에서 흔들리는 반면, AG News / MR 은 LC(r) 가 높고 ER 이 안정적으로
1 에 붙습니다. cross-dataset correlation 만 놓고 보면 FC(r) 는 ER 을
거의 설명하지 못하고 (Fig. 6(b), Pearson(ER, FC) 가 r = 5–15% 에서 음수로
떨어짐), LC(r) 와 LC/FC 만 일관되게 0.7 이상으로 올라갑니다. 이 관찰을
그대로 받아들이면, **rank 가 같다면 같은 budget 으로 LC(r) 또는 LC/FC 를
더 끌어올리는 방향이 ER 을 직접 끌어올린다** 는 가설이 자연스럽습니다.

사용자가 던진 핵심 질문은 그래서 이렇습니다. (i) top-r by λ 가 아니라 LC
또는 LC/FC 를 maximize 하도록 r 개의 eigenvector 를 *재선택* 하면 ER 이
올라가는가, (ii) 그 류의 prior work 가 있는가, (iii) maximizer 를 어떻게
효율적으로 계산하는가. 이 plan 은 (i)–(iii) 에 대해 알고리즘 3 개와 4
트랙 prior work mapping 을 제시합니다.

## 2. Theoretical backing — eq. (9) 가 직접 주는 supervised score

LRFShap 본문 §5.5 (lrfshap.pdf p.7, eq. (9)) — 그리고 같은 식이 appendix
§A (p.12, eq. (8)–(9)) 에서 더 자세히 재기술됩니다 — 는 in-sample
predictor gap 을 정확한 spectral identity 로 적어둡니다:

  ‖f_ρ^FNTK(X_N) − f_ρ^LRNTK(X_N)‖² = Σ_{i>r} (λ_i / (λ_i + ρ))² · (u_i⊤ Y_N)²
                                    ≤ (1 − LC(r)) · ‖Y_N‖².

오른쪽 상한이 (1 − LC) 로 나오긴 하지만 진짜 동등식은 왼쪽입니다. 그리고
왼쪽은 **선택된 인덱스 집합 I ⊆ {1, …, n} 의 함수** 입니다 — 단, 두 가지
가정 위에서 그렇습니다. (a) 우리가 minimize 하는 V 는 자유로운 r-차원
subspace 가 아니라 **K(X_N) 의 eigendecomposition U Λ U⊤ 가 미리 고정해 둔
basis 안에서 r 개를 고르는 인덱스 집합** I 로 한정합니다. 만약 V 를
임의의 r-차원 subspace 로 풀어 주면 minimizer 가 "PCA-of-Y" 같은 다른
basis 로 옮겨가서 알고리즘 자체가 달라지므로, 본 plan 은 fixed-basis
selection 만을 다룹니다. (b) 이 식 자체는 **S = N (전체 학습셋)** 위의
predictor gap 입니다. 즉 임의의 I 에 대해

  E(I) := Σ_{i∉I} s_i,    s_i := (λ_i / (λ_i + ρ))² · (u_i⊤ Y_N)²

를 정의하면, |I| = r 이라는 budget 하에서 E(I) 를 최소화하는 I 는

  **I\* = top-r by s_i, i.e., supervised score s_i = (λ_i/(λ_i+ρ))² · (u_i⊤Y_N)²**

입니다. 반면 LRFShap 이 쓰는 것은 top-r by λ_i. 두 선택의 차이는 ρ 의
자리에서 나옵니다. ρ → ∞ 면 (λ/(λ+ρ))² 가 λ² 에 비례해 수축하므로 큰
λ 가 그대로 이깁니다 — 그래서 top-r by λ 가 ρ 가 클 때 합리적입니다.
ρ → 0 면 필터가 1 로 saturate 하고 (u_i⊤ Y)² 만 남아서 score 가 *순수
label alignment* 가 됩니다. LRFShap 은 ρ = 1e-3 같은 작은 값에서 돌리니
후자에 가깝고, 그래서 supervised selection 의 이득이 가장 클 가능성이
있습니다.

S = N 한정성에 대한 명시적 단서. TMC 가 거치는 임의 subset S ⊊ N 에서는
‖f_S^FNTK(X_S) − f_S^LRNTK(X_S)‖² 같은 *coalition-내부* gap 이 진짜
신경 써야 할 양이고, K(X_S) 의 spectrum 은 K(X_N) 의 spectrum 과 다릅니다.
본 plan 은 score s_i 를 매 coalition 마다 재계산하지 않고 **K(X_N) 에서
한 번 계산한 뒤 모든 |S| ≤ n permutation 에 동일하게 사용**합니다 — 즉
"S = N 의 minimizer 가 모든 |S| < n coalition 에 대해서도 average-case 로
유의한 개선을 준다" 는 것이 우리가 짊어진 가정입니다. 이 가정은 식으로는
보장되지 않으니 다음 iter 의 실험에서 verify 해야 합니다 (synthetic 에서
|S| = 0.1n, 0.3n, 0.7n, n 별로 ER gap 측정).

수식 단위로 차이는 명료합니다. top-r by λ 는 식 (9) 의 *상한* (1 − LC) 만
줄이는 반면, supervised top-r 은 *equality* 인 LHS 자체를 줄입니다. Prop
4.2 (p.5) 의 Shapley value 안정성 bound 가 ‖K̃ − K‖₂ 에 의존하긴
하지만, 거기서 ‖K̃ − K‖₂ 자리는 사실 in-sample predictor gap 의 proxy 일
뿐이고 (Cortes et al. 2010 의 원래 bound 도 predictor gap 을 직접 씁니다),
우리가 minimize 하는 양과 알고리즘이 신경 쓰는 양이 일치하는 쪽은
supervised selection 입니다. 다만 critique 가 지적한 대로 Prop 4.2 의
원문 statement (operator norm 인지 nuclear norm 인지) 는 plan 단계에서
재확인하지 못했으니, 다음 iter 의 reading 단계에서 한 줄 인용으로
보강합니다.

## 3. Algorithmic proposals (단순 → 정교)

### Algorithm A1 — Greedy supervised top-r by s_i (baseline 변형, 본 제안의 main contribution)

Input: full eigendecomp K = U Λ U⊤, labels Y_N, ridge ρ, budget r,
minimum eigenvalue floor ε (default ε = 10·ρ).

Procedure:
1. Compute c_i := u_i⊤ Y_N for i = 1, …, n (cost O(n²C) 한 번).
2. Compute s_i = (λ_i / (λ_i + ρ))² · c_i² (binary case) or Σ_c c_{i,c}²
   (multi-class, §5).
3. **Null-mode floor**: λ_i < ε 인 인덱스를 candidate pool 에서 제외.
4. I\* := indices of top-r values of s_i (within the surviving pool).
5. Build Φ_tr := U_{I\*} Λ_{I\*}^{1/2}, Φ_te := K_te U_{I\*} Λ_{I\*}^{−1/2}.
6. Run LRFShap Algorithm 1 line 5–14 unchanged with these Φ_tr, Φ_te.

Cost: 한 번의 추가 비용은 O(n²C) for c_i + O(n) for sorting. 본 LRFShap
의 preparation cost 가 이미 O(n³) (eigen decomp, lrfshap.pdf Table 2,
p.23) 이므로 **preparation 단계에서는 dominated**. TMC iteration 안의
cost 는 r 이 동일하다면 완전히 동일 (line 11, O(n²r² + nr³) per
permutation).

Eq. (9) 와의 관계: §2 에서 도출한 minimizer 를 그대로 사용. predictor
gap 의 exact spectral identity 를 minimize.

위험 / 한계: (i) **null mode 위험**. ρ → 0 한계에서는 (λ_i/(λ_i+ρ))² 가
모든 i 에서 1 로 saturate 되어 score 가 λ-independent 가 되고, λ_i ≈ 0
인 null direction (K 가 effective low-rank 이거나 BERT eNTK 의 flat tail
영역) 도 selection 에 들어와 Φ_S⊤Φ_S + ρI 의 conditioning 이 망가질 수
있습니다. 본 procedure 의 step 3 floor ε = 10·ρ 는 (λ_i/(λ_i+ρ))² ≥
(10/11)² ≈ 0.83 인 mode 만 살리는 효과라 score 의 의미가 보존됩니다.
실험에서 ε ∈ {ρ, 10ρ, 100ρ} 로 grid 해서 sensitivity 측정. (ii) TMC 의
marginal gain 분포가 변해 early-stop ratio 가 달라지고 wall-clock speedup
이 LRFShap 의 figure 4 (p.6) 와 다른 trend 를 보일 수 있음. (iii) SST-2
/ MNLI 의 1% selection 에서 LRFShap 이 baseline 을 *초과* 한 것 (Fig. 2,
p.6) 은 unsupervised low-rank truncation 이 우연히 noise filter 역할을
했을 가능성이 있는데, supervised selection 은 noise 까지 같이 fitting
해버려서 이 "regularizer 효과" 를 잃을 수 있음 (falsifier #2 참조).

### Algorithm A2 — α_i-based selection (ablation, novelty 제한)

Critique round 1 §1 의 평가를 수용해 A2 의 위치를 재조정합니다. r_i =
(u_i⊤Y)² / (λ_i ‖Y‖²) 는 Canatar–Bordelon–Pehlevan 2021 (Nat. Comm.) 의
task-model alignment α_i = (u_i⊤Y)²/λ_i 와 *같은 양* 이고, A2 의 실질적
contribution 은 "α_i 를 LRFShap 의 selection criterion 으로 그대로 옮긴
것" 입니다. 즉 본 plan 의 main novelty 는 A1 (s_i 가 ρ-aware spectral
filter 를 포함) 과 A3 (partial decomp + supervised refinement) 에 있고,
A2 는 "α_i selection 이 s_i selection 대비 어디서 무너지는가" 를 보여주는
**ablation** 으로 위치시킵니다.

Input: 위와 동일 + null-mode floor ε.

Procedure:
1. c_i² = (u_i⊤ Y_N)², 각 i 에 대해 ratio r_i := c_i² / (λ_i · ‖Y_N‖²) 계산.
2. **Floor**: λ_i < ε 인 인덱스 제외 (A2 는 분모에 λ_i 가 들어가 floor 가
   더 절실 — falsifier #1 참조).
3. Top-r 선택 → Φ_tr, Φ_te 빌드 → Algorithm 1.

Cost: A1 과 동일 O(n²C) preparation 추가.

Eq. (9) 와의 관계: §5.5 의 cross-dataset 분석이 정성적으로 motivate.
ρ → 0 한계에서 s_i 를 λ_i 로 한 번 더 나눈 양에 해당하지만, ρ-independent
라 ρ 가 작지 않은 영역에서는 정당화가 약합니다.

위험 / 한계: 큰 λ 의 eigenvector 를 제외하면 line 11 ridge regression 의
features 가 작은 norm 에 몰려 numerical error 가 키워질 수 있음. 또 r_i
는 ρ-independent 라 ρ 가 크지 않은 영역에서만 정당화. RTE 같이 n 이 작고
LC/FC 가 들쭉날쭉한 dataset (Fig. 12, p.13) 에서는 ratio 가 unstable.
더 결정적으로, 다음 §6 falsifier #1 이 보여주는 small-λ degenerate
시나리오에서 A2 는 의도적으로 망가지므로, 본 plan 은 A2 를 ER 향상
주장의 근거로 쓰지 않고, A1/A3 의 baseline 으로만 사용합니다.

### Algorithm A3 — Partial-decomp + supervised refinement (정교)

가장 큰 우려는 A1, A2 가 모두 "full eigendecomp 이미 있다" 를 전제로
한다는 점입니다. n = 10000 일 때 O(n³) eigen decomp 자체가 비싸집니다
(lrfshap.pdf Table 2, p.23: preparation O(n³) 가 그대로 살아있음). A3 는
**top-2r by λ 까지만 partial Lanczos / randomized SVD 로 뽑은 뒤, 그
안에서 supervised top-r 을 재선택** 합니다.

Input: K, Y_N, ρ, r, oversample factor μ (default μ = 2), floor ε.

Procedure:
1. Partial decomp: top-(μr) eigenpairs (U_{μr}, Λ_{μr}) via randomized
   SVD (Halko–Martinsson–Tropp 2011 류). Cost O(n²(μr) + n(μr)²).
2. c_i = U_{μr}^⊤ Y_N (μr × C). s_i = (λ_i / (λ_i + ρ))² · ‖c_i‖² (multi-class
   합성, §5).
3. **Floor**: λ_i < ε 인 인덱스 제외 (partial pool 안에서도 적용).
4. I\* := top-r of s_i within the surviving μr pool.
5. Φ_tr, Φ_te 빌드 후 Algorithm 1.

Cost: preparation O(n²(μr) + n(μr)²) ≈ O(μrn² + μ²r²n) — μ = 2, r = 0.1n
이면 0.2n³ + 0.04n³ = 0.24n³ 로 *flop 기준* O(n³) 보다 약 4× 빠름. n =
10000 에서 의미 있음. 다만 critique §3(f) 가 지적한 대로 GPU 위에서는
dense O(n³) eigh 가 매우 잘 최적화돼 있어 wall-clock 비교는 randomized
방식이 항상 빠르다고 단정할 수 없습니다 — wall-clock crossover 는 다음
iter 실험에서 직접 측정. TMC step 은 동일.

Eq. (9) 와의 관계: 진짜 supervised optimum I\* 는 *all-n* score 위에서
정의되는데, eq. (9) 의 score s_i 가 i > μr 영역에서 (λ_i/(λ_i+ρ))² 로
강하게 suppress 된다는 사실을 사용. λ 가 power-law decay 하면
(lrfshap.pdf Remark E.3, p.21: λ_j ≤ C j^{−α}) i > μr 의 잔여가
(μr+1)^{−2α} 로 빠르게 떨어지고, μ = 2, α ≥ 1 이면 약 25% 미만의 score
mass 만 buried — 충분히 작은 truncation error.

위험 / 한계: power-law decay 가 깨지면 (예: spectrum 이 plateau + tail)
truncation 이 큰 supervised score 를 놓칠 수 있음. 그리고 randomized SVD
의 정확도는 λ 의 spectral gap 에 의존 — eNTK 가 BERT 같은 사전훈련 모델
에서 flat tail 을 갖는다는 보고가 있어 실제로 spectral gap 이 약하면
μ 를 키워야 함 (μ = 4–8).

3 개를 한눈에:

| Algo | Score 사용 | Decomp cost | Theory 정당화 강도 | Novelty 위치 | 실패 가능 |
|------|-----------|-------------|-------------------|-------------|----------|
| A1   | s_i (full) | O(n³) | eq. (9) exact min | **main**: ρ-aware spectral filter | small-data noise overfit |
| A2   | α_i = c_i²/λ_i | O(n³) | §5.5 cross-dataset Pearson | **ablation**: Canatar α_i 를 selection 으로 사용 | small-λ degenerate (falsifier #1) |
| A3   | s_i (partial) | O(μrn²) flop | eq. (9) + power-law tail bound | **main**: partial decomp + supervised refinement | spectrum plateau, randomized SVD wall-clock |

## 4. Related work mapping (4 트랙)

**Track A — kernel-side leverage scores (label 무관)**: Alaoui–Mahoney
NeurIPS 2015, "Fast randomized kernel ridge regression with statistical
guarantees" (arxiv.org/abs/1411.0306) 이 ridge leverage score
ℓ_i(ρ) = [K(K+ρI)⁻¹]_{ii} 를 column 선택 기준으로 사용. Bach JMLR 2013
(lrfshap §2 [B] 참조) 의 effective dimension d_eff(ρ) = tr(K(K+ρI)⁻¹) =
Σ λ_i/(λ_i+ρ) 도 같은 줄.

**우리 A1–A3 와 차이 — supervised refinement of ridge leverage**: ridge
leverage 의 mode-wise 형태는 λ_i/(λ_i+ρ) 로, 우리 score s_i 의 *square
root × label term 제외* 에 정확히 대응합니다. 즉

  s_i = ( λ_i / (λ_i + ρ) )² · (u_i⊤ Y)² = (mode-i ridge leverage)² · (u_i⊤ Y)²

로 factorize 됩니다. 이 관점에서 본 plan 의 핵심은 "ridge leverage score
의 supervised refinement" — leverage 가 column sampling (Nyström) 에 쓰는
*label 무관* 가중을 우리는 *label-aligned* term 과 곱해 eigenvector
selection 에 씁니다. A1–A3 는 fixed kernel basis 위의 supervised
*coordinate selection* 이지 새 basis 를 학습하지 않습니다. 이 한 줄이
기존 leverage score literature 와의 자리매김을 깔끔하게 정리합니다.

**Track B — kernel-target alignment**: Cristianini et al. NeurIPS 2001
의 KTA 정의 후, Cortes–Mohri–Rostamizadeh JMLR 2012 "Algorithms for
learning kernels based on centered alignment" 가 centered KTA 로 base
kernel 들의 combination 을 학습. **차이**: KTA 는 *kernel 자체* 를
(Y Y⊤ 와 align 하도록) 재구성. A1–A3 는 kernel 은 고정하고 그 spectral
basis 안에서 *coordinate selection*. Wang, Q., Zhang, K., Chen, Z., Wang,
D., Jiang, G., Marsic, I. AAAI 2014 / Neurocomputing 2016 "Improving
Semi-Supervised Target Alignment via Label-Aware Base Kernels" /
"Enhancing semi-supervised learning through label-aware base kernels"
(ojs.aaai.org/index.php/AAAI/article/view/8958, sciencedirect.com/science/
article/abs/pii/S0925231215010796) 가 directive 가 호명한 "Liu et al."
의 실제 저자입니다 (critique §2 셋째 발견에 따라 정정). 이쪽은 ideal
kernel 의 eigenfunction 을 unlabeled data 위로 extrapolate 해 *새로운 base
kernel* 을 만드는 흐름이라, kernel 을 고정한 채 eigenvector index 만
재선택하는 우리와 방향이 반대입니다.

**Track C — eigenlearning / spectrum-dependent generalization**:
Canatar–Bordelon–Pehlevan, Nature Comm 2021, "Spectral bias and task-model
alignment explain generalization in kernel regression" (lrfshap §5.5 가
이미 인용) 이 task-model alignment α_i := (u_i⊤ Y)² / λ_i 와 mode-wise
learnability 를 제시. **Simon, Dickens, Karkada, DeWeese, TMLR 2023, "The
Eigenlearning Framework: A Conservation Law Perspective on Kernel Ridge
Regression and Wide Neural Networks"** (arXiv 2110.03922,
openreview.net/forum?id=FDbQGCAViI) 가 후속 — 같은 spectral 구조를
*예측 시점* 에서 mode-wise generalization 에 연결합니다. (v1 의 "Nature
Comm 2023" 표기는 오기였고 실제 venue 는 TMLR 2023 — critique §2 둘째
발견 반영.)

**차이**: 이쪽 라인은 generalization *분석* 도구이지, low-rank Shapley
안의 budget r 에서 어떤 mode 를 keep 할지를 다루지 않습니다. A2 는
Canatar 의 α_i 를 그대로 selection 으로 옮긴 것이므로 novelty 가 제한적
이고 (critique §1 셋째 단락 반영), A1 (s_i 가 ρ-aware), A3 (partial
decomp 로 cost 까지 잡음) 가 본 plan 의 main contribution 입니다.

**Track D — data Shapley + low-rank**: FreeShap (Wang et al. 2024a,
lrfshap baseline) 이 eNTK 로 retraining 을 우회한 것이 가장 가깝고,
LRFShap 자체가 그 위에 low-rank 를 얹은 것. TRAK (Park et al. ICML 2023,
arxiv.org/abs/2303.14186) 은 attribution 을 random projection +
ensembling 으로 estimate — kernel spectrum 을 직접 쓰지 않는다는 점에서
다름. DataInf (Kwon–Wu–Wang–Mohri ICLR 2024, arxiv.org/abs/2310.00902)
는 LoRA layer 의 Hessian 에 closed-form inverse — Shapley 가 아니고
influence function 계열. **차이**: A1–A3 는 LRFShap 안의 *low-rank basis
선택* 자체를 supervise 하는 직교적 contribution.

## 5. Practical issues

Multi-class y 처리부터 정리해야 합니다. lrfshap §5.5 (p.7) 의 LC 정의는
binary 로 ‖P_r y‖² / ‖y‖² 로 적혀 있는데, multi-class 에서는 Y_N ∈ {0,
1}^{n×C} 라 class 별 ‖P_r y_c‖² 를 합치는 방법이 두 가지입니다 — sum:
Σ_c (u_i⊤ y_c)² (Frobenius inner product), 또는 normalize-then-sum.
lrfshap.pdf §D.2 (eq. (12)–(19)) 의 scalar single-logit eNTK 를 쓰는 한
prediction 도 column 별 독립적이라 sum 형태가 자연스럽고, 이게 LRFShap 의
Algorithm 1 line 11 (Φ_te Φ_S⊤ ... Y_S, Y_S 가 |S|×C) 과 호환됩니다.
따라서 본 plan 은 모든 algorithm 에서 s_i = (λ_i/(λ_i+ρ))² · ‖U_{:i}^⊤
Y_N‖_F² 로 통일하고, 이는 lrfshap §5.5 binary 정의의 multi-class
Frobenius extension 입니다 (다음 iter reading 단계에서 §5.5 LC 정의 한
줄을 그대로 옮겨 적어 명시화).

Partial decomposition 의 cost 는 위 A3 에서 분석했지만, 실용적으로
PyTorch 의 torch.lobpcg 또는 SciPy 의 sparse.linalg.eigsh 가 GPU dense
O(n³) 보다 빠른 crossover 가 n ≈ 5000 부근에 있다는 것이 일반적 경험치
— n = 10000 까지 가는 lrfshap §5.4 의 SST-2 / MNLI Pareto 실험 (Fig. 5,
p.6) 에서 의미가 살아남. Wall-clock 측정은 다음 iter 실험에서.

마지막으로 TMC 안의 비용은 본질적으로 변하지 않는다는 점을 분명히 해두면,
이 plan 이 제안하는 변경은 모두 *preparation 단계* — Algorithm 1 line
3–4 에서 U_r 을 어떻게 고르냐의 문제고, line 5–14 의 TMC loop 와 그 cost
O(M(n²r² + nr³)) 는 그대로입니다. 즉 본 제안의 speedup 손익은 preparation
에 한정, **accuracy 측 (ER) 손익은 모든 TMC iteration 에 누적**. 실험은
다음 iter 에서 SST-2, MNLI (LC 가 낮은 dataset) + AG News (LC 가 높아
saturated 인 baseline) 세 개로 ER vs r 곡선을 그려 직접 비교하는 것이
핵심입니다.

## 6. Falsifier 시나리오 (다음 iter 실험에 명시 반영)

**Falsifier #1 (재정비) — degenerate small-λ + large-(u⊤Y) mode**.
v1 falsifier 가 "small-λ noise 를 supervised score 가 잘못 뽑는다" 정도로
정성적이었다면, critique §4 가 정량 시나리오를 잡아 줬으니 받아들여
sharp 하게 재정의합니다. spectrum 이 λ_1 ≥ … ≥ λ_n 으로 빠르게 떨어지다
tail 끝에 λ_k ≈ 1e-8 mode 가 하나 있고 우연히 (u_k⊤Y)² 가 ‖Y‖²/n 수준
으로 큰 경우, A2 의 r_k = (u_k⊤Y)²/(λ_k‖Y‖²) 는 λ_k 가 분모라 ≈ 1e8
규모로 *터집니다* — A2 는 이 mode 를 1순위로 뽑고 Φ_tr 의 k-th column
이 √λ_k = 1e-4 norm noise 로 들어가 ridge solve 의 effective conditioning
이 망가집니다. A1 은 같은 시나리오에서 score s_k = (1e-8/(1e-8+1e-3))² ·
(u_k⊤Y)² ≈ 1e-10 · ‖Y‖²/n 이라 자동 배제 — falsifier 가 A1 ↔ A2 를
*분리* 합니다. 이게 plan_v2 가 A2 를 ablation 으로 강등하는 결정적 이유.
검증: synthetic spectrum (power-law + 인위적 tail spike) 위에서 A1, A2,
top-r-by-λ 의 ER 을 측정. tail spike 가 강해질수록 A2 의 ER 만 baseline
아래로 떨어지면 confirm. (A1 도 ε floor 가 없으면 ρ → 0 한계에서
유사하게 무너지니, ε ∈ {ρ, 10ρ, 100ρ} grid 도 같은 실험에 포함.)

v1 의 더 약한 "small-λ + large-(u⊤Y) noise" 표현은 A1/A2 를 한꺼번에
공격하는 형태라 falsifier 로서 sharpness 가 부족했습니다. v2 의 재정비
버전은 (a) A2 만 정량적으로 무너뜨리고 (b) A1 의 floor 메커니즘을
같은 setup 안에서 검증할 수 있어, 두 알고리즘을 *분리* 해서 평가합니다.

**Falsifier #2 — Y 가 noise-aligned with low-eigenvalue modes** (label
noise 시나리오). label 일부가 random flip 으로 망가져 있을 때, 큰
(u_i⊤Y)² 가 사실은 *noise component* 의 alignment 일 수 있습니다.
supervised selection 은 이 noise mode 를 "유의한 supervised score" 로
선택해서 학습하지만, unsupervised top-r by λ 는 큰 λ 만 보니까 *우연히*
noise mode 를 거를 수 있습니다. lrfshap.pdf Fig. 2, p.6 에서 SST-2 1%
selection 이 LRFShap 이 baseline 을 *초과* 한 것이 정확히 이 메커니즘일
가능성 — low-rank truncation 이 implicit denoiser 였다면 supervised
selection 은 그 효과를 잃습니다. 검증: SST-2 의 train label 5–20% 를
random flip 시켜 놓고 A1 vs top-r-by-λ 의 ER 을 비교. flip rate 가
높을수록 A1 의 ER 이 baseline 아래로 떨어지면 partial confirm.

이 두 falsifier 와 §2 의 S = N → S ⊊ N transfer 가정 검증, 그리고 A3 의
randomized SVD wall-clock crossover 측정 — 이 네 가지가 다음 iter 실험
design 의 obligatory 항목입니다.

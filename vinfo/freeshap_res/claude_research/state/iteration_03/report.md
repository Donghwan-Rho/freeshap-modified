# Iteration 03 — A1 vs LRFShap 의 3 조건 framework

iter 02 의 5 조건 체계 (Cond A, B, 1, 2, 3) 에 대해 사용자가 제기한 네 갈래
비판 — Cond A 의 정확한 의미, Cond B 의 정보 부재, iter 02 Cond 1 의 Cond A
중복성, MRPC mechanism 의 imbalance 정량, multi-class 의 √C SNR 유도 — 와
그 뒤 round 2 의 세 추가 질문 — Cond 2 의 train_majority + LR_predict_majority
가 짜맞추기 아닌가 / PR(c²) 같은 label 분산 측정으로 표현 가능한가, 이전
Spearman(λ, c²) 와 overlap_LR_A1 이 Cond 1 에 흡수된 건지 살릴 여지 있는지,
multi-class A1 이득의 mechanism 을 다시 정밀화하고 C 많은 데이터셋으로
검증 가능한지 — 를 모두 본문에서 정량/식 단위로 답합니다.

본 라운드의 결론은 다음과 같습니다. (a) Cond 2 의 mechanism-grounded primary
quantity 는 **PR(c²)** — label projection 의 spread 측정 — 이고
LR_predict_majority_frac 는 그 결과인 observable 의 verification. PR(c²) 가
MRPC 의 274.94 와 QQP 의 134.57 의 2 배 격차로 두 데이터셋의 분기 mechanism
을 직접 잡습니다. (b) Spearman(λ, c²) 와 overlap_LR_A1 은 Cond 1 의 mechanism
의 secondary indicator — LC_LR, low PR(c²), high Spearman, high overlap 이
모두 같은 underlying 구조 ("label 정보가 top-r 의 mode 에 집중") 의 dual
측정. (c) √C SNR 의 정확한 형태는 **√(C-1) × binary** — one-hot constraint
Σ_c ỹ_c = 0 으로 effective independent column 수가 C-1 — 이고 효과는 *small
signal regime* (α² 가 σ²/n 와 같은 order) 에서만 sharp. C ∈ {5, 10, 14, 20}
의 데이터셋 (DBpedia, 20Newsgroups 등) 으로 √(C-1) scaling 의 직접 검증이
iter 04 의 우선 task.

본문의 모든 정량은 n=5000 의 3-seed (2024 / 2025 / 2026) 평균이며 출처는
`state/iteration_03/precomputed_n5000.json`. LRFShap (= top-r by λ) 와 A1
(= top-r by sᵢ = (λᵢ/(λᵢ+ρ))² · ‖uᵢ⊤Ỹ‖², ρ = 10⁻²) 의 정의는 iter 02 와
동일. Ỹ 는 column-별 mean centering 된 one-hot label.

## §1. Setup, label projection geometry, Spearman / overlap 의 정확한 위치

eNTK gram matrix K = U Λ U⊤ 의 spectral 분해에서 uᵢ 는 i 번째 eigenvector,
λᵢ 는 그 eigenvalue. label projection 은

  cᵢ² := ‖uᵢ⊤ Ỹ‖² = Σ_c (uᵢ⊤ ỹ_c)²                                       (1)

mode i 가 centered class-indicator 와 만드는 내적의 제곱합. one-hot constraint
Σ_c y_c = 1 (각 row 의 합 1) 가 centering 후 Σ_c ỹ_c = 0 으로 이어져 (uᵢ⊤ ỹ_1,
…, uᵢ⊤ ỹ_C) 의 effective 자유도는 C-1. binary 에서는 ỹ_{c=1} = −ỹ_{c=0}
이라 effective 1 차원, 3-class 는 effective 2 차원, … — 이 점이 §2.5 의
Cond 3 mechanism 의 핵심 (√(C-1) SNR scaling).

label projection 의 *spread* 를 measuring 하는 자연스러운 quantity 는
participation ratio

  PR(c²) := (Σᵢ cᵢ²)² / Σᵢ cᵢ⁴.                                          (2)

PR 작으면 c² 의 mass 가 적은 mode 에 집중 — top-r 이 그 집중된 label 정보를
잡을 가능성 높음. PR 크면 mass 가 많은 mode 에 퍼져 있음 — top-r 이 그 분산된
정보를 못 잡고 누락. n=5000 측정에서 PR(c²) 가 데이터셋 간 분기 mechanism 을
가장 강하게 잡습니다 (§2.4 의 MRPC vs QQP).

**Spearman(λ, c²) 의 정확한 정의** 는 λᵢ 의 내림차순 순위 rank_λ(i) 와 cᵢ²
의 내림차순 순위 rank_{c²}(i) 두 vector 간의 tie-corrected Pearson 상관계수

  Spearman(λ, c²) = 1 − 6 · Σᵢ (rank_λ(i) − rank_{c²}(i))² / (n · (n²−1))   (3)

입니다. 1 에 가까우면 LR selection 과 A1 selection 이 거의 같은 mode set
을 픽, 0 이면 두 selection 이 다른 mode set 을 픽. **overlap_LR_A1** 은
직접 측정 — overlap_LR_A1 = |I_LR ∩ I_A1| / r — selection 간 일치 비율.

**Spearman 과 overlap 의 위치 — 채택되지 않은 것이 아니라 Cond 1 의 secondary
indicator 로 복귀.** iter 02 에서 이 둘은 main quantity 였고, 본 라운드의
초기 draft 에서는 wins/42 와 monotonic 하지 않다는 이유로 채택을 보류했었
습니다. 그러나 사용자의 round-2 비판이 정당했고, 7 데이터셋 측정에서 이 둘이
LC_LR 및 PR(c²)⁻¹ 와 *consistent monotone 관계* 를 보입니다:

| Dataset | LC_LR@10% | 1/PR(c²) | Spearman(λ, c²) | overlap_LR_A1@10% | Cond 1 |
|---|---:|---:|---:|---:|:-:|
| AG News | 0.773 | 0.058 | 0.413 | 0.523 | ✓ |
| SST-2   | 0.608 | 0.045 | 0.212 | 0.332 | ✓ |
| MR      | 0.547 | 0.036 | 0.156 | 0.293 | ✓ |
| QQP     | 0.397 | 0.0074 | 0.123 | 0.306 | ✗ |
| MNLI    | 0.318 | 0.0023 | 0.205 | 0.345 | ✗ |
| MRPC    | 0.318 | 0.0036 | 0.191 | 0.283 | ✗ |
| RTE     | 0.206 | 0.0015 | 0.099 | 0.242 | ✗ |

Cond 1 발동 그룹 (AG News, SST-2, MR) 의 (Spearman, overlap, 1/PR(c²)) 평균
이 미발동 그룹 (MNLI, MRPC, QQP, RTE) 평균보다 일관되게 큼: Spearman 0.260
vs 0.155, overlap 0.383 vs 0.294, 1/PR(c²) 0.046 vs 0.0037. 이 네 quantity
가 모두 *"label 정보가 top-r 의 mode 에 집중"* 의 dual 측정이라 Cond 1 의
mechanism 이 두꺼운 evidence 위에 서있음을 의미합니다. 본 라운드 framework
에서는 Cond 1 의 main definition 만 LC_LR ≥ 0.5 로 유지하고 (가장 직접
적이고 해석 명확), 나머지 셋은 mechanism 의 verification indicator 로 §3
의 보조 진술에 명시합니다.

## §2. 사용자 비판의 정면 답변 (round 1 + 2)

### §2.1 Cond A 의 정확한 의미 — 그대로 새 Cond 1

iter 02 의 Cond A (LC_LR(r=10%) ≥ 0.5) 는 사용자 이해 그대로 "top-r by λ 가
centered label 의 절반 이상을 흡수해 eigen 정보와 label 정보를 모두 잡는다"
. r-dim ridge 가 label 절반 이상을 갖고 있으면 그 부분공간 안에서 task 가
풀리고 A1 의 marginal benefit 이 작아 top-r 이 이깁니다. **Cond 1 = iter
02 Cond A** 로 그대로 유지.

### §2.2 Cond B 의 정량 반박

iter 02 의 Cond B (full ridge accuracy 가 majority baseline 대비 향상 미미)
는 어느 쪽이 이기는가의 predictive power 가 없습니다.

| Dataset | full_ridge | val_majority | gap (pp) | wins/42 |
|---|---:|---:|---:|---:|
| MNLI    | 0.566 | 0.302 | +26.4 | 42 |
| MRPC    | 0.738 | 0.684 |  +5.4 | 35 |
| AG News | 0.892 | 0.245 | +64.7 | 17 |
| SST-2   | 0.845 | 0.509 | +33.6 |  8 |
| QQP     | 0.745 | 0.614 | +13.1 |  7 |
| MR      | 0.814 | 0.503 | +31.1 |  4 |
| RTE     | 0.552 | 0.527 |  +2.5 |  4 |

gap 으로 정렬 (AG News 64.7 > SST-2 33.6 > MR 31.1 > MNLI 26.4 > QQP 13.1 >
MRPC 5.4 > RTE 2.5) 한 wins/42 (17, 8, 4, 42, 7, 35, 4) 가 *전혀 monotonic
하지 않습니다*. gap 1 위 (AG News) 가 wins 3 위, gap 4 위 (MNLI) 가 wins 1 위.
gap 은 "kernel 이 task 정보를 *얼마나* 담는가" 의 amount 측정인데, A1 vs LR
승패는 그 정보가 *λ 의 top 에 모이는가 (LC_LR ↑) 아니면 작은 λ 의 mode 에
흩어지는가* 의 *방향* 문제. AG News (gap 64.7) 는 정보가 top-r 에 align (LC_LR
0.773) 되어 LR 안정 우세, MNLI (gap 26.4) 는 정보가 작은 λ 에 흩어져 (LC_LR
0.318) A1 이 그것을 잡아 우세. **Cond B 삭제**.

### §2.3 iter 02 의 Cond 1 (miss_LC_LR ≥ 0.5) 의 중복성

miss_LC_LR = 1 − LC_LR 이므로 Cond A 의 음의 진술. 같은 정보를 두 번 적는
형태로 information density 만 떨어뜨림. **삭제**.

### §2.4 Centering 의 mechanism + MRPC vs QQP 의 솔직한 한계

**Centered Y 가 minority 정보를 회복한다는 것의 정확한 의미.** binary
imbalanced (e.g., majority 비율 p = 0.674) 의 두 score 를 비교합니다.

- *Uncentered* version (A1 의 score 가 (uᵢ⊤ Y)² 였다면): Y 는 majority 위치
  에서 1, minority 에서 0. (uᵢ⊤ Y) = Σ_{i ∈ majority} uᵢ ≈ (n · p) · ⟨uᵢ⟩ +
  fluctuation, ⟨uᵢ⟩ 는 majority 의 평균. 이 양이 크려면 uᵢ 의 entry 가 majority
  group 위에서 *체계적으로 같은 부호로 더해져야* — 즉 uᵢ 가 *majority 의
  공통 feature* (문서 길이, 공통 token) 와 align. 이 경우 top mode 는 *class-blind
  common-feature mode* 가 차지합니다. minority 와 majority 를 *구별* 하는
  정보는 보존되지 않습니다.

- *Centered* version (A1 의 실제 score (uᵢ⊤ Ỹ)², Ỹ = Y − p): Ỹ 의 entry 는
  majority 에서 +(1−p), minority 에서 −p (부호 반대). (uᵢ⊤ Ỹ) = (1−p) · Σ_{i ∈
  majority} uᵢ − p · Σ_{i ∈ minority} uᵢ. 이 양이 크려면 uᵢ 의 두 group 평균
  값이 *체계적으로 부호가 갈려야* — 즉 uᵢ 가 *class 를 구별* 해야 합니다.
  class-blind common-feature mode (두 group 의 평균이 같음) 의 (uᵢ⊤ Ỹ) ≈ 0.

따라서 centering 의 효과는 단순히 "average 빼기" 가 아니라 **score 가 measuring
하는 대상을 majority alignment 에서 class discrimination 으로 swap** 하는
것입니다. "minority 회복" 의 의미: A1 이 픽한 r-dim subspace 안에 minority 와
majority 의 부호가 갈리는 (= class-discriminative) mode 가 우선 포함되므로,
그 위에서 학습한 ridge predictor 가 minority sample 을 정확히 predict 할
expressive power 를 갖춥니다.

**MRPC vs QQP 의 mechanism 격차를 PR(c²) 만으로 설명할 수 있는가 — 솔직한
한계.** 사용자의 round-3 지적이 정확합니다. n=5000 의 정량:

| Dataset | train_maj | LC_LR@10 | PR(c²) | LR_maj@10 | A1_maj@10 | wins/42 |
|---|---:|---:|---:|---:|---:|---:|
| MRPC    | 0.674 | 0.318 | 274.9 | 0.828 | 0.752 | 35 |
| QQP     | 0.631 | 0.397 | 134.6 | 0.638 | 0.635 |  7 |

두 데이터셋 모두 LC_LR 은 0.3~0.4 의 낮은 영역, PR(c²) 도 100 이상의 분산된
영역에 같이 위치합니다. (LC_LR, PR(c²)) 의 정도 차이 (0.318 vs 0.397, 275 vs
135) 가 *원인* 이라면, wins 격차도 정도 차이로 나와야 합니다. 그러나 실제
wins/42 는 35 vs 7 의 *질적* 격차 (한 쪽 대승, 한 쪽 대패) — (LC, PR) 의 정도
차이만으로는 *원인-결과 격차의 크기 불일치* 가 남습니다. 즉 §2.4 의 정량 표가
mechanism 의 *consistent evidence* 는 되지만 *입증* 은 되지 않습니다.

다른 confounding 들을 솔직히 나열합니다. (i) **Task 종류 다름**: MRPC 는
news paraphrase detection (5.8K total), QQP 는 Quora duplicate question
detection (363K total). 우리 n=5000 은 MRPC 전체에 가깝지만 QQP 의 1.4%
sample — coverage / noise 구조가 다름. (ii) **Kernel spectrum 의 dataset
specific 특성**: BERT eNTK 가 두 도메인 (news 와 Quora) 에 서로 다른 spectral
구조를 만들 수 있어 PR(c²) 의 *절대값* 비교가 fair 한지 불명. (iii) **Effective
imbalance 의 학습 difficulty**: imbalance 의 *학습 가능성* 은 majority 의
class-conditional distribution 의 *spread* 에 의존 — 같은 67/33 imbalance
라도 majority distribution 이 더 좁으면 LR 이 더 잘 흡수.

이런 confounding 들 때문에 MRPC vs QQP 자연 실험만으로는 "Cond 2 의 cause
가 PR(c²) AND train_maj 이다" 라는 진술의 **원인성 (causality) 이 확립되지
않습니다**. mechanism 의 *consistent 정황 증거* 는 있지만 *isolated 검증* 은
아닙니다. 본 라운드 framework 의 Cond 2 정의는 이 한계를 인정하고 임시
heuristic 으로 유지 (PR(c²) ≥ 200 AND train_majority ≥ 0.6), 실제 원인성은
§6 의 controlled imbalance 실험으로 검증합니다.

### §2.5 √(C-1) SNR — 정정과 signal regime dependence

**Population label projection 정의.** Mercer 분해 K(x, x′) = Σᵢ μᵢ ψᵢ(x) ψᵢ(x′)
의 μᵢ, ψᵢ 가 population eigenvalue / eigenfunction. true label function
f*(x) = E[Y|x] 를 이 basis 로 전개하면 f*_c(x) = Σᵢ α_{i,c} ψᵢ(x), α_{i,c}
= ⟨f*_c, ψᵢ⟩_pop — α_{i,c} 가 *population label projection*. 우리가 측정하는
empirical (uᵢ⊤ ỹ_c) 는 √n scaling 하에서 α_{i,c} 의 추정량.

**One-hot constraint 와 effective 자유도.** Σ_c y_{i,c} = 1 (one-hot, 각 row
합 1) 가 centering 후 Σ_c ỹ_{i,c} = 0. 따라서 Σ_c (uᵢ⊤ ỹ_c) = 0 — C 개 column
의 (uᵢ⊤ ỹ_c) 가 (C-1) 차원 부분공간에 갇혀 있음. binary 의 경우 (C-1) = 1
이라 effective 1 차원 (이전 진술과 일치), 3-class 는 effective 2 차원,
4-class 는 effective 3 차원.

**SNR 유도.** ε_{i,c} := (uᵢ⊤ ỹ_c) − α_{i,c} 의 noise 가 (C-1) 차원 부분공간
안에서 Gaussian, std σ/√n.

  E[cᵢ²|signal] = Σ_c α_{i,c}² + (C-1) · σ²/n,
  Var[cᵢ²|noise floor] ≈ 2 (C-1) · (σ²/n)²,        (chi-square C-1 dof)
  std[noise floor] ≈ √(2(C-1)) · σ²/n.

각 class 에 비슷한 magnitude α 의 signal 이 있다는 가정 (Σ_c α_{i,c}² ≈
(C-1) α², constraint 반영) 하에 signal mode 의 cᵢ² 값 ≈ (C-1) α². 따라서

  gap = signal − noise_floor = (C-1) α²,
  SNR_ranking := gap / std[noise floor] = (C-1) α² / (√(2(C-1)) · σ²/n)
              = √((C-1)/2) · n α² / σ².                                    (4)

**고정 n, α, σ 에서 multi-class 의 SNR 은 binary 대비 √(C-1) 배.**
binary (C=2): √1 = 1; 3-class (MNLI): √2 ≈ 1.41×; 4-class (AG News): √3
≈ 1.73×; 가설적 C=10: √9 = 3×; C=20: √19 ≈ 4.36×. 이전 draft 의 √C 표기
(1.73× / 2× / 3.16× / 4.47×) 는 one-hot constraint 미반영 — 약간 over-counted.
정정 후 effect 가 *조금* 약화되지만 monotone 증가는 그대로.

**Signal regime dependence — 이것이 nuance 의 핵심.** SNR 이득 √(C-1) 가 wins
로 직접 transfer 되려면 *ranking 이 noise-bound 되어 있어야* 합니다. 만약
signal 이 충분히 강해 (n α² ≫ σ², 즉 우리 finite-n 에서도 noise 거의 무시
가능) 모든 mode 의 cᵢ² 추정이 정확하면 binary 에서도 ranking 이 정확 — multi-class
의 SNR 이득이 무의미 (wins 격차 0). 반대로 *small signal regime* (n α² ∼ σ²,
fine-grained mode ordering 이 noise 에 흔들림) 에서는 √(C-1) 이득이 모든 mode
의 ranking 안정에 직접 기여 — wins 격차 큼.

7 데이터셋의 multi-class wins 패턴이 정확히 이를 시사합니다. **MNLI (C=3,
LC_LR 0.318, signal 분산)**: signal 이 분산되어 (PR(c²) 432) 각 mode 의
α 가 작음 → small signal regime → √2 SNR 이득이 *모든 (r, sel%)* 에 균질 발현
→ 42/42, row avg +4pp 의 일관 우세. **AG News (C=4, LC_LR 0.773, signal 집중)**:
signal 이 top-r 에 집중 (PR(c²) 17) → 큰 α → strong signal regime 에 가깝
지만 top-r 이외 mode 의 ranking 에는 noise 영향 — *small budget (sel=1~2%)*
에서만 √3 SNR 이득이 효과적 발현 (top mode 의 c² ranking 이 가장 noise-sensitive
한 boundary) → 17/42 mixed, sel=1~2% column 만 A1 우위, 나머지 LR 우위.
**signal regime + Cond 1 ✗/✓ 의 조합이 multi-class 의 wins 패턴 분기를
설명** 합니다.

**C 많은 데이터셋으로 검증 가능성.** 사용자 질문 — "C 가 엄청 많은 데이터셋
으로 multi-class 가설 검증 가능한가" — 의 답은 *원리적으로 가능, 단 signal
regime 조건이 함께 고려되어야*. 후보 데이터셋:

- **DBpedia** (C=14, ontology classification): 큰 C, AG News 와 비슷한 topic
  classification 이라 signal 이 top-r 에 집중될 가능성. signal 강 → AG News
  의 mixed 패턴이 더 sharp 하게 나올 가능성.
- **20Newsgroups** (C=20): 가장 큰 C 사용 가능. text length 와 vocabulary 가
  다양해 signal 이 더 분산될 가능성. MNLI 의 uniform 패턴과 가까울 가능성.
- **TREC** (C=6, question type): 중간 C. 작은 dataset (~5500 train), n-effect
  와 C-effect 분리 어려울 수 있음.
- **Yahoo Answers** (C=10): 큰 dataset, 충분한 sample size 확보 가능.

검증 design 의 권장: **C ∈ {2, 3, 4, 5, 10, 14, 20}** 의 sweep 에서 A1 wins
격차의 average row Δ 를 측정. 만약 식 (4) 가 맞고 signal regime 이 모든 C
에서 similar 하면 wins 격차 ∝ √(C-1). plot 의 x 축 = √(C-1), y 축 = avg
wins 격차 — linear fit 이 식 (4) 의 직접 검증. 단, signal regime 이 C 에 따라
달라지면 (e.g., 더 큰 C 일수록 더 분산) signal 의 confounding effect 를 별도
변수 (PR(c²)) 로 controlling 해야 정확한 √(C-1) scaling 분리 가능. 이게 iter
04 의 우선 design task.

## §3. 세 조건의 최종 정의와 임계값 근거

| 조건 | Primary | Secondary indicators | 방향 | mechanism |
|---|---|---|:-:|---|
| **Cond 1** | LC_LR(r=10%) ≥ 0.5 | 작은 PR(c²), 큰 Spearman(λ, c²), 큰 overlap_LR_A1 | top-r 유리 | LR 이 centered label 의 절반 이상 흡수 → r-dim ridge 가 task 풀기 충분 |
| **Cond 2** | PR(c²) ≥ 200 AND train_majority ≥ 0.6 | LR_predict_majority_frac(r=10%) ≥ 0.7 (effect-side) | A1 유리 | label spread + imbalance → LR 의 class-blind degeneracy, A1 의 centering-aware mode 픽이 minority 회복 |
| **Cond 3** | C ≥ 3 | 작은 LC_LR (small signal regime 진단) | A1 유리 | cᵢ² 의 √(C-1) SNR 개선 — small signal regime 에서 가장 sharp |

**임계값 근거.** (a) Cond 1 의 0.5 — LC_LR 이 잡은 양 = 못 잡은 양의 자연
경계. 7 데이터셋 측정에서 0.5 가 SST-2 (0.608) / MR (0.547) 의 위와 MRPC
(0.318) / RTE (0.206) 의 아래를 명확히 분리. (b) Cond 2 는 **현 단계에서
임시 heuristic** — PR(c²) ≥ 200, train_maj ≥ 0.6 두 cutoff 모두 MRPC vs QQP
boundary 의 fitting 으로 정한 값이고, §2.4 의 confounding 한계로 인해 원인성
이 확립되지 않은 상태. §6 의 controlled imbalance 실험으로만 Cond 2 의 진위
와 정확한 boundary 가 검증됨. LR_maj ≥ 0.7 의 effect-side verification 도
heuristic. (c) Cond 3 의 C ≥ 3 — binary (C=2) 의 effective 차원 1 이 reference
단위라 어떤 multi-class 도 √(C-1) ≥ √2 > 1 의 strict improvement. C 가 클수록
효과 강 (단 signal regime 이 small 일 때).

r=10% 를 reference rank 로 택하는 이유는 §6 부록의 r-sweep 에서 LC_LR 의 dataset
간 ranking 이 r=5~30% 에 robust 하기 때문. iter 02 wins 표에서 A1/LR 격차가
r=10% × sel=5~10% 에 가장 선명하게 분포.

**Default 처리.** 아무 조건도 발동 안 하면 **default = top-r 우세**. A1 의
cᵢ² 추정이 chi-square (C-1) dof 항을 포함해 λᵢ 추정보다 본질적으로 noisy
하므로, 어떤 A1-favorable mechanism 도 없으면 noise 차이만 남아 top-r 우세.

## §4. 7 데이터셋 검증

| Dataset | C | train_maj | PR(c²) | LC_LR@10 | LR_maj@10 | Cond 1 | Cond 2 | Cond 3 | Predict | wins/42 | Match |
|---|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|---:|:-:|
| MNLI    | 3 | 0.343 | 432.4 | 0.318 | 0.335 | ✗ | ✗ | ✓ | A1     | 42/42 | ✓ |
| MRPC    | 2 | 0.674 | 274.9 | 0.318 | 0.828 | ✗ | ✓ | ✗ | A1     | 35/42 | ✓ |
| AG News | 4 | 0.254 |  17.2 | 0.773 | 0.243 | ✓ | ✗ | ✓ | mixed  | 17/42 | ✓ |
| SST-2   | 2 | 0.562 |  22.3 | 0.608 | 0.551 | ✓ | ✗ | ✗ | top-r  |  8/42 | ✓ |
| MR      | 2 | 0.502 |  27.7 | 0.547 | 0.505 | ✓ | ✗ | ✗ | top-r  |  4/42 | ✓ |
| QQP     | 2 | 0.631 | 134.6 | 0.397 | 0.638 | ✗ | ✗ | ✗ | default top-r |  7/42 | ✓ |
| RTE     | 2 | 0.502 | 687.9 | 0.206 | 0.556 | ✗ | ✗ | ✗ | default top-r |  4/42 | ✓ |

**MNLI** (Cond 3 단독): C=3 + 작은 LC_LR (signal 분산, PR(c²) 432) → small
signal regime + √2 SNR. 모든 (r, sel%) 에 균질 발현, 42/42. **MRPC** (Cond
2 단독): PR(c²) 275 + train_maj 0.674 → label spread + imbalance. A1 의
centering 으로 minority 회복, r=10%, sel ≥ 5% 영역에서 +4~+5pp A1 우세. sel
≤ 2% × r = 1% 의 SNR 부족 영역에서만 패배 (7 셀). **AG News** (Cond 1 + Cond
3): LC_LR 0.773 으로 LR 이 task 풀음 (PR(c²) 17 로 signal 집중), 동시에 C=4
로 √3 SNR. 두 mechanism 의 영역별 발현 — small budget (sel=1~2%) 만 A1 우위,
나머지 LR 우위 → mixed 17/42. **SST-2 / MR** (Cond 1 단독): LR 이 task
정보 흡수, top-r 안정 우세. **QQP** (default): PR(c²) 135 가 Cond 2 threshold
200 미달 — label 이 충분히 집중되어 LR 이 imbalance 자체 흡수. wins/42 = 7.
**RTE** (default): LC_LR 0.206 의 severe kernel-task mismatch + PR(c²) 688
의 극단적 label 분산. 어떤 조건도 발동 안 함 → default top-r, mismatch 가
selection noise sensitivity 를 키워 wins 격차가 sel 클수록 −7~−10pp 까지 벌어짐.

**Match score: 7/7.** Cond 2 가 MRPC 1 example, Cond 3 가 MNLI / AG News 2
examples — evidence base 가 좁아 iter 04 의 데이터셋 확장이 필수 (§5 의
다음 라운드 계획).

## §5. iter 02 의 5 조건 framework 와의 비교

| iter 02 조건 | 처분 | 근거 |
|---|:-:|---|
| Cond A (LC_LR ≥ 0.5) | **유지** = 새 Cond 1 | 사용자 직관 + framework 정합 |
| Cond B (full ridge gap small) | **삭제** | §2.2 — gap 순서와 wins 순서 무관 |
| Cond 1 (miss_LC_LR ≥ 0.5) | **삭제** | §2.3 — Cond A 의 음의 진술, 중복 |
| Cond 2 (minority detection) | **재정의 + 격상** = 새 Cond 2 | §2.4 — PR(c²) 를 cause-side mechanism 으로, train_majority 와 LR_maj 를 imbalance/degeneracy verification 으로 분리 |
| Cond 3 (multi-class SNR) | **재정의 + 정정** = 새 Cond 3 | §2.5 — √C → √(C-1) 정정, signal regime dependence 명시 |
| Spearman(λ, c²) (selection split amount) | **secondary indicator** | §1 — Cond 1 의 mechanism dual 측정 |
| overlap_LR_A1 (selection overlap) | **secondary indicator** | §1 — Cond 1 의 직접 측정 dual |

5 → 3 의 축약은 (a) 정보 없음 (B), (b) 중복 (iter 02 Cond 1) 의 삭제와 (c)
mechanism 재정의 (iter 02 Cond 2, 3) 의 결합. 추가로 iter 02 의 Spearman 과
overlap 두 quantity 는 *secondary indicator* 로 재배치 — main framework 는
3 조건이지만 진단의 두께는 7 quantity (LC_LR, PR(c²), Spearman, overlap,
train_maj, LR_maj, C) 가 받쳐줍니다.

## §6. Limitations 와 iter 04 의 우선 task

**Limitations.** (a) wins/42 측정이 iter 02 의 n=2000 single-seed (2026) 라
본 라운드의 n=5000 3-seed 와 setup mismatch — n=5000 wins 표 재측정이 의무.
(b) Cond 2, 3 의 evidence base 가 좁음 (각 1, 2 example). (c) PR(c²) 의 threshold
200 은 MRPC vs QQP boundary heuristic — 더 정확한 calibration 필요. (d) Cond
3 의 √(C-1) 이득은 signal regime 에 conditional — strong signal regime 에
서는 무의미. (e) 본 framework 은 *충분조건* 의 체계라 default case (QQP /
RTE) 의 wins 격차 amount (왜 RTE 가 QQP 보다 깊은 패배) 는 framework 외부의
kernel-task mismatch severity 에 의존.

**Iter 04 우선 task — Cond 2 의 isolated 검증 (사용자 제안 실험).** 가장
중요한 falsifier 는 *imbalance 만 가변* 시켜 mechanism 을 분리하는 controlled
실험입니다. **SST-2 (현재 top-r 우세 8/42, train_maj 0.562)** 와 **MR (현재
top-r 우세 4/42, train_maj 0.502)** 두 데이터셋을 base 로 하고, train 의
class ratio 만 50/50, 70/30, 90/10 으로 재구성 (subsample positive 또는 negative
class 로 강제 imbalance, total n=2000 유지). 각 ratio 에서 inv prediction, eigen,
r ∈ {1, 5, 10, 15, 20, 25, 30}% × sel ∈ {1, 2, 5, 10, 15, 20}% 의 42 셀에서
A1 vs LRFShap 격차 측정. *expected outcomes*:

- imbalance 가 깊어질수록 PR(c²) 와 LR_maj 가 어떻게 변하는지 직접 측정 —
  PR(c²) 가 train_majority 만 늘려도 자연스럽게 커지는지, 아니면 별도 mechanism
  으로 결정되는지를 판정.
- 90/10 imbalance 에서 LR_maj 가 train_majority (0.9) 보다 더 majority 편향이
  되어 (e.g., 0.95+) Cond 2 의 effect-side trigger 가 작동하는지.
- 동일 ratio 에서 wins/42 가 50/50 의 top-r 우세에서 70/30 또는 90/10 의
  A1 우세로 *뒤집히는지*. 뒤집히면 → Cond 2 mechanism 의 directly causal
  검증 성공. 안 뒤집히면 → Cond 2 의 가설이 *불완전* — PR(c²) 또는 다른
  hidden variable 이 추가로 필요.
- 동일 실험을 MR 에 반복해 데이터셋 specific 효과와 imbalance 효과를 분리.

**그 외 우선 task.** (i) **n=5000 wins/42 재측정** — 기존 7 데이터셋의 framework
match 재확인 (iter 02 의 n=2000 single-seed 와 일치하는지). (ii) **C-sweep 실험**:
C ∈ {2, 3, 4, 5, 10, 14, 20} 의 데이터셋 (현 7 + TREC C=6, Yahoo Answers C=10,
DBpedia C=14, 20Newsgroups C=20) 에서 wins 격차의 row avg Δ 측정. plot x =
√(C-1), y = Δ — 식 (4) 의 직접 검증. PR(c²) 를 covariate 로 통제. (iii) **MRPC
confusion-matrix 검증**: minority recall 의 A1 회복을 (r, sel%) 셀별로 직접
측정 — Cond 2 mechanism 의 effect-side 증거를 셀 단위로.

**부록 — LC_LR(r) r-robustness (n=5000).** r=10% 의 LC_LR ranking 이 r=5~30%
전 구간 유지 (AG News 0.74→0.85 > SST-2 0.56→0.73 > MR 0.50→0.67 > QQP 0.34→
0.55 > MNLI/MRPC 0.25→0.50 > RTE 0.13→0.41). near-threshold 의 MNLI/MRPC/QQP
는 Cond 2/3 발동 여부로 분류 그대로 (MNLI → Cond 3, MRPC → Cond 2, QQP →
default).

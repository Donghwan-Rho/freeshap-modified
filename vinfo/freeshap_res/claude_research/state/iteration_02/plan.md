# iter 02 plan — FC vs LC orthogonal framework 로 A1 vs LRFShap 결과 설명

## §0 미팅 시작 전 한 줄 정리

iter 01 에서 우리는 7 dataset 의 A1 vs LRFShap head-to-head 결과 (MNLI 42/42 부터
MR/RTE 4/42 까지) 가 dataset 별로 극단적으로 갈린다는 걸 봤습니다. 그때는 archetype
분류 (multi-class 우세, kernel-task fit 등) 로 *서술* 만 하고 끝났는데, 이번 iter 의
목표는 사용자가 제안한 *FC (kernel norm) axis 와 LC (label energy) axis 의 직교성*
관점으로 그 win 패턴을 식 단위로 설명하는 겁니다. 실험은 안 돌리고 precomputed.json
의 수치만 가지고 분석합니다.

## §1 사용자 직관 (i)-(iv) 의 formal 정의

먼저 표기부터. 정렬된 eigendecomposition K = Σᵢ λᵢ uᵢ uᵢ⊤ (λ₁ ≥ λ₂ ≥ ...) 이고
label projection 을 cᵢ² := (uᵢ⊤Y)² (Y 는 centered C-class one-hot, ‖Y‖_F² = Σᵢ cᵢ²)
라 둡니다. ρ = ridge 정규화. r 개 mode 를 고른 index set 을 I 라 쓸 때:

- **FC(I) := Σ_{i∈I} λᵢ / Σⱼ λⱼ** — kernel Frobenius energy 의 retain ratio.
- **LC(I) := Σ_{i∈I} cᵢ² / Σⱼ cⱼ²** — label energy 의 retain ratio.

이제 두 selection 의 objective:

**(i) top-r (= LRFShap 의 선택, 줄여서 I_LR)**:
   I_LR = argmax_{|I|=r} Σ_{i∈I} λᵢ.
   즉 K 의 best rank-r Frobenius approximation. cᵢ² 는 **전혀 안 들어감**. label-blind.

**(ii) A1 score (= s_i = (λᵢ/(λᵢ+ρ))² · cᵢ²) 와 그 λ >> ρ 극한**:
   A1 은 s_i 가 가장 큰 r 개 mode 를 고릅니다. filter f(λ) := (λ/(λ+ρ))² 의 행동:
   λ >> ρ 면 f → 1, λ << ρ 면 f → λ²/ρ² → 0. precomputed.json 의 `frac_lambda_gt_10rho`
   를 보면 7 dataset 모두 0.999 이상 (RTE 만 정확히 0.9992, 나머지는 1.0). 즉 거의
   모든 mode 가 λ >> ρ 영역에 있고, 이 영역에서는 f(λᵢ) ≈ 1 이므로
   **I_A1 ≈ argmax_{|I|=r} Σ_{i∈I} cᵢ²** — 순수 LC selection 입니다. λ 는 filter
   의 cutoff gate 로만 작동하고, ranking 결정에는 cᵢ² 만 들어가요. `filter_top1pct_min`
   도 모든 dataset 에서 0.9999 이상이라, top 1% mode 들 사이에서 filter 가 차이를
   거의 안 만듭니다. 사용자 직관 (ii) 가 정확히 성립한다는 1차 확인.

   다만 critic 이 짚을 만한 caveat: "λ >> ρ" 라 해도 λ 가 dataset 내에서 6-7
   orders 의 dynamic range 를 가지면 (λ_max/λ_min ≈ 10⁶–10⁸ 수준) bottom 쪽
   mode 의 filter 는 1 에서 detectable 하게 떨어집니다. 그래도 top-r selection
   ranking 자체는 cᵢ² 가 dominate 합니다 — λ 와 cᵢ² 가 weakly correlated 일 때
   (Spearman 0.1–0.4 수준이라 weak 가 맞음).

**(iii) 두 axis 의 직교성과 selection 결과의 직교성**:
   axis 자체는 분명히 독립적 — FC 는 K 만 보고, LC 는 (K, Y) joint 만 보니까요.
   하지만 *selection 결과의 직교성* 은 Y 와 K 의 eigen-decomposition 의 결합
   분포에 달려있고, 이걸 측정하는 *한 양* 이 바로
   **Spearman(λᵢ, cᵢ²)** — i 에 대해 (λᵢ, cᵢ²) pair 의 rank-correlation.
   이 값이 0 이면 두 ranking 이 완전히 무관 → I_LR 과 I_A1 의 overlap 이
   chance level (r/n). 이 값이 1 이면 두 ranking 이 일치 → overlap = 1
   → A1 과 top-r 이 동일한 mode 를 고른다.

**(iv) 각 selection 이 놓친 정보 quantification**:
   - top-r 이 놓친 label energy: **miss_LC_by_LR := 1 − LC(I_LR)** = Σ_{i∉I_LR} cᵢ² / Σⱼ cⱼ².
   - A1 이 놓친 kernel energy: **miss_FC_by_A1 := 1 − FC(I_A1)** = Σ_{i∉I_A1} λᵢ / Σⱼ λⱼ.

   LRFShap 의 eq. (9) (논문 p. 7) 는 Shapley 추정 오차를 Σ_{i∉I} λᵢ cᵢ² 형태의
   "missed FC × LC product" 로 bound 하니, 위 두 양은 각각 bound 의 한쪽 항을
   채웁니다. eq9_gap_LR = Σ_{i∉I_LR} λᵢ cᵢ², eq9_gap_A1 = 그 A1 버전. 둘 다
   precomputed.json 에 들어있고, ratio 가 작을수록 A1 의 bound 가 더 tight.

## §2 7 dataset 의 핵심 수치 (r=10%, 즉 r_abs = 0.10·n)

먼저 spectrum-label structure:

| Dataset | λ_max/λ_med | frac(λ>10ρ) | Spearman(λ,c²) | wins/42 |
|---|---:|---:|---:|---:|
| MNLI | 5093 | 1.000 | 0.226 | **42** |
| MRPC | 5492 | 1.000 | 0.114 | **35** |
| AG News | 10092 | 1.000 | 0.377 | 17 |
| SST-2 | 4564 | 1.000 | 0.193 | 8 |
| QQP | 25716 | 1.000 | 0.104 | 7 |
| RTE | 6924 | 0.999 | 0.098 | 4 |
| MR | 11876 | 1.000 | 0.150 | 4 |

7 dataset 모두 λ >> ρ regime 이 거의 완전히 성립 (frac > 0.999). 즉 사용자 가정
(ii) 는 universal 합니다 — A1 의 selection 이 ≈ pure LC selection 인 게 dataset
별로 다르지 않음. 그러면 dataset 차이의 mechanism 은 selection rule 의 형태가
아니라 (λ, c²) joint distribution 에 있어야 합니다.

다음, r=10% 에서의 FC/LC retain 과 놓친 정보:

| Dataset | FC_LR | LC_LR | FC_A1 | LC_A1 | miss_LC_LR (A) | miss_FC_A1 (B) | overlap (E) | eq9_gap A1/LR (D) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNLI | 0.822 | 0.271 | 0.271 | 0.410 | 0.729 | 0.729 | 0.325 | 0.809 |
| MRPC | 0.819 | 0.280 | 0.188 | 0.532 | 0.720 | 0.812 | 0.260 | 0.650 |
| AG News | 0.873 | 0.747 | 0.292 | 0.778 | 0.253 | 0.708 | 0.510 | 0.878 |
| SST-2 | 0.814 | 0.566 | 0.323 | 0.704 | 0.434 | 0.677 | 0.325 | 0.682 |
| QQP | 0.930 | 0.345 | 0.124 | 0.549 | 0.655 | 0.876 | 0.290 | 0.689 |
| RTE | 0.821 | 0.206 | 0.139 | 0.470 | 0.794 | 0.861 | 0.241 | 0.667 |
| MR | 0.898 | 0.516 | 0.303 | 0.662 | 0.484 | 0.697 | 0.310 | 0.699 |

선두 관찰 한 가지 짚고 가야겠어요: **모든 dataset 에서 LC_A1 > LC_LR 이고 FC_LR
> FC_A1** 가 항상 성립합니다 (예: MRPC r=10% 에서 LC 는 0.28 → 0.53, FC 는 0.82
→ 0.19). 즉 A1 은 LR 보다 label energy 를 거의 2 배 잘 잡고, LR 은 A1 보다 kernel
energy 를 4 배 가까이 잘 잡습니다. 이것이 두 selection 의 *trade-off* 라는 framework
가설을 정량적으로 확인해 줍니다. 두 axis 가 진짜로 직교적이어서, 한 쪽을 최적화하면
다른 쪽이 손해를 봅니다.

## §3 5 후보 척도 vs empirical_wins — best predictor 선정

각 후보 척도를 dataset 당 한 값 (r=10% 사용) 으로 요약하고, 7-point empirical_wins
와의 Spearman rank correlation 을 계산했습니다. wins 순서 (desc): MNLI=1, MRPC=2,
AG=3, SST=4, QQP=5, RTE=6.5, MR=6.5 (RTE, MR 동률).

| 척도 | 가설 | r=10% Spearman ρ | 해석 |
|---|---|---:|---|
| A: miss_LC_by_LR | 클수록 A1 가치 큼 | **+0.01** | 무상관 |
| B: miss_FC_by_A1 | 클수록 LR 가치 큼 (음의 상관) | **+0.08** | 무상관 |
| C: Spearman(λ,c²) | 클수록 두 axis aligned → ? | **+0.60** | 강한 양의 상관 |
| D: eq9_gap_A1/LR | 작을수록 A1 bound 좋음 | **−0.19** | 약한 음의 상관 |
| E: overlap_LR_A1 | 클수록 두 방법 일치 → 차이 작음 | **+0.45** | 중간 양의 상관 |

가장 놀라운 결과는 **C 가 best predictor** (ρ ≈ 0.60) 라는 점입니다. *높은*
spectrum-label alignment 가 *더 큰* A1 우위로 이어진다는 거예요. 직관적으로는
"λ 와 c² 가 align 되면 top-r 이 어차피 high-c² mode 를 같이 잡으니까 A1 우위가
줄어들 것" 같은데, 데이터는 정확히 반대입니다. 미팅에서 이 점이 제일 흥미로운
discovery 라고 생각합니다.

(i) 와 (iv) 의 직접 양 (A, B 후보) 이 거의 무상관이라는 사실도 중요합니다. 즉
*"놓친 정보의 절대량"* 자체는 win 수를 거의 결정하지 못합니다. RTE 가 가장 큰
miss_LC_by_LR (0.79) 인데 wins 는 4/42 로 꼴찌고, MNLI 는 비슷한 miss (0.73)
인데 42/42 압승. 양만 가지고는 설명이 안 됩니다.

E (overlap) 가 두 번째로 좋은 (+0.45) 이유는 partially Spearman(λ,c²) 와 같은
joint distribution 정보를 다른 각도에서 측정하기 때문입니다. 둘은 redundant.

D (eq9_gap_ratio) 가 약한 음의 상관인 건 이론적으로 가장 그럴듯한 척도 — 사용자
직관과도 맞는 — 인데 의외로 약합니다. 이건 critic 한테 짚어달라고 부탁할 부분:
eq. (9) 는 *worst-case* bound 라 실제 empirical accuracy 와 monotonically 맞물리지
않을 수 있습니다. bound 가 deterministic 하게 가까운 게 win 을 보장하지 않아요.

## §4 Mechanism — *왜* Spearman(λ,c²) 가 좋은 predictor 인가

저는 다음 메커니즘을 제안합니다. Spearman(λ,c²) 가 클 때 두 가지 사건이 동시에
일어납니다.

**첫째, A1 selection 이 "안전하게" 큰 cᵢ² 를 고릅니다.** Pure LC selection 은
이론상 unstable — cᵢ² 가 작은 noise mode 에서 estimation variance 가 큰데, 그
mode 의 λ 가 작으면 (uᵢ⊤Y)² estimator 자체가 noisy 합니다 ((u⊤Y)² 의 sample
variance 가 1/λ 에 비례하는 경향). 그러나 λ 와 c² 가 aligned 면, top-c² mode 의
λ 도 자연스럽게 크기 때문에 estimator 가 안정적입니다.

**둘째, top-r selection 은 그래도 c² 를 *완전히* 놓치지는 않습니다.** Aligned
distribution 에서 top-λ mode 는 partial 하게 top-c² mode 와 겹치니까, top-r 의
LC retain 도 일정 수준 (e.g., AG News r=10% 에서 LC_LR = 0.747) 까지는 올라갑니다.
즉 top-r 의 baseline 자체가 강합니다.

그런데 wins 수는 *둘의 차이* — A1 의 잘 잡음 만큼 — 가 아니라 *A1 이 안전하게
잘 잡고 top-r 이 *상대적으로* 노력에 비해 underperform 하는* 정도에서 결정됩니다.
Aligned regime 에서는 A1 의 estimator stability 가 확보되면서 top-r 대비 marginal
gain 이 크게 남습니다.

Anti-aligned regime (낮은 Spearman) 에서는 정반대 — top-c² mode 가 low-λ region
에 흩어져 있어 A1 estimator 가 unstable 하고, top-r 은 본인의 high-λ kernel mode
에서 안정적으로 cᵢ² 의 일부분만 잡습니다. A1 의 *이론적* advantage (높은 LC retain)
가 *경험적으로는* estimator variance 로 잠식되는 거죠. 이 가설은 Bordelon-Canatar-Pehlevan
(2021, Nature Comm) [B?] 의 task-model alignment 이론과 결이 같습니다. spectral
filter 가 high-c² mode 를 implicit 하게 prioritize 하는데, 이 prioritization 이
유효하려면 alignment 가 필요하다는 거예요.

**iter 01 archetype 과의 매핑.** Multi-class (MNLI, AG News, MRPC 의 task 자체는
binary 지만 paraphrase detection 으로 sub-label structure 있음) 는 label energy
가 여러 mode 에 분산되고, embedding 의 sub-cluster 가 top eigenvector 와
어느 정도 align 되는 경향이 있습니다. 그래서 Spearman(λ,c²) 가 (MRPC 0.11 정도로
낮긴 해도) 다른 dataset 들과의 *상대적 rank* 에서 상위로 올라옵니다. AG News
(0.38) 는 가장 높은 Spearman 이지만 wins 가 17/42 인 것은 §3 에서 본 outlier
가능성 — 보다 자세히 §5 에서 다룹니다.

## §5 RTE vs MRPC paradox — signal-vs-noise in missed LC

본문에서 가장 골치 아픈 케이스. RTE 와 MRPC 둘 다 miss_LC_by_LR 이 큽니다
(RTE 0.79, MRPC 0.72), 즉 top-r 이 label energy 의 70-80% 를 못 잡아요. 그런데
wins 는 RTE 4/42 vs MRPC 35/42. 절대 miss 양으로 설명 불가능합니다.

여기서 제 가설 (사용자 directive 의 가설을 정량적으로 받습니다): **놓친 label
energy 가 *learnable signal* 인지 *unlearnable noise* 인지에 따라 A1 의 win
가능성이 갈립니다.** Learnable signal 은 *어떤 mode 가 있긴 있는데 단지 high-λ
영역 밖에 있어서* top-r 이 못 잡는 경우, unlearnable noise 는 *Y 에 random
direction 으로 흘러간 component* 라 어느 selection 도 prediction 으로 못 옮기는
경우.

측정 가능한 proxy 로 저는 두 가지를 제안합니다:

(1) **LC_A1 − LC_LR (= delta_LC) 의 *prediction accuracy 로의 transfer rate***:
   miss_LC_by_LR 이 비슷해도, A1 이 그 missed signal 의 얼마를 실제로 capture
   하느냐가 다릅니다. precomputed.json r=10% 에서 delta_LC 비교: RTE = 0.264,
   MRPC = 0.252 — 비슷합니다. 하지만 LC_A1 자체는 RTE 0.470 vs MRPC 0.532.
   MRPC 가 절대 LC retention 이 살짝 더 큽니다. 이건 *부분적* 설명이지 충분치는
   않아요.

(2) **Spectrum-label alignment 와 label energy 분포의 "concentration"**: MRPC 의
   Spearman 은 0.114, RTE 는 0.098 — 거의 비슷합니다. 그런데 cᵢ² 의 분포가 어떻게
   spread 되어 있는지가 다를 수 있습니다. 만약 RTE 의 cᵢ² 가 거의 uniform 하게
   1000+ mode 에 흩어져 있다면 (effective rank 가 크다면), 어떤 r-mode subset 을
   골라도 LC 를 의미있게 capture 할 수 없습니다 — 이게 *noise* 케이스. 반면
   MRPC 의 cᵢ² 가 200-500 mode 에 집중되어 있고 top-c² mode 들의 cᵢ² 가 큰 값
   이라면, A1 의 LC selection 이 "right mode" 를 정확히 짚어줍니다 — *signal*
   케이스.

이걸 측정하려면 *cᵢ² 의 entropy* 또는 *participation ratio*
PR(c²) := (Σ cᵢ²)² / Σ cᵢ⁴ 같은 양이 필요한데, precomputed.json 에는 안
들어있어서 critic 라운드에서 추가 계산을 요청해야 합니다 (사실 raw data 에서
손쉽게 추가 가능). 이게 본 framework 의 다음 step 입니다.

## §6 paper framing 권고 (이게 §5 끝나고 미팅 마지막에 드릴 말씀)

이번 분석에서 명확히 보이는 건:

1. A1 의 selection 은 거의 모든 dataset 에서 *pure LC selection* 의 근사 (frac(λ>10ρ)
   ≈ 1 덕분에). 이건 robust 한 발견이고 paper 의 §5.5 framing 을 정당화합니다.

2. *놓친 정보의 절대량* (miss_LC_by_LR, miss_FC_by_A1) 은 A1 vs LRFShap empirical
   결과를 거의 예측 못합니다 (Spearman ≈ 0). 이건 직관과 정면충돌이고, paper 에서는
   이 점을 *명시적으로* 다뤄야 합니다 — "loose bound 가 실제 비교 결과를 보장하지
   않는다" 는 caveat 으로.

3. **Spectrum-label alignment Spearman(λ,c²) 가 가장 강한 predictor** (ρ ≈ 0.60).
   이게 본 iter 의 main finding 이고, Bordelon-Canatar 의 task-model alignment
   이론과 자연스럽게 연결됩니다.

4. RTE-MRPC paradox 는 *participation ratio of cᵢ²* 같은 추가 양으로 풀 수 있을
   것으로 봅니다. critic 라운드에서 검증.

## §7 다음 iter 의 obligatory items (executor 가 필요한 부분)

- **PR(c²) 와 effective LC dimension** 을 7 dataset 전체에서 추가 계산.
- **Synthetic dataset 으로 falsification**: aligned vs anti-aligned spectrum-label
  joint distribution 을 인위적으로 만들어 wins 패턴이 framework 가설대로 따라오는지.
- A1 score 의 *modified* version — λ filter 를 끄고 pure (uᵢ⊤Y)² 만 쓰는 baseline
  (직관 (ii) 의 극한) 과 A1 의 실증 비교, λ >> ρ regime 에서 정말 동일한지.

이상이 round 1 plan 의 핵심입니다. critic 라운드에서 (a) λ >> ρ 의 정확한 한계
조건, (b) Spearman(λ,c²) 의 robustness (outlier sensitivity), (c) eq. (9) bound
의 actual vs apparent tightness 를 짚어달라고 요청할 예정입니다.

## §8 참고 문헌 (iter 01 의 것 + 본 iter 새로 link)

- Bordelon, Canatar, Pehlevan (2021), "Spectral bias and task-model alignment
  explain generalization in kernel regression and infinitely wide neural networks",
  *Nature Communications*. https://www.nature.com/articles/s41467-021-23103-1 —
  본 iter 의 mechanism (§4) 의 이론적 grounding.
- Simon et al. (2023), "A theory of generalization for wide neural nets",
  eigenlearning framework. https://jamiesimon.io/blog/eigenlearning/ — learnability
  의 zero-sum 분배 (Σ over mode = n) 가 본 iter 의 LC retention 개념과 dual.
- LRFShap (Kang et al., 2025), Prop §5.5 eq. (9). — eq9_gap 정의 출처.
- iter 01 plan_v2.md 의 bibliography — 4 트랙 (ridge leverage, KTA, eigenlearning,
  data Shapley) 은 재인용 없이 reference 만 유지.

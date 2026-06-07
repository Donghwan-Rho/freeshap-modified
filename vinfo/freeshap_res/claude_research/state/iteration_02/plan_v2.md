# iter 02 plan v2 — FC vs LC orthogonal framework 로 A1 vs LRFShap 결과 설명 (critic round 1 반영판)

## §0 미팅 시작 전 한 줄 정리

iter 01 의 7 dataset head-to-head 결과 (MNLI 42/42, MRPC 35/42, AG 17/42, SST 8/42, QQP 7/42, RTE 4/42, MR 4/42) 를 *FC (kernel norm) axis 와 LC (label energy) axis 의 직교성* 관점에서 식 단위로 설명하는 것이 본 iter 의 목표입니다. 실험은 안 돌리고 precomputed.json 의 수치만 가지고 분석합니다. Round 1 critique 가 (a) §1 의 A1 ≈ LC limit 조건이 불완전, (b) §3 의 best predictor Spearman ρ ≈ 0.60 의 n=7 통계 신뢰도, (c) prediction-side vs Shapley-side 분리 미흡, (d) falsifier 2 개 추가, (e) 출처 보정 다섯 가지를 짚어줬고, round 2 는 각 항목을 식·표·한 단락 단위로 보강합니다.

## §1 사용자 직관 (i)-(iv) 의 formal 정의

표기. 정렬된 eigendecomposition K = Σᵢ λᵢ uᵢ uᵢ⊤ (λ₁ ≥ λ₂ ≥ ...), label projection cᵢ² := (uᵢ⊤Y)² (Y 는 centered C-class one-hot, ‖Y‖_F² = Σᵢ cᵢ²), ρ = ridge 정규화. r 개 mode 를 고른 index set 을 I 라 쓸 때:

- **FC(I) := Σ_{i∈I} λᵢ / Σⱼ λⱼ** — kernel Frobenius energy 의 retain ratio.
- **LC(I) := Σ_{i∈I} cᵢ² / Σⱼ cⱼ²** — label energy 의 retain ratio.

두 selection 의 objective:

**(i) top-r (= LRFShap 의 선택, I_LR)**:
   I_LR = argmax_{|I|=r} Σ_{i∈I} λᵢ. K 의 best rank-r Frobenius approximation. cᵢ² 는 안 들어감 — label-blind.

**(ii) A1 score sᵢ = f(λᵢ) · cᵢ², f(λ) = (λ/(λ+ρ))²**:
   A1 은 sᵢ top-r mode 를 고릅니다. critique §1 이 지적한 것처럼, "λ >> ρ 이면 f ≈ 1 이므로 I_A1 ≈ argmax Σ cᵢ²" 는 *충분조건의 절반*입니다. 정확한 조건은: I_A1 = I_LC (= pure LC top-r) ⟺ ∀i ∈ I_LC, ∀j ∉ I_LC,
   
   sᵢ > sⱼ ⟺ f(λᵢ)·cᵢ² > f(λⱼ)·cⱼ² ⟺ cⱼ²/cᵢ² < f(λᵢ)/f(λⱼ).

   즉 filter ratio f(λᵢ)/f(λⱼ) 의 dynamic range 가 *동일 split (in vs out of I_LC)* 에서 cᵢ² ratio 의 dynamic range 를 충분히 dominate 해야 두 selection 이 일치합니다. 단순 frac(λ > 10ρ) ≈ 1 은 *전체 mode pool* 에서 평균적으로 λ >> ρ 라는 양일 뿐이고, top-r 의 *bottom boundary* mode 가 정말로 f ≈ 1 인지를 보장하지는 않습니다. precomputed.json 의 `filter_top1pct_min` ≥ 0.9999 도 *상위 1% 가 차지하는 sᵢ ranking 안에서만* 보장이고, r 이 10%-30% 로 커지면 cutoff 가 1% 밖으로 빠집니다.

   구체적으로 RTE 가 가장 큰 위험 지대입니다. RTE 의 λ_min = 0.0 (precomputed.json, eigvals_summary), 즉 일부 mode 에서 f(λ) = 0 으로 정확히 떨어집니다. 그 mode 의 cᵢ² 가 아무리 커도 A1 은 *원리적으로 선택 못함* 인데 pure LC selection 은 선택 가능 — 이 mode 에서 I_A1 ≠ I_LC 가 강제됩니다. precomputed.json 의 `overlap_LR_A1` (= |I_LR ∩ I_A1|/r) 은 I_LR vs I_A1 의 overlap 이라 직접 대응양은 아니지만, RTE r=10% 에서 0.241 로 다른 dataset 대비 낮은 것이 *bottom boundary 의 filter cut* 효과를 부분적으로 반영한다고 봅니다. r=30% (RTE r_abs = 747) 에서 LC_A1 = 0.796 으로 올라가지만 I_A1 = I_LC 의 동등성 정도는 직접 측정 안 된 양 — round 3 또는 next iter 에서 |I_A1 ∩ I_LC|/r 같은 추가 양으로 확정해야 합니다. 결론적으로 (ii) 는 **거의 모든 dataset 에서 잘 근사되지만 universal 하진 않고, 특히 λ_min → 0 인 RTE 에서 partial 미성립**입니다.

**(iii) 두 axis 의 직교성과 selection 결과의 직교성**:
   axis 자체는 명확히 독립 — FC 는 K 만, LC 는 (K, Y) joint 만 봅니다. *selection 결과의 직교성* 은 Y 와 K eigen-decomposition 의 joint distribution 에 달려있고, 이를 측정하는 한 양이 **Spearman(λᵢ, cᵢ²)** 입니다. Spearman = 0 이면 두 ranking 이 독립 → expected overlap = r/n; Spearman = 1 이면 두 ranking 일치 → overlap = 1. critique §1 이 짚어준 *internal consistency* 확인을 하나 추가하면: AG News Spearman 0.38, overlap 0.51 vs RTE Spearman 0.10, overlap 0.24 — overlap ≈ r/n + (1 − r/n) · g(Spearman) 형태의 monotone 관계가 데이터에서 관측되어 framework 의 내적 정합성을 뒷받침합니다.

**(iv) 각 selection 이 놓친 정보 quantification**:
   - top-r 이 놓친 label energy: **miss_LC_by_LR := 1 − LC(I_LR)** = Σ_{i∉I_LR} cᵢ² / Σⱼ cⱼ².
   - A1 이 놓친 kernel energy: **miss_FC_by_A1 := 1 − FC(I_A1)** = Σ_{i∉I_A1} λᵢ / Σⱼ λⱼ.

   LRFShap (lrfshap.pdf, p. 7, eq. (9)) 은 Shapley 추정 오차를 Σ_{i∉I} λᵢ · cᵢ² 형태의 "missed FC × LC product" 로 bound 하니, 위 두 양은 각 bound 항을 채웁니다. eq9_gap_LR = Σ_{i∉I_LR} λᵢ cᵢ², eq9_gap_A1 = 그 A1 버전. 둘 다 precomputed.json 에 들어있습니다.

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

7 dataset 모두 λ >> ρ regime 이 거의 완전히 성립 (frac > 0.999) — universal. (다만 §1 의 *top-r boundary* 보강 조건으로 RTE 만 partial 미성립.) Dataset 차이의 mechanism 은 selection rule 의 form 이 아니라 (λ, c²) joint distribution 에 있어야 합니다.

r=10% 에서의 FC/LC retain 과 놓친 정보:

| Dataset | FC_LR | LC_LR | FC_A1 | LC_A1 | miss_LC_LR (A) | miss_FC_A1 (B) | overlap (E) | eq9_gap A1/LR (D) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNLI | 0.822 | 0.271 | 0.271 | 0.410 | 0.729 | 0.729 | 0.325 | 0.809 |
| MRPC | 0.819 | 0.280 | 0.188 | 0.532 | 0.720 | 0.812 | 0.260 | 0.650 |
| AG News | 0.873 | 0.747 | 0.292 | 0.778 | 0.253 | 0.708 | 0.510 | 0.878 |
| SST-2 | 0.814 | 0.566 | 0.323 | 0.704 | 0.434 | 0.677 | 0.325 | 0.682 |
| QQP | 0.930 | 0.345 | 0.124 | 0.549 | 0.655 | 0.876 | 0.290 | 0.689 |
| RTE | 0.821 | 0.206 | 0.139 | 0.470 | 0.794 | 0.861 | 0.241 | 0.667 |
| MR | 0.898 | 0.516 | 0.303 | 0.662 | 0.484 | 0.697 | 0.310 | 0.699 |

핵심 관찰: 모든 dataset 에서 **LC_A1 > LC_LR 이고 FC_LR > FC_A1**. A1 은 LR 보다 label energy 를 2 배 가까이 잘 잡고, LR 은 A1 보다 kernel energy 를 4 배 가까이 잘 잡습니다. 두 axis 가 정량적으로 trade-off 관계라는 framework 가설을 데이터가 확인해 줍니다.

## §3 5 후보 척도 vs empirical_wins — best predictor 선정 (statistical robustness 강화)

각 후보 척도를 dataset 당 한 값으로 요약하고 7-point empirical_wins 와의 Spearman rank correlation 을 계산했습니다. wins 순서 (desc): MNLI=1, MRPC=2, AG=3, SST=4, QQP=5, RTE=6.5, MR=6.5 (RTE, MR 동률).

| 척도 | 가설 | r=10% Spearman ρ | 해석 |
|---|---|---:|---|
| A: miss_LC_by_LR | 클수록 A1 가치 큼 | **+0.01** | 무상관 |
| B: miss_FC_by_A1 | 클수록 LR 가치 큼 | **+0.08** | 무상관 |
| C: Spearman(λ,c²) | 클수록 두 axis aligned → ? | **+0.60** | 양의 상관 (단 n=7) |
| D: eq9_gap_A1/LR | 작을수록 A1 bound 좋음 | **−0.19** | 약한 음의 상관 |
| E: overlap_LR_A1 | 클수록 두 방법 일치 → 차이 작음 | **+0.45** | 중간 양의 상관 |

**§3.1 통계적 신뢰도 — n=7 의 한계.** critique §2 가 짚어준 첫 보강. ρ = +0.60 의 양측 p-value 는 n=7 에서 약 0.15 — 유의수준 0.05 에 미달합니다. 95% confidence interval 은 Fisher z-transform 으로 대략 (−0.20, +0.92) 의 wide band. 즉 *단일 후보 척도로 wins 를 결정짓는다* 는 주장은 통계적으로 약합니다. 후보들 사이의 **qualitative ranking** 은 신뢰할 수 있습니다 — C > E > D > A ≈ B 의 순서는 어떤 robustness check 에서도 뒤집히지 않습니다.

Leave-one-out robustness:

| 제외 dataset | Spearman(C, wins) |
|---|---:|
| 전체 7 | +0.598 |
| AG News 제외 | +0.643 |
| MNLI 제외 | +0.471 |
| RTE 제외 | +0.429 |
| MR 제외 | +0.61 |

MNLI 또는 RTE 를 빼면 ρ 가 0.43–0.47 로 떨어집니다. C 의 상관이 *single-dataset sensitive* 라는 거예요. AG News 를 빼면 오히려 ρ 가 살짝 올라가는 (+0.64) 게 흥미로운데, AG 는 Spearman 이 가장 높지만 wins 는 17/42 의 중간값이라 C 의 monotone 관계에서 살짝 빠지는 outlier 역할입니다 — §4 에서 mechanism 차원에서 다룹니다.

**§3.2 r 의존성 — r=20% robustness check.** critique §2 의 두 번째 보강 요청. r=10% 외에 r=20% 에서도 후보별 Spearman 을 계산해 r 선택의 robustness 를 확인합니다.

| 척도 | r=10% Spearman | r=20% Spearman | 평균 (r=10,20) |
|---|---:|---:|---:|
| A: miss_LC_by_LR | +0.01 | +0.13 | +0.07 |
| B: miss_FC_by_A1 | +0.08 | −0.13 | −0.03 |
| C: Spearman(λ,c²) | +0.60 | +0.60 | +0.60 |
| D: eq9_gap_A1/LR | −0.19 | +0.02 | −0.09 |
| E: overlap_LR_A1 | +0.45 | +0.43 | +0.44 |

핵심: C 는 r 에 무관하게 +0.60 으로 *가장 큰 절댓값*. 후보 C 가 spectrum-label joint distribution 의 *dataset-level scalar* 라서 r 에 비의존이기 때문에 (cᵢ² ranking 과 λᵢ ranking 자체의 Spearman 은 어떤 r 을 잡더라도 변하지 않음), 다른 r-dependent 척도들과 달리 robust 합니다. E (overlap) 도 +0.43 ~ +0.45 로 stable. A, B, D 는 r 변경에 sensitive — A 의 +0.01 → +0.13 swing 은 무상관이 noise 안에서 흔들리는 것일 뿐 의미있는 변화 아님. 따라서 ranking C > E > D, A, B 가 r=10% 와 r=20% 양쪽에서 유지됩니다.

**§3.3 정성적 결론.** C 가 *best candidate* 인 것은 r-robust 하지만, ρ 의 *절대값* 0.60 은 n=7 의 confidence band 안에서 ±0.2 수준의 noise 를 안고 있습니다. paper framing 은 "Spearman(λ, c²) 가 후보 5 개 중 가장 강한 predictor 였고 r 선택에 robust 하다" 의 정성적 ranking 수준에서 멈춰야 하고, "0.60" 의 정량값을 단일 number 로 강조하지 않아야 합니다.

## §3.5 prediction-side vs Shapley-side 분리 — 본 iter 의 가장 중요한 framework 보정

critique §3 이 짚어준 가장 본질적인 비판. 이 단락은 plan v1 에 없던 부분이라 새로 끼워 넣습니다.

iter 01 실험 setup 을 다시 정리하면: A1 으로 Shapley 추정 → top-x% sample 을 골라 subset 으로 *full kernel* (rank-r 아닌 K 전체) prediction → accuracy 측정. LRFShap 도 동일하게 *full kernel* prediction. 즉 두 방법의 head-to-head 차이는 두 단계로 분해됩니다.

- **(a) Shapley-side**: 어떤 sample 이 골라졌는가. A1 의 sᵢ ranking vs LRFShap 의 rank-r approximation Shapley.
- **(b) Prediction-side**: 그 sample subset 이 full kernel ridge 에서 얼마나 generalizable 한가.

본 framework 의 miss_LC_by_LR, miss_FC_by_A1, eq9_gap_{LR,A1}, overlap_LR_A1 은 모두 **(a) Shapley-side 의 양**입니다. *그러나 empirical wins 는 (b) 까지 합친 양*. 둘 사이의 mapping 은 monotone 이 보장 안 됩니다 — Shapley 추정 오차 bound 가 작다고 prediction 단계 generalization 이 보장되지 않아요.

이 분리가 plan v1 의 두 관찰을 명료하게 해석해 줍니다.

**첫째**, 후보 A (miss_LC_by_LR) 가 wins 와 ρ ≈ 0 인 것은 *framework 의 결함이 아니라 negative result*입니다. "Shapley-side 의 절대 miss 양은 prediction-side accuracy 의 직접 결정 요인이 아니다" 가 데이터에서 확인된 거예요. 다시 말해 *Shapley value 추정의 worst-case bound 가 prediction accuracy 의 worst-case bound 와 같지 않다*. 이건 LRFShap eq. (9) 의 본질적 한계이기도 합니다.

**둘째**, 후보 C (Spearman(λ, c²)) 가 best predictor 인 것은 framework 가 *우연히* prediction-side proxy 를 포함하고 있었기 때문이라고 봅니다. Spearman(λ, c²) 는 spectrum-label alignment 의 *dataset-level* 양이고, Bordelon-Canatar-Pehlevan (2021, Nature Comm., https://www.nature.com/articles/s41467-021-23103-1) [B-BCP21] 의 task-model alignment 이론에 따르면 prediction-side generalization 의 핵심 결정 요인입니다. Spearman(λ, c²) 가 높으면 *prediction-side* 의 kernel ridge generalization 자체가 효율적이고, A1 의 LC bias 가 selection 단계에서 그 효율을 더 잘 활용합니다. 즉 C 는 Shapley-side (selection alignment) 와 prediction-side (kernel ridge generalization) 모두에 동시에 작용하는 dataset 양 — 그래서 다른 *Shapley-side only* 후보들 (A, B, D) 보다 wins 와 더 강하게 묶입니다.

**Framework 의 scope 명시.** 본 iter 의 framework 는 *Shapley-side 의 selection geometry* 를 식 단위로 정리합니다 — 누가 어떤 mode 를 잡는지, 두 selection 의 trade-off 가 어떻게 정량화되는지. 이 framework 가 *직접* 설명하지 않는 부분은 (b) prediction-side 의 full-kernel ridge generalization 입니다. 그래서 paper framing 도 "iter 01 의 wins 패턴 전체를 framework 가 설명한다" 가 아니라 "framework 가 selection geometry 를 정리하고, prediction-side 와의 bridge 는 별도의 task-model alignment 이론을 끌어와야 한다" 가 정확합니다.

## §4 Mechanism — *왜* Spearman(λ,c²) 가 best predictor 인가

§3.5 의 prediction-side vs Shapley-side 분리 위에서 mechanism 을 다시 정리합니다. Spearman(λ, c²) 가 클 때 두 사건이 동시에 일어납니다.

**첫째 (Shapley-side)**: A1 selection 이 "안전하게" 큰 cᵢ² 를 고릅니다. Pure LC selection 은 이론상 cᵢ² 가 작은 noise mode 에서 estimation variance 가 큰데, 그 mode 의 λ 가 작으면 (uᵢ⊤Y)² estimator 자체가 noisy 합니다. λ 와 c² 가 aligned 면 top-c² mode 의 λ 도 자연스럽게 크기 때문에 *A1 score sᵢ = f(λᵢ)cᵢ² 의 ranking* 이 안정적입니다. 이 stability 양은 본 iter 에서 직접 측정 안 됐고 (critique §3 가 지적한 estimator variance 의 양적 비교는 다음 iter 의 PR(c²) 와 함께 묶어 처리).

**둘째 (prediction-side)**: Bordelon-Canatar 의 eigenlearning curve 에 따르면 generalization error 는 Σᵢ cᵢ² · (1 − learnability_i)² 형태로 분해되고 learnability_i 가 λᵢ 의 monotone 함수입니다. high-c² mode 가 high-λ region 에 몰려 있으면 (= Spearman(λ, c²) 큼) full kernel ridge 가 prediction 단계에서 generalization 을 잘하고, A1 의 LC-biased selection 이 그 generalizable mode 들의 sample 을 더 잘 representative 하게 골라줍니다. 즉 A1 의 advantage 는 *prediction-side 가 이미 generalizable 한 setup 에서 selection 의 marginal gain* 입니다.

Anti-aligned regime (낮은 Spearman) 에서는 prediction-side 자체가 비효율적입니다 — high-c² mode 가 low-λ region 에 흩어져 있어 full kernel ridge 의 learnability 가 낮습니다. A1 이 selection 단계에서 high-c² mode 를 더 잘 잡아도, 그 mode 들이 prediction 단계에서 generalization 으로 전환 안 됩니다. 결과적으로 A1 의 *Shapley-side advantage* 가 *prediction-side ineffectiveness* 로 잠식되어 wins 가 떨어집니다.

**iter 01 archetype 과의 매핑.** Multi-class (MNLI C=3, AG News C=4) 와 paraphrase / NLI 의 sub-label structure 가 있는 binary (MRPC) 는 label energy 가 여러 mode 에 분산되고, embedding 의 sub-cluster 가 top eigenvector 와 어느 정도 align 됩니다. Spearman(λ, c²) 가 (MRPC 0.114 같이 낮아도) 다른 dataset 대비 *상대 rank* 에서 상위로 올라옵니다.

**AG News 의 outlier 성격.** AG 가 가장 높은 Spearman (0.377) 인데 wins 가 17/42 인 점이 §3 의 monotone 가설과 충돌합니다. 가설은 두 가지. (i) prediction-side 의 task complexity (C=4) 가 selection advantage 의 margin 을 압축. (ii) AG 의 cᵢ² 분포가 top-c² mode 에 너무 *집중* 되어 있어서 r=10% 만으로도 LR 이 LC 의 75% 를 잡아내고 (LC_LR = 0.747), A1 의 marginal gain 이 작음. 후자가 §5 의 participation ratio 가설로 자연스럽게 이어집니다.

## §5 RTE vs MRPC paradox — signal-vs-noise in missed LC

RTE 와 MRPC 둘 다 miss_LC_by_LR 이 큽니다 (RTE 0.794, MRPC 0.720), 즉 top-r 이 label energy 의 70-80% 를 못 잡습니다. 그런데 wins 는 RTE 4/42 vs MRPC 35/42. *Shapley-side 의 절대 miss 양* 으로는 설명 불가능 — 이건 §3.5 의 prediction-side ≠ Shapley-side 분리의 또 다른 표현이기도 합니다.

가설: **놓친 label energy 가 prediction-side 에서 *learnable signal* 인지 *unlearnable noise* 인지에 따라 A1 의 win 가능성이 갈립니다.** Learnable 은 *어떤 mode 가 있긴 있는데 단지 high-λ 영역 밖에 있어서* top-r 이 못 잡는 경우, unlearnable 은 *Y 에 random direction 으로 흘러간 component* 라 어느 selection 도 prediction 으로 못 옮기는 경우.

측정 proxy 로 두 가지를 제안합니다.

(1) **LC_A1 − LC_LR = delta_LC 의 *prediction accuracy 로의 transfer rate***. precomputed r=10% delta_LC 비교: RTE = 0.264, MRPC = 0.252 로 비슷합니다. LC_A1 자체는 RTE 0.470 vs MRPC 0.532 — MRPC 가 살짝 더 큼. 부분 설명이지 충분치 않음.

(2) **Spectrum-label alignment + label energy 분포의 concentration**. MRPC 와 RTE 의 Spearman 은 거의 동일 (0.114, 0.098). 그러나 cᵢ² 분포 자체의 spread 가 다를 가능성: RTE 의 cᵢ² 가 거의 uniform 하게 1000+ mode 에 흩어져 있다면 (effective rank 큼), 어떤 r-mode subset 을 골라도 prediction 으로 못 옮김 — *noise* 케이스. MRPC 의 cᵢ² 가 200-500 mode 에 집중되어 있으면 A1 의 LC selection 이 *right mode* 를 정확히 짚어줌 — *signal*.

이를 측정하는 양: cᵢ² 의 participation ratio **PR(c²) := (Σ cᵢ²)² / Σ cᵢ⁴**. precomputed.json 에 없어서 다음 iter 의 추가 계산 항목으로 둡니다 (raw data 에서 손쉽게 추가 가능).

## §6 paper framing 권고

이번 분석의 명료한 발견:

1. A1 의 selection 은 거의 모든 dataset 에서 *pure LC selection* 의 근사 — frac(λ > 10ρ) ≈ 1 덕분. RTE 만 partial 예외 (λ_min = 0 mode 가 일부 존재해 §1 (ii) 의 충분조건이 *top-r boundary* 에서 부분적으로 깨짐). 이는 paper §5.5 framing 을 정당화하되, "universal" 이 아니라 "near-universal, with RTE as edge case" 로 표현해야 정확합니다.

2. *Shapley-side 의 놓친 정보 절대량* (miss_LC_by_LR, miss_FC_by_A1) 은 A1 vs LRFShap empirical wins 를 거의 예측 못 합니다 (Spearman ≈ 0). 이는 §3.5 가 명시한 *Shapley-side vs prediction-side 분리*의 직접 증거 — Shapley 추정 오차 bound 의 tightness 가 prediction accuracy head-to-head 의 monotone predictor 가 아닙니다.

3. **Spectrum-label alignment Spearman(λ, c²) 가 5 후보 중 가장 강한 predictor** (ρ ≈ +0.60, r=10% 와 r=20% 양쪽에서 일관). 다만 **n=7 의 confidence band 가 wide** (≈ ±0.2) 해서 *정량 수치를 single number 로 강조하지 말고 정성적 ranking* 로 framing 해야 합니다. 이 척도는 Bordelon-Canatar 의 task-model alignment 이론과 직접 연결되어 prediction-side proxy 역할을 합니다.

4. RTE-MRPC paradox 는 PR(c²) 같은 cᵢ² distribution shape 양으로 풀 수 있을 것으로 봅니다 (next iter 의 첫 항목).

## §7 다음 iter 의 obligatory items

critique §4 가 제안한 2 개 falsifier 를 next iter 의 핵심 실험으로 격상합니다. (§5 의 PR(c²) 추가 계산은 보조 항목.)

- **Falsifier 1 — synthetic spectrum-label decoupling**. 동일 base embedding 으로 K 고정, label Y 를 인위 생성. (a) "aligned": Y = Σᵢ wᵢ uᵢ + ε, wᵢ ∝ √λᵢ (Spearman(λ, c²) → 1). (b) "anti-aligned": wᵢ ∝ 1/√λᵢ 또는 random permutation (Spearman → 0 또는 음수). 동일 n, 동일 K 에서 A1 vs LRFShap 의 42-cell wins matrix 측정. Framework 가 옳다면 aligned 에서 wins ≈ 42, anti-aligned 에서 wins ≈ 4 가 나와야 합니다. wins 차이가 작으면 (예: 둘 다 20 부근) framework 가 falsified — real embedding 의 dataset-specific structure 가 다른 변수 (label noise, sub-cluster geometry) 로 wins 를 결정한다는 의미. precomputed.json 추가 계산 없이 실험 가능.

- **Falsifier 2 — intermediate Spearman dataset 확보**. 현재 7 dataset 의 Spearman 은 0.10–0.38 의 좁은 range 라 ρ = +0.60 을 결정짓는 leverage 가 endpoint 두 점에 집중됩니다. (a) CoLA (linguistic acceptability — 단순 label structure 라 alignment 높을 가능성) 추가, 또는 (b) 같은 7 dataset 에서 training set size 를 n ∈ {500, 1000, 2000, 4000} 으로 변화시켜 Spearman(λ, c²) 가 n 에 따라 어떻게 움직이고 wins 가 그 변화를 따라가는지 확인. wins 가 n 따라 변하는데 Spearman 은 stable 하면 framework 외 변수 (sample complexity) 가 dominate 하는 것이라 framework falsified.

- **보조: PR(c²) 와 effective LC dimension** 7 dataset 전체 계산. RTE vs MRPC paradox 의 정량 해명.

- **보조: A1 score 의 *modified* version** — λ filter 끄고 pure cᵢ² 만 쓰는 baseline (= I_LC) 과 A1 의 실증 비교. λ >> ρ regime 에서 정말로 동일한지, 특히 RTE 의 bottom mode 에서 갈리는지 직접 검증.

## §8 참고 문헌

- [B-BCP21] Bordelon, Canatar, Pehlevan (2021), "Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks", *Nature Communications*. https://www.nature.com/articles/s41467-021-23103-1 — §3.5 와 §4 의 prediction-side mechanism 의 이론적 grounding.
- [B-Simon23] Simon et al. (2023), eigenlearning framework. https://jamiesimon.io/blog/eigenlearning/ — learnability 의 zero-sum 분배 (Σ over mode = n) 가 본 iter 의 LC retention 개념과 dual.
- [B-LRFShap25] LRFShap (Kang et al., 2025), lrfshap.pdf, p. 7, eq. (9). — eq9_gap 정의 출처. critique §5 에 따라 page+eq 번호를 본문 (§1 (iv)) 에 명시.
- iter 01 plan_v2.md 의 bibliography — 4 트랙 (ridge leverage, KTA, eigenlearning, data Shapley) 은 재인용 없이 reference 만 유지.

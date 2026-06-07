# Iteration 04 — Executive Summary (v2, mnli cls90 INV 완료 반영)

## 0. 연구 질문과 setup 한 단락

같은 정도의 class imbalance 를 train 측 (valbal: train imbalanced + val balanced) vs val 측 (trainbal: train balanced + val imbalanced) 에 가했을 때, FreeShap (Wang et al. 2024 의 inv approximation) 과 LR (lrfshap workshop 의 top-r eigen approximation) 이 *collapse* 하는 mechanism 은 무엇이며, A1 (label-aware top-r eigen, sᵢ = (λᵢ/(λᵢ+ρ))² · ‖uᵢ⊤Ỹ‖² 의 score 로 mode 선택) 이 회복시키는 *정확한 quantity* 는 무엇인가? 왜 trainbal 의 상당 case 에서 A1 도 random selection 한테 패배하는가?

실험은 7 NLP dataset (MR, SST-2, MRPC, QQP, RTE, AG News, MNLI) × valbal/trainbal × imbalance level (balanced + mild 70/30 또는 cls60/55 + extreme 90/10 또는 cls90/85) 의 *모든 cell 완성*. 마지막 mnli cls90 trainbal extreme 의 FreeShap baseline (INV) 이 2026-06-03 22:18 KST 에 완료되어 7-dataset trainbal 전체 fully done.

## 1. 용어와 수식 정의

분석에 등장하는 모든 quantity 를 한 곳에 모았다. 이후 가설 표에서 이 정의를 그대로 쓴다.

### 1.1 확률·label 관련

- **P_train(c)**: train set 의 *class c 의 empirical 비율*. 분포가 아니라 분포의 한 값 (즉 0 과 1 사이 숫자). 모든 c 에 대한 P_train 의 vector 가 분포 자체. 예: SST-2 train 5000 개 중 class 0 이 4500 개, class 1 이 500 개 → P_train(0) = 0.9, P_train(1) = 0.1.
- **max_c P_train(c)**: train 의 majority class 의 비율. valbal 의 90/10 setting 이면 0.9. *imbalance 의 깊이* 의 직접적 measure.
- **P_val(c)**: 동일하나 val set 위.
- **P_S(y)**: top-Shapley 가 *선택한 subset S 의 label 분포*. y = 1 의 비율.
- **w(c) = P_val(c) / P_train(c)**: BBSE (아래) 의 importance weight. valbal 의 train 90/10 + val 50/50 이면 w(maj) = 0.55, w(min) = 5. trainbal 의 train 50/50 + val 90/10 이면 w(maj) = 1.8, w(min) = 0.2.

### 1.2 KL divergence (label shift 의 magnitude)

KL(P ‖ Q) = ∑_c P(c) · log(P(c) / Q(c))

두 확률 분포 P, Q 의 *비유사도* 의 표준 측도. 0 이면 P = Q, 클수록 P 와 Q 가 떨어져 있음. *asymmetric* — KL(P‖Q) ≠ KL(Q‖P) 일반적으로.

- **KL(P_val(y) ‖ P_train(y))** (= `kl_val_train`): val 분포와 train 분포의 label shift 크기. trainbal extreme (val cls90_05_05, train cls33) 이면 KL 약 0.65.
- **KL(P_val(y) ‖ P_S(y))** (= `kl_val_S`): val 의 *target* 분포와 *선택된 subset* 의 분포의 KL. small budget 에서 P_S 가 거의 1-class 로 collapse 하면 KL → +∞ (또는 매우 큼). *얼마나 잘못 골랐는지* 의 proxy.

### 1.3 Pearson r, Spearman r, p-value, α, Bonferroni

두 변수 (X = predictor, Y = outcome) 의 상관성 측정.

- **Pearson r** ∈ [−1, +1]: 두 변수의 *linear* 상관계수. r = +1 이면 완벽한 양의 선형, −1 이면 완벽한 음의 선형, 0 이면 무상관. r² 가 분산 설명률.
- **Spearman r** ∈ [−1, +1]: 두 변수의 *rank* 상관계수. linear 가정 없이 monotone 관계만 측정. linear 가정이 깨질 때 (outlier 있을 때) Pearson 보다 robust.
- **p-value**: "no correlation (null hypothesis)" 일 때 관찰된 r 만큼의 r 또는 더 큰 r 이 나올 확률. *작을수록* 우연이 아니라는 strong evidence.
- **α (significance level)**: p-value 의 cutoff. p < α → "reject null" = "유의미한 상관 있다" 라고 결론. 일반 α = 0.05.
- **Bonferroni correction**: K 개 가설을 동시에 검증할 때 false positive 누적을 막기 위해 *각 가설의 α 를 α/K 로 조정*. iter_04 의 main hypothesis 가 H1, H2, H3, H4 의 4 개이므로 α = 0.05 / 4 = **0.0125**. 즉 p < 0.0125 일 때만 *confirm*. **α 는 작을수록 더 strict — 즉 더 강한 evidence 요구**.
- **95% CI** (Fisher z transform 으로 계산): r 의 *95% 신뢰구간*. CI 가 0 을 cross 하지 않으면 r ≠ 0 이 significant 라는 다른 형태의 표현.

### 1.4 KTA (Kernel-Target Alignment) — lens 2 의 핵심 quantity

KTA (Cristianini, Shawe-Taylor, Elisseeff, Kandola 2001, NIPS, eq. (1)) 의 정의:

KTA(K, Y) = ⟨K, Y Y⊤⟩_F / ( ‖K‖_F · ‖Y Y⊤‖_F )

- K: n × n kernel matrix (우리 setting 의 eNTK)
- Y: one-hot label matrix (n × C, C = class 수). Ỹ = centered (mean-subtracted) version.
- ⟨A, B⟩_F = ∑_{i,j} A_{ij} B_{ij}: Frobenius inner product
- ‖A‖_F = √⟨A, A⟩_F: Frobenius norm

직관: *kernel 이 task (label structure) 와 얼마나 잘 정렬되는지*. KTA = 1 이면 K ∝ Y Y⊤ (즉 kernel 이 label structure 와 완전 일치 — same class 의 sample 쌍에 K = 1, different class 쌍에 K = 0). KTA = 0 이면 무관.

- **KTA_full**: 전체 NTK kernel K = ∑_i λᵢ uᵢ uᵢ⊤ (full eigen 분해) 의 KTA.
- **KTA(r) = KTA_topr**: top-r eigen mode 만 보존한 truncated kernel K_r = ∑_{i ≤ r} λᵢ uᵢ uᵢ⊤ 의 KTA. *r 이 작을 때 alignment 가 얼마나 보존되는지*.
- **kta_train_r10**: train kernel + train Ỹ 에 대한 KTA(r=10%). *train spectrum 의 top 10% mode 가 train task 와 얼마나 align 하는지*.
- **kta_val_r10**: train kernel 의 top-r eigenvector 를 val 에 *Nyström extension* (Williams-Seeger 2000, eq. (4) 의 √n factor 포함) 으로 transfer 한 후 val Ỹ 에 대해 계산한 KTA.
- **kta_gap_r10** = kta_train_r10 − kta_val_r10: *train task ↔ val task 의 alignment 격차*. trainbal extreme 처럼 train Ỹ 와 val Ỹ 분포가 다를수록 이 gap 이 음수 방향으로 큼.

### 1.5 BBSE (Black-Box Shift Estimator) — lens 1 의 추정 기법

Lipton, Wang, Smola (2018, ICML), eq. (4)-(5) + Theorem 3. label shift 의 importance weight w(y) = P_val(y) / P_train(y) 를 *labels 없는 val* 위에서 추정하는 기법. *black-box classifier* 의 train 위 confusion matrix C 와 val 위 prediction 분포 q̂_val(ŷ) 만 사용:

ŵ = C⁻¹ q̂_val

여기서 C_{ij} = P_train(ŷ = i | y = j) (joint 가 아닌 confusion matrix; classifier 가 class j 를 i 로 잘못 분류하는 conditional 비율).

우리 분석에선 oracle P_train, P_val 을 알고 있어 BBSE 의 *추정* 자체는 직접 w = P_val / P_train 으로 계산 (estimator 의 fit 만 baseline classifier = "always predict train majority" 로 sanity check 위해 monitor).

### 1.6 Outcome (acc gap) 들

- **balanced acc** = per-class recall 의 평균 = (1/C) ∑_c recall_c. label shift 에 *invariant* — val 의 class 비율이 바뀌어도 (per-class recall 자체는 안 바뀌면) balanced acc 안 바뀜.
- **gap_top_random_balanced**: balanced_acc(top-Shapley selection) − balanced_acc(random selection). > 0 이면 top-Shapley 가 random 보다 나음.
- **recovery_sel5**: balanced_acc(A1, sel=5%) − balanced_acc(LR, sel=5%). A1 이 LR 대비 얼마나 회복했나. > 0 = A1 better.
- **LR_loss_vs_random_balanced_sel5**: balanced_acc(random, sel=5%) − balanced_acc(LR, sel=5%). LR 이 random 보다 얼마나 *못한지*. > 0 = LR worse than random.

## 2. 5 가설의 결과 표 (mnli cls90 INV 포함)

각 가설의 predictor (X) 와 outcome (Y) 정의:

- **H1**: X = `kl_val_S` (선택된 subset 의 label 분포와 val 분포의 KL), Y = `gap_top_random_balanced` (top-Shapley 의 balanced acc − random 의 balanced acc; sel ∈ {1,2,3,5}% 평균; r=10%; A1 method)
- **H2**: X = `max_c P_train(c)` (train majority class 의 비율), Y = `A1 − LR balanced acc gap` (sel=5%, r=10%)
- **H3**: X = `kta_gap_r10` (train Ỹ 와 val Ỹ 의 alignment 격차), Y = `recovery_sel5` (= H2 의 Y 와 동일)
- **H4**: X = `kta_train_r10` (train kernel + train Ỹ 의 top-r=10% KTA), Y = `LR_loss_vs_random_balanced_sel5` (random 의 balanced acc − LR 의 balanced acc; sel=5%; r=10%)
- **H4b**: X = `kta_gap_r10` (= H3 의 X), Y = H4 의 Y (trainbal regime 에서의 gap 효과 점검)
- **H5**: X = `d_eff_ratio_r10` (effective dimension 의 r=10% truncate ratio), Y = `recovery_sel5`

| H | regime | n | r (Pearson) | p (Bonf. α=0.0125) | 결정 |
|---|---|---|---|---|---|
| **H1** | trainbal | 14 | −0.001 | 0.997 | **reject** |
| **H2** | valbal | 21 | +0.71 | 3.2 × 10⁻⁴ | **confirm** |
| **H3** | valbal | 21 | −0.71 | 3.5 × 10⁻⁴ | **confirm** |
| **H4** | trainbal | 14 | +0.99 | 7.6 × 10⁻¹¹ | **confirm ★** (dataset confounding 의심) |
| **H4b** | trainbal | 14 | −0.22 | 0.46 | partial |
| **H5** | valbal | 21 | +0.14 | 0.54 | **reject** |

mixed model β (dataset random effect 흡수 후의 effect) 가 산출된 경우:
- H1: mixed β = −0.0005 [95% CI −0.0033, 0.0023] — CI 가 0 cross → 효과 없음, primary reject 와 일치
- H2: mixed β = +0.255 [95% CI 0.188, 0.322] — CI 가 0 cross 안 함 → dataset 흡수 후에도 effect 살아남음, primary confirm 강화
- H3, H4, H4b, H5: lens2 분석에서 mixed β 산출 누락 (NaN) — round 2 에서 보강 필요

stratified (imbalance level 별):

| H | level | n | r | p | 결정 |
|---|---|---|---|---|---|
| H1 | mild | 14 | −0.001 | 0.997 | reject |
| H2 | mild | 14 | +0.48 | 0.084 | partial (p > 0.0125) |
| H2 | extreme | 7 | +0.80 | 0.030 | partial (n 작아 CI wide) |
| H3 | mild | 14 | −0.68 | 0.007 | confirm |
| H3 | extreme | 7 | **+0.62** | 0.138 | partial (**부호 반대**) |
| H4 | mild | 14 | +0.99 | 7.6 × 10⁻¹¹ | confirm (= main, trainbal mild + extreme 가 *모두* mild 로 분류된 경우; 별도 partition 안 됨) |

stratified (imbalance level 별):

| H | level | n | r | p | 결정 |
|---|---|---|---|---|---|
| H1 | mild | 14 | −0.001 | 0.997 | reject |
| H2 | mild | 14 | 0.48 | 0.084 | partial (p > 0.0125 — strict cutoff 못 넘김) |
| H2 | extreme | 7 | 0.80 | 0.030 | partial (n 작아 CI wide) |
| H3 | mild | 14 | −0.68 | 0.007 | confirm |
| H3 | extreme | 7 | +0.62 | 0.138 | partial (**부호 반대**) |

mild vs extreme 의 stratification 이 *균일하지 않음* — H3 의 부호가 mild 에서 음, extreme 에서 양으로 뒤집힘. 이 inversion 은 *small n (7) 의 chance* 일 수도 있고 진짜 regime split 일 수도 — round 2 의 mixed model 로 점검 필요.

## 3. 가설별 깊이 해석 (왜 됐는지/안 됐는지)

### H1 (label shift KL → trainbal random-loss): **reject 이나 단변량 reject 일 뿐**

가설: trainbal regime 에서 top-Shapley 가 random 한테 지는 cell 은 *label shift 가 큰 cell* 일 것이다 — 즉 train 과 val 분포가 떨어진 setting 일수록 잘못된 sample 을 고른다.

결과: r = −0.001, 사실상 무상관. 왜?

`lens1_cell_kl.csv` 의 small-sel cell 을 보면 `kl_val_S` 가 19.34 또는 11.25 의 *plateau* 에 몰린다. 즉 small sel% (1, 2%) 에서 top-Shapley 가 *거의 100% majority class* 로 collapse 해 P_S 가 한 점에 모이고, 이로 인해 between-setting 분산이 outcome 분산보다 훨씬 작아져 correlation 추정 자체가 무력화된다. 단변량으로는 신호 부재라는 결론이지만, label shift 가 *무의미한 게 아니라*, lens 2 의 KTA 같은 spectral quantity 와 *결합한 interaction* 으로만 작동할 가능성이 큼 (round 2 에서 OLS `outcome ~ kl_val_S × kta_train_r10` 으로 점검 예정).

### H2 (train majority → A1 recovery): **confirm**

가설: valbal 에서 train imbalance 가 깊을수록 (max_c P_train(c) 클수록) LR 은 더 큰 collapse, A1 의 회복 폭도 크다.

결과: r = 0.71, p = 3.2 × 10⁻⁴, mixed β = 0.255 [0.188, 0.322]. mixed model 의 CI 가 0 을 cross 하지 않음 → dataset 효과 흡수 후에도 *진짜* 효과로 남음. **label-side mechanism 의 strong evidence**.

해석: A1 의 score sᵢ = (λᵢ/(λᵢ+ρ))² · ‖uᵢ⊤Ỹ‖² 의 ‖uᵢ⊤Ỹ‖² 부분이 minority class 의 mode 를 *명시적으로* 골라낸다. train 이 imbalanced 일수록 minority mode 가 LR (λ 기반 선택) 에서 누락되는데, A1 가 이를 회복한다.

### H3 (train-val alignment gap → A1 recovery): **confirm**

가설: train Ỹ 와 val Ỹ 의 alignment 격차 (`kta_gap_r10`) 가 클수록 LR 은 train task 에 specialize 되어 val 에서 collapse, A1 은 *train Ỹ* 를 직접 score 에 쓰므로 (val Ỹ 는 unknown) 동일한 alignment gap 에서도 회복 가능.

결과: r = −0.71, p = 3.5 × 10⁻⁴ (음의 부호는 *gap 이 음수 방향으로 클수록 recovery 가 양수* 라는 의미 — gap = train alignment − val alignment 이므로 train spectrum 이 val task 와 *덜* 정렬될수록 LR 더 collapse). **spectral-side mechanism 의 strong evidence**.

### H4 (top-r KTA on train → trainbal LR loss): **confirm 이나 dataset confounding 의심**

가설: trainbal 에서 LR 이 random 한테 지는 cell 은 train kernel 의 top-r KTA 가 *큰* cell 이다 (train task 위에서 spectrum top 이 잘 정렬돼 있지만 그게 val task 에 안 맞음).

결과: r = 0.99, p = 7.6 × 10⁻¹¹ — *너무 sharp*. Spearman 만 보면 0.85 < Pearson 0.99 라 *outlier-driven linear fit* 의 가능성. 결정적으로 trainbal 의 train 이 모든 setting 에서 balanced 고정 (cls33_33_33, cls25_25_25_25 등) 이라 `kta_train_r10` 가 *dataset 마다 단일 값* 으로 degenerate — 14 setting 에 unique 값이 7 개 (= 7 dataset). 즉 fit 이 *kta_train_r10 → LR_loss* 가 아니라 *dataset identity → LR_loss* 의 흡수일 가능성. mixed model 의 β (dataset random effect 흡수 후 effect) 가 plan 의 secondary tier 였으나 lens2 의 산출에 NaN — round 2 에서 statsmodels `MixedLM` 으로 직접 측정 필요.

이 의문 *해소 후* 에야 H4 가 mechanism evidence 인지 dataset artifact 인지 확정.

### H5 (d_eff_ratio → A1 recovery): **reject — lens 3 의 inherent 한계**

`d_eff(ρ) = ∑_i λᵢ / (λᵢ + ρ)` 의 r-truncated version 의 ratio 가 모든 setting 에서 거의 0.100 (d_eff_full ≈ 5000, ratio = 500 / 5000 = 0.1). 이는 NTK spectrum 의 단조감소 특성에 따른 *inherent* 결과이지 setting 사이 변별력이 없는 것. lens 3 (d_eff) 는 본 분석에선 활용 불가 — plan §4 의 *brief* 결정이 옳았음.

## 4. iter_03 framework 와의 연결 (확장)

iter_03 의 7 quantity (LC_LR, PR(c²), Spearman(λ, c²), overlap_LR_A1, train_majority, LR_predict_majority_frac, C) 가 iter_04 의 lens 에 어떻게 mapping 되는지 — *quantity 정의* 도 함께.

| iter_03 quantity | 정의 | iter_04 의 위치 | 검증 상태 |
|---|---|---|---|
| **LC_LR(r)** | LR 의 selected sample 들의 *learning curve 의 r-truncation 시점에서의 confidence*. iter_03 §1 에서 "small sel 에서 LR 의 majority recall" 의 proxy 로 사용. | lens 2 의 LC_train(r), LC_val(r) | round 2 에서 r ∈ {5,10,15,20} 일관성 측정 |
| **PR(c²)** | participation ratio of c² values: PR(x) = (∑x)² / ∑x². iter_03 의 Cond 2 cause-side mechanism — small budget 에서 어느 mode 가 dominant 한지 측정 | lens 2 의 pr_c2_train + KTA(r)/KTA_full ratio | H3, H4 가 KTA quantity 의 cause-side spectral 정량화로 *지지*. complementary 인지 redundant 인지는 round 2 에서 |
| **train_majority** | max_c P_train(c). Cond 2 verify-side | lens 1 의 max_c P_train(c) | H2 가 *label-side* 와 *spectral-side* (H3) 분리 가능성 보여줌 |
| **LR_predict_majority_frac** | LR 이 val 에서 majority class 로 예측한 비율. Cond 2 의 effect-side trigger | lens 1 의 BBSE-weighted risk | round 2 에서 측정 |
| **C** | class 수. Cond 3 의 SNR axis | lens 1 의 KL + √(C-1) factor | mapping 약화 (next_directions 로 이관) — C 와 KL 은 independent mechanism |

**핵심 mapping**: iter_03 의 *Cond 2 (PR(c²) cause-side mechanism)* 은 iter_04 의 lens 2 KTA quantity 의 spectral 정량화로 강하게 지지된다 (H3, H4 confirm). 단 plan §6 의 "동치" 주장은 round 2 의 PR(c²) ↔ KTA(r)/KTA_full 의 *redundancy 측정* 결과에 따라 *complementary partition* 으로 약화될 수 있음.

## 5. 핵심 결론 3 개

1. **Spectral KTA gap (lens 2) 가 가장 sharp 한 single predictor** — valbal 에선 H3 (r = −0.71) 이 A1 recovery 의 약 절반 분산 설명, trainbal 에선 H4 (r = 0.99) 이 LR collapse 의 거의 모든 분산 설명. 단 H4 의 dataset confounding 의문 해소 필요.

2. **Label shift (lens 1) 는 단독으론 insufficient** — KL 단변량으로는 trainbal random-loss 설명 못함 (H1 reject). lens 2 와 *interaction* 으로만 작동할 가능성.

3. **A1 의 recovery 는 train-side imbalance 와 cross-Ỹ alignment 두 축 모두에서 measurable** — paper main story 는 "A1 = label-aware mode selection 이 imbalance + small budget regime 에서 LR collapse 의 *spectral-side 회복*" 으로 잡힘.

## 6. Round 2 의 우선 task

| # | task | 시간 | 무엇을 점검 |
|---|---|---|---|
| 1 | `experiments/lens_mixed_model.py` 신설 | 15-20 분, CPU | (i) H4 의 mixed β + within-dataset Spearman 으로 dataset confounding 점검, (ii) `outcome ~ kl_val_S × kta_train_r10` 의 interaction 항으로 H1 의 moderator 효과 검증, (iii) `Spearman(max_P_train, kta_gap_r10)` 의 H2-H3 redundancy 측정, (iv) H4 를 valbal 21 setting 에 regime-cross 적용 (mechanism generality), (v) (Q3) "trainbal random-loss 절반 case" 의 operational rate (`gap < −0.02` cell 비율) 계산 |
| 2 | paper main story arc 결정 | round 2 후 | scenario A ("lens 2 가 lens 1 흡수") vs scenario B ("complementary partition"). trigger: `Spearman(max_P_train, kta_gap_r10)` ≥ 0.7 → A, < 0.5 → B |
| 3 | iter_05 의 분기 (사용자 선택) | — | (Task A) KTA-based generalization bound 도출 (paper §Theory), (Task B) vision domain 확장, (Task C) training dynamics 의 epoch dependence |

## 7. 산출 file

- 분석 데이터: `state/iteration_04/grand_df.csv` (79,800 cells, mnli cls90 INV 포함), `lens1_table.csv`, `lens2_table.csv`, `eig_cache/` (52 npz)
- 통계 결과: `lens1_corr.csv`, `lens2_corr.csv`, `lens1_cell_kl.csv`
- 그림: `lens1_figs/H{1,2}_scatter.png`, `lens2_figs/H{3,4}_scatter.png`
- 문서: `plan.md` (7300 단어 v2), `critique.md` (8500 단어), `next_directions.md` (1700 단어), `critique_priorart.md` (보존)
- 스크립트: `experiments/build_grand_df.py`, `lens1_label_shift.py`, `lens2_kta_spectral.py`
- 전체 archive: `reports/iter_04.pdf` (45 pages)
- 데이터 PDF: `reports/lrfshap_vs_a1_trainbal.pdf` (43 pages, 18 settings, 7 dataset trainbal fully done), `reports/lrfshap_vs_a1_valbal.pdf` (27 pages, 21 settings)

# Iteration 04 — Next directions (round 2 + paper draft outline + iter_05 task)

본 문서는 `state/iteration_04/critique.md` §9 (executor 1st-round 분석) 의 결론을
받아, 두 시야 — (i) **iter_04 round 2 의 분석 task** (수정사항 31-41), (ii) **paper
draft 의 main story arc** 의 두 candidate scenario, (iii) **iter_05 의 우선 task**
(KTA quantity 의 theory bound, vision domain 확장, training dynamics) — 의
우선순위를 정리합니다. 분량 약 1500 단어.

## §1. Round 2 분석 task — `lens_mixed_model.py` 의 신설

iter_04 plan v2 §3.3 의 statistical 3-tier procedure (primary = setting 단위 Pearson +
Fisher CI, secondary = cell 단위 mixed model β + CI, tertiary = dataset 별 within-
dataset Spearman 분포) 중 primary 만 1st-round 에서 구현되었습니다. round 2 의 첫
task 는 secondary + tertiary 를 *retroactively* 산출하는 단일 script 의 신설입니다.

**`experiments/lens_mixed_model.py` 의 spec**:

입력: `state/iteration_04/grand_df.csv` (79,700 cell), `lens1_table.csv`,
`lens2_table.csv`, `lens1_cell_kl.csv`.

산출: `state/iteration_04/round2_corr.csv` — 모든 H1-H4 의 (primary, secondary,
tertiary) 의 통합 row. column schema: `hypothesis_id, predictor, outcome, regime,
imbalance_level, n_settings, primary_pearson_r, primary_fisher_ci_lo, primary_fisher_
ci_hi, primary_spearman_r, primary_p, mixed_beta, mixed_ci_lo, mixed_ci_hi, mixed_p,
within_dataset_spearman_median, within_dataset_spearman_iqr_lo, within_dataset_spearman_
iqr_hi, decision_3tier`.

각 H1-H4 의 추가 task:
- **H1**: trainbal regime 의 cell-level mixed model `gap_top_random_balanced ~
  kl_val_S + (1 | dataset)` (수정사항 11, 14, 39). H1 interaction extension —
  `gap ~ kl_val_S * kta_train_r10 + (1 | dataset)` 의 β_interaction (수정사항 32).
- **H2**: valbal 21 setting 의 cell-level mixed model `(A1-LR balanced acc gap) ~
  max_P_train + (1 | dataset)`. partial r² 분해 — `(A1-LR gap) ~ max_P_train +
  kta_gap_r10`, 두 predictor 의 각 partial r² + variance decomposition (수정사항 34).
- **H3**: valbal 21 setting 의 cell-level mixed model `recovery_sel5 ~ kta_gap_r10 +
  (1 | dataset)`. predictor redundancy check — `recovery_sel5 ~ kta_gap_r10 +
  max_P_train` 의 partial r².
- **H4**: trainbal 14 setting 의 cell-level mixed model `LR_loss_vs_random_balanced_
  sel5 ~ kta_train_r10 + (1 | dataset)`. fixed β 의 CI 가 zero 를 cross 안 하면
  dataset-confounding 후에도 mechanism evidence 유지. cross-regime test (수정사항
  35) — valbal 21 setting 의 `recovery_sel5 ~ kta_train_r10` 의 Pearson + Spearman +
  Fisher CI + mixed model. valbal r > 0.6 이면 H4 generality 확보.

추가 task (수정사항 33, 37, 41):
- **lens 1 primary direction**: setting-level `risk_w_proxy_sel5 ~ kl_val_train`
  (Lipton 2018 의 target → source 방향) 의 Pearson, 14 trainbal setting.
- **PR vs KTA redundancy**: `pr_c2_train ~ kta_train_r10` 의 valbal 21 + trainbal
  14 의 Spearman. ρ > 0.8 이면 Cond 2 의 PR-based 와 KTA-based 표현이 *경험적
  동치*.
- **(Q3) operational random-loss-rate 계산**: trainbal 14 setting 의
  `P(gap_top_random_balanced[A1, r=10%, sel ∈ {1, 2, 5}] < −0.02)` 의 dataset-별
  + setting-별 값. 사용자 directive 의 *절반* 표현이 quantitative 으로 얼마인지
  확정.

산출 figure: `state/iteration_04/round2_figs/`
- `H4_regime_cross.png` — valbal 21 + trainbal 14 의 `kta_train_r10 ↔ outcome`
  scatter 2-panel.
- `partial_r2_decomposition.png` — H2, H3, H4 의 multivariate partial r² stacked
  bar chart.
- `within_dataset_heterogeneity.png` — 7 dataset 별 within-dataset Spearman 의
  forest plot (median + IQR).

estimated cost: 1 CPU core, 15-20 분. statsmodels MixedLM dependency.

## §2. mnli cls90 INV cell rerun (executor 별도 round)

mnli cls90_05_05 의 trainbal INV (FreeShap) 가 누락 (사용자 비평 4) 으로 H4 의 n=15
가 아닌 14. 누락 cell 이 H4 의 *extreme x-value* (mnli 의 NTK spectrum 으로 결정되는
kta_train_r10, lens2_table 추정 mnli trainbal cls60 row 의 값 기준 cls90 도 동일
하므로 새 x-value 가 아님) 에 해당하지 않아도, *extreme y-value* (LR_loss 가
cls90 의 극단 imbalance 로 가장 클 expected) 의 fit 점이라 H4 fit 의 robustness
검증의 핵심.

**Rerun spec**: 기존 `n05_INV.sh` 또는 동등 script 의 mnli cls33_33_33 + valimb1000_
cls90_05_05 setting 만. method=FreeShap (TMC), seed=2026, n_train=5000, n_val=1000.
NTK pickle 은 이미 존재 (`imbalance_ntk/mnli/cls33_33_33/bert_seed2026_num5000_
valimb1000_cls90_05_05_signFalse.pkl`). 학습은 BERT base + GLUE-MNLI fine-tune 의
표준 pipeline (~ 1 day on single A100 / V100).

완료 후 round 3 — `lens_mixed_model.py` 재실행, H4 의 n=15 fit 의 Pearson r 비교.
r=0.99 가 유지되면 mechanism evidence 강화, r<0.9 로 떨어지면 mnli cls90 가 outlier
로 H4 의 strict-mechanism interpretation 약화.

## §3. Paper draft 의 main story arc — 두 candidate scenario

round 2 의 redundancy 정량화 결과에 따라 paper 의 main narrative 가 갈립니다.

### scenario A: "lens 2 가 dominant, lens 1 은 moderator"

조건: `corr(max_P_train, kta_gap_r10)` 의 Spearman > 0.7 AND `recovery_sel5 ~
kta_gap_r10 + max_P_train` 의 multivariate OLS 에서 max_P_train 의 partial r² <
0.05 (kta_gap_r10 의 partial r² > 0.4 의 절반 미만).

narrative: "label distribution shift (lens 1) 의 effect 는 본질적으로 *kernel-target
alignment* (lens 2) 의 *kernel-side 표현* 으로 흡수된다. trainbal 에서 LR top-r 의
collapse mechanism 은 *train kernel 의 top-r mode 가 majority class 에 dominantly
align* 되어 selected subset P_S 가 majority-only 로 degenerate 하는 것 — 이는 raw
label distribution P_train 의 majority probability max_c P_train(c) 와 monotone
이지만, *원리적* mechanism 은 NTK 의 spectral structure × label alignment 의 결합
인 KTA(r). A1 score 의 spectral filter `(λᵢ/(λᵢ+ρ))²` 가 LR top-cᵢ² 대비 *high-λ
mode* 만 강조해 collapse 회복 — 이 회복은 KTA(r) gap 의 함수로 monotone."

paper 구조: §1 Intro / §2 Related (BBSE, KTA, FreeShap, A1) / §3 Setup (controlled
imbalance benchmark) / §4 Empirical observation (valbal recovery + trainbal random-
loss) / §5 Spectral framework (KTA decomposition + A1 score derivation) / §6
Quantitative validation (H2, H3, H4 의 confirm + lens 1 흡수의 redundancy evidence)
/ §7 Theory (BCP21 generalization closed-form 의 우리 cell 단위 reformulation) /
§8 Discussion / §9 Limitations (H4 의 dataset confounding caveat + mnli cls90 rerun
결과 + lens 4 의 TMC variance bound 의 한 단락).

target venue: NeurIPS 2027 (deadline 추정 2027 5월) main track, or ICML 2027 (1월).

### scenario B: "lens 1 과 lens 2 의 complementary partition"

조건: `corr(max_P_train, kta_gap_r10)` 의 Spearman < 0.5 AND partial r² 분해에서
두 predictor 가 *각각* 0.15 이상 기여.

narrative: "label distribution shift 와 kernel-target alignment 는 *complementary
mechanism* 의 두 측면이다. lens 1 은 *global* label distribution shift 의 effect
를 capture 하고, lens 2 는 *local* kernel-side spectral concentration 의 effect 를
capture 한다. 두 mechanism 이 합산되어 collapse 의 full picture 를 형성. paper 의
main contribution 은 lens 1, lens 2 의 *quantitative separation* + 각각의
predictive metric (KL, KTA gap) 의 *operational threshold*."

paper 구조: scenario A 와 동일하나 §5 + §6 가 *two complementary mechanism* 으로
재구성. 두 lens 의 partial r² + joint variance explained 의 정량적 stacked bar
chart 가 main figure.

### scenario 결정

round 2 의 `lens_mixed_model.py` 결과 의 두 quantity — `corr(max_P_train,
kta_gap_r10)` 와 `partial r² of max_P_train in recovery_sel5 ~ both` — 가 결정.
1st-round 의 r(H2)=+0.71 와 r(H3)=−0.71 의 *비슷한 magnitude* 가 scenario A 의 약한
정황 (두 predictor 가 redundant 일 가능성) 이지만, 두 변수의 *cross-setting Spearman*
없이는 확정 불가. paper draft 작성 시작은 round 2 완료 후.

## §4. iter_05 의 우선 task (KTA quantity 의 theory bound + cross-domain robustness)

1st-round 결과 + round 2 의 mixed model 결과가 모두 H3/H4 의 mechanism evidence 를
강화하면, iter_05 는 *KTA quantity 의 theory bound* 의 도출과 *vision domain 확장*
의 두 줄기로 분기.

### iter_05 task A: KTA-based generalization bound 의 도출

핵심 question: cell 단위로 `gap_top_random_balanced ≤ f(kta_train_r10, kl_val_train,
sel, ρ)` 의 *explicit* upper bound 를 만들 수 있는가. 출발점은 Cortes-Mohri 2012
JMLR 의 centered KTA-based generalization bound `R(f) − R̂(f) ≤ O(√(complexity/n) /
KTA_centered)` 와 Garg 2020 NeurIPS arXiv 2003.07554 의 weighted ERM excess risk
의 `O(‖P_S − P_val‖_TV²)` 결합. selected subset 의 KTA `KTA(K_S, Ỹ_S)` 와 full
KTA `KTA(K_train, Ỹ_train)` 의 차이가 *selection rule* 의 dependence 의 직접 측정.

iter_05 task 분해:
1. KTA(K_S, Ỹ_S) 의 정확한 정의 (selected subset 의 sub-kernel) — `kta_selected.py`
   script.
2. KTA(K_train) − KTA(K_S) ≤ ? 의 *explicit* deterministic bound 의 도출 시도.
   LR top-r selection rule (top-cᵢ² index) 의 경우 `KTA(K_S) ≤ KTA(K_train) ·
   (사이즈 ratio + concentration term)` 의 형태가 expected.
3. ridge regression 의 generalization error `‖f̂_S − f*‖² ≤ ?` 의 KTA-side bound
   derivation. ρ = 10⁻² 의 explicit 처리.

예상 분량: 한 iteration. 결과는 paper 의 §7 Theory 의 *quantitative bound* 한
proposition.

### iter_05 task B: Vision domain 확장 (CIFAR-10 / ImageNet sub-class imbalance)

NLP-only 의 generalization risk — H4 의 r=0.99 가 BERT NTK 의 특수성일 가능성.
CIFAR-10 의 sub-class imbalance (예: 10-class 의 majority 70% / minority 5%) 와
ResNet-18 의 NTK spectrum 으로 H2-H4 의 재검증. 이 task 는 새 학습 실험 launch 가
필요 (data_selection_test 의 vision branch 가 있으면 사용, 없으면 new pipeline).

예상 분량: 1-2 iteration (NTK 계산 + sidecar 산출 + lens 1-2 분석).

### iter_05 task C: Training dynamics — importance weighting 효과의 epoch 의존성

Liu et al. 2025 arXiv 2505.03617 의 "importance weighting effect fades with training"
관찰의 우리 setup 검증. BERT fine-tune 의 1-epoch vs 3-epoch vs 5-epoch 의 valbal
A1 recovery 의 epoch-dependent rate 측정. 만약 fade 가 *cell 별로 다른 속도* 라면
paper 의 §Limitations 의 한 단락 + iter_06 의 *dynamic mechanism* 분기 ground 가
됨.

예상 분량: 1 iteration (기존 NTK 사용, 학습만 epoch 변경).

### iter_05 우선순위 결정

- **Task A (KTA bound 도출)**: paper 의 *theory contribution* 의 핵심, 필수.
- **Task B (vision 확장)**: paper 의 *generality* 의 핵심, 강추.
- **Task C (training dynamics)**: paper 의 *limitation* 의 한 단락 + iter_06
  expansion, 권장.

round 2 결과 후 사용자가 1 개를 선택. round 2 결과가 H4 의 mechanism evidence 약화
면 (regime-cross r < 0.4 등) task A 의 우선순위 하향, task B 의 vision 확장으로
mechanism 의 *cross-domain* robustness 먼저 검증.

## §5. 종합 — round 2 의 dependency graph

```
round 2 task 의존성:
1. lens_mixed_model.py  (15-20 분, single core)
   ├── secondary tier (H1-H4 mixed model β + CI)
   ├── tertiary tier (within-dataset Spearman)
   ├── partial r² 분해 (H2, H3, H4)
   ├── regime-cross (H4 valbal application)
   └── predictor redundancy (max_P_train ↔ kta_gap_r10, etc.)
        │
        ├─ decision: scenario A or B (paper main story arc)
        └─ decision: H4 mechanism vs confounding

2. mnli cls90 INV rerun  (1 day, single GPU)
   └── H4 fit robustness 재검증 (n=14 → 15)

3. paper draft outline 작성  (round 2 + mnli rerun 완료 후)
   └── target venue 결정 (NeurIPS 2027 or ICML 2027)

4. iter_05 task 분기  (task A / B / C 의 사용자 선택)
```

round 2 의 1, 2 가 *parallel* 실행 가능 (mnli rerun 은 GPU, mixed_model 은 CPU).
1 의 결과가 paper draft scenario 결정의 trigger, 2 의 결과가 H4 의 final
interpretation 의 robustness 결정. 두 결과가 *모두* 일관 (mechanism evidence
강화) 이면 paper draft 진입 + iter_05 task A 우선. 결과가 다르면 (H4 의 dataset
confounding warning 이 유지) iter_05 task B (vision 확장) 우선으로 mechanism 의
cross-domain robustness 먼저.

## §6. 참고

- 1st-round 결과 파일 위치 (모두 `state/iteration_04/` 하위):
  - `grand_df.csv`, `grand_meta.csv`, `lens1_table.csv`, `lens1_corr.csv`,
    `lens1_cell_kl.csv`, `lens2_table.csv`, `lens2_corr.csv`, `ntk_schema.md`.
  - figure: `lens1_figs/H{1,2}_scatter.png`, `lens2_figs/H{3,4}_scatter.png`.
- critique 의 본 결과 분석: `state/iteration_04/critique.md` §9.
- iter_03 framework 의 7-quantity / Cond 1-3: `state/iteration_03/report.md` §3.
- 외부 reference 추가:
  - Liu et al. 2025 arXiv 2505.03617 "Understand the Effect of Importance Weighting
    in Deep Learning on Dataset Shift" — training-stage dependence of IW effect.
  - Cortes, Mohri, Rostamizadeh 2012 JMLR — KTA-based generalization bound (iter_05
    task A 의 출발점).
  - Garg, Wu, Smyl, Lipton 2020 NeurIPS arXiv 2003.07554 — weighted ERM excess
    risk bound (iter_05 task A 의 lens 1 흡수 framework).

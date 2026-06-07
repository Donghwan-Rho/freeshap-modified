# Iteration 04 (재정의, v2 — critic 반영) — Plan: collapse mechanism 의 정량 분석

이 plan 의 목표는 추가 학습 실험을 시작하는 것이 아니라, 이미 모인 7-dataset ×
valbal/trainbal × 21+15 의 sidecar 더미와 동일 폴더의 NTK gram cache (`imbalance_ntk/
<ds>/<tag>/bert_seed2026_num<N>_<val_tag>_signFalse.pkl`) 를 *사후 mechanism 분석*
하여, 두 핵심 발견 — (A) valbal 에서 A1 이 FreeShap / LR collapse 를 균일하게 회복,
(B) trainbal 에서 A1 이 balanced acc 로는 압도하지만 절반 case 에서 random 에 패배 —
의 *원인* 을 lens 1-4 로 분리하는 것입니다. 사용자 지시 (`state/directives/
20260603_iter04.md`) 의 (Q1) train ↔ val 의 imbalance 위치가 왜 collapse 양상을
바꾸는지, (Q2) A1 이 정확히 어떤 quantity 를 회복하는지, (Q3) trainbal 에서 A1 이
random 한테 지는 case 의 predictive metric 이 무엇인지를 — 각 lens 가 1 개씩 잡도록
설계했고, iter_03 의 7-quantity / Cond 1-3 framework (`state/iteration_03/report.md`
의 §3, §4) 와 명시적으로 mapping 합니다.

본 v2 plan 은 첫 plan 에 대한 자체 critique (`state/iteration_04/critique.md`) 의
30 개 수정사항 (필수 20 + 권장 8 + 선택 2) 을 반영한 결과입니다. 핵심 변경 — Nyström
extension 식의 √n factor 복원 (§3.2(2)), KL 방향 통일 (§2.2(1)), BBSE confusion
matrix 의 joint vs conditional 표기 정정 (§2.2(3)), H1-H4 의 통계 절차를 setting
단위 Pearson + cell 단위 mixed model + multiple-testing 보정의 3-tier 로 확장 (§3.3),
lens 1 script spec 신설 (§2.5), 2 번째 라운드 critique 가 받을 minimum output set
명시 (§8). 첫 plan 의 5-단계 backbone (데이터 통합 → lens 1 → lens 2 → lens 3 brief
→ lens 4 brief → 종합) 은 유지합니다.

전체는 5 단계로 묶입니다. **§1** 은 데이터 통합 (모든 sidecar → grand long-format
CSV) + NTK pickle schema 의 사전 확인, **§2** 는 lens 1 (label shift / BBSE /
weighted risk) + script spec, **§3** 은 lens 2 (spectral target alignment, KTA,
train-Ỹ vs val-Ỹ alignment gap), **§4** 는 lens 3 (effective dimension d_eff(ρ))
의 brief 보강, **§5** 는 lens 4 (Shapley value variance) 의 brief 진단입니다. **§6**
가 종합 + iter_03 framework 와의 mapping, **§7** 이 next_directions 의 윤곽, **§8**
이 산출물 체크리스트와 2-라운드 critique 의 minimum input set. lens 1-2 는 깊게,
lens 3-4 는 brief.

서두에 두 가지 *제약* 을 못 박습니다. 첫째, **dataset-specific 분석 금지** — 모든
주장은 7-dataset 의 cross-dataset correlation 또는 group-level statistic 위에서만
정당화하고, "AG News 는 cls85 에서…" 식의 single-cell anecdote 는 supporting evidence
로만 허용. 둘째, **추가 실험 launch 금지** — 새 ratio, 새 dataset, 새 NTK 계산 모두
차단. 기존 NTK pickle 의 eigendecomposition 과 기존 sidecar 의 acc 시퀀스만 사용.
분석 스크립트가 GPU 를 쓸 일은 NTK eigen (numpy/SciPy 의 CPU eigh) 뿐이라 GPU
contention 도 없습니다.

## §1. 데이터 통합 — grand long-format dataframe + NTK schema 사전 확인

분석 전반의 토대가 되는 단일 CSV (`state/iteration_04/grand_df.csv`) 와 setting-level
metadata (`state/iteration_04/grand_meta.csv`) 를 만듭니다. sidecar 더미가 두 갈래
setup (valbal 21 settings: train imbalanced + val balanced; trainbal 18 settings:
train balanced + val imbalanced) 으로 나뉘어 있고 각 setting 안에 (method ∈
{FreeShap=inv, LR top-r=lrfshap+eigen, A1 top-r=a1+eigen}) × (r ∈ {1, 5, 10, 15, 20,
25, 30}%) × (sel ∈ 1..100) 의 acc 시퀀스가 들어 있어, 정합 분석을 위해서는 *long
format* — 한 row = (dataset, regime, ratio_tag, method, rank_pct, sel) 의 unique
cell — 으로 풀어두는 것이 필수입니다.

**사전 sanity check — NTK pickle schema 의 인덱싱 (수정사항 7, 18).** lens 2 의
val-side projection 가능 여부 자체가 NTK pickle 에 K_{train,val} block 이 있는지에
의존하므로, build script 의 *첫 단계* 로 임의 pickle 한 개의 schema 를 출력해
인덱싱합니다. 구체적으로 — sst2 의 `pos70 / bert_seed2026_num5000_valbal856_signFalse.pkl`
한 개를 열어 `bundle.keys()` 와 각 value 의 `(type, shape, dtype)` 를 출력하고,
결과를 `state/iteration_04/ntk_schema.md` 에 기록. 가장 critical 한 점검은
`bundle["ntk"].shape == (n + n_val, n + n_val)` 의 full gram 여부입니다 (sst2 의
경우 expected shape `(5856, 5856)`). 만약 train-only `(n, n)` 면 lens 2 의
val-projection 식 (§3.2(2)) 자체를 *제거*; *proxy fallback 사용 금지* — proxy 를
쓰면 Nyström extension 의 수학적 의미가 깨져 H3, H4 의 falsifier 가 무력화됩니다.

FreeShap codebase 의 `compute_ntk` 함수 (상위 repo `../../freeshap/`) 를 직접 확인해
schema 를 plan 단계에서 못 박는 것이 가능하지만, plan 분량 절약을 위해 build script
첫 단계의 sanity check 로 대체. ntk_schema.md 의 산출이 plan 의 §2-§3 의 *수식이
유효한지의 전제조건* 임을 명시.

**ρ (eigen_lambda_) 의 cross-sidecar 일관성 (수정사항 29).** sidecar 의 `eigen_
lambda_` field 가 모든 setting 에서 동일 값 (10⁻²) 인지, 아니면 method/setting 별로
다른지를 build script 에서 출력. 만약 다르면 §3.2(3) 의 mode-wise learnability score
sᵢ 의 ρ 가 cell 별로 달라져 cross-cell 비교의 일관성이 깨지므로, ρ-그룹 별 separate
분석이 필요. 다행히 sidecar 의 표준 default 가 ρ=10⁻² 라 거의 일관할 expect, 하지만
이 가정의 명시적 검증이 *분석 전* 에 필요.

**스크립트 spec — `experiments/build_grand_df.py`.**

입력: 두 root — `data_selection_test/imbalance/data_selection/` (sidecar 의 부모),
`imbalance_ntk/` (NTK pickle 의 부모). 두 root 의 디렉토리 트리를 walk 하면서 sidecar
JSON 의 metadata (`dataset_name`, `class_ratios`, `ratio_tag`, `method`, `seed`,
`num_train_dp`, `val_sample_num`, `setting_name`, `approximate`, `eigen_rank_pct`,
`eigen_lambda_`, `random_seed_used` if present) 와 acc 4-tuple (`top_results_<inv|
eigen>`, `random_results_<inv|eigen>`, `top_results_<inv|eigen>_balanced`,
`random_results_<inv|eigen>_balanced`) 를 추출.

출력 CSV grand_df.csv 의 column schema (한 row = (setting × method × rank_pct × sel)):

| column | 정의 |
|---|---|
| `dataset` | sst2 / mr / mrpc / qqp / rte / mnli / ag_news |
| `regime` | "valbal" / "trainbal" / "single" |
| `train_ratio_tag` | sidecar 의 `ratio_tag` (예: `pos70`, `cls60_20_20`) |
| `val_ratio_tag` | setting_name 에서 파싱 (`valbalN` 또는 `valimbN_pos<XX>` / `valimbN_cls<...>`) |
| `imbalance_level` | "mild" if 최대 majority ≤ 0.7, "extreme" otherwise (수정사항 21 의 stratification key) |
| `num_train, num_val` | sidecar 의 두 size |
| `method` | "FreeShap" / "LR" / "A1" |
| `rank_pct, sel` | sidecar 의 `eigen_rank_pct`, selection count |
| `acc_top_naive, acc_random_naive, acc_top_balanced, acc_random_balanced, acc_at_f0` | sidecar acc 의 normalising (정수/10000) |
| `gap_top_random_naive, gap_top_random_balanced` | `top − random` 의 두 metric |
| `rho_used` | sidecar 의 `eigen_lambda_` (수정사항 29 의 cross-sidecar 확인용) |

setting-level metadata `grand_meta.csv` 의 column:

| column | 정의 |
|---|---|
| `dataset, regime, train_ratio_tag, val_ratio_tag, num_train, num_val` | grand_df 와 동일 join key |
| `ntk_path` | 대응 NTK pickle 의 상대경로 |
| `ntk_shape` | `(n+n_val, n+n_val)` 또는 `(n, n)` — schema 확인 결과 |
| `n, C` | train dim, class 개수 |
| `train_class_counts, val_class_counts` | length-C list (NTK pickle 의 sampled idx 와 dataset label join) |
| `P_train_y, P_val_y` | length-C 정규화 vector |
| `top_eigs_path` | NTK eigendecomposition 캐시 위치 (`eig_cache/<ds>_<tag>_<val_tag>.npz`) |

eigendecomposition 은 lens 2 의 핵심 입력이라 §1 에서 한 번만 계산하고 캐시. n ≤
5000 의 train kernel 이라 numpy.linalg.eigh 로 충분 (~ 60-90 초/ setting, 50 settings
≈ 1 시간). 캐시 schema: `npz` 에 `eigvals` (shape `[n]`, 내림차순), `eigvecs` (shape
`[n, k]`, k=min(n, 500)), `Y_tilde` (shape `[n, C]`, P_train(c) 로 column-wise mean
centering), `Y_val_tilde` (shape `[n_val, C]`, P_val(c) 로 centering), `K_train_val`
(shape `[n, n_val]`, NTK pickle 이 full gram 일 때만; train-only 면 이 key 부재).

검증 (sanity check 셋): (i) `acc_at_f0` 가 한 setting 안에서 method 들 사이에 동일,
(ii) random_results 의 method 간 일치 — *수정사항 30* — `random_seed_used` field 가
sidecar 에 없으면 *strict* (정확 일치) 가 아닌 *moderate* (cross-method Spearman ≥
0.9, 또는 acc 값의 cross-method MAE ≤ 0.01) 기준 적용, (iii) `acc_top_naive[sel=100] ≈
acc_at_f0`. 셋 중 하나라도 깨지면 parsing 버그.

estimated cost: 1 CPU core, 단일 노드 1.5 시간 (NTK eigh 1 시간 + sidecar parsing
30 분). GPU 불필요.

## §2. Lens 1 — Label shift / BBSE / weighted risk

### §2.1 motivation 과 hypothesis

valbal 과 trainbal 의 *유일한 구조적 차이* 는 P_train(y) 와 P_val(y) 의 mismatch
direction 입니다. valbal 에서는 P_train 이 skewed, P_val 이 uniform 이라 train →
val 의 *importance weight* w(y) = P_val(y) / P_train(y) 가 majority class 에서 < 1,
minority 에서 ≫ 1 (예: sst2 pos70 → w(0) = 0.5/0.3 ≈ 1.67, w(1) = 0.5/0.7 ≈ 0.71).
trainbal 에서는 정반대 — P_train(y) uniform, P_val(y) skewed 라 w 의 방향이 역전.
label shift 의 BBSE framework (Lipton-Wang-Smola 2018, ICML, arXiv 1802.03916) 은 이
mismatch 의 *통계적 영향* 을 classifier risk side 에서 정량화하는 표준 도구입니다.
Saerens, Latinne, Decaestecker (2002, Neural Computation) 의 EM-based estimator 는
*framework 의 origin* 으로만 인용 — oracle P_val(y) 를 우리가 갖고 있으므로 EM
iteration 은 직접 사용하지 않고, 실제로 쓰는 closed-form 은 Lipton 2018 eq. (3)
및 후속 (수정사항 23).

가설 H1 — **trainbal 에서 A1 이 random 한테 지는 case 는 P_val(y) skew 가 깊고
small selection budget 의 train sample 이 우연히 majority class 로 쏠려 weighted
balanced risk 가 random selection 보다 더 큰 cell 이다.**

*operational 정의 (수정사항 20)* — trainbal random-loss-rate 를
`P(gap_top_random_balanced[A1, r=10%, sel ∈ {1, 2, 5}] < −0.02)` (즉 A1 이 random
보다 balanced acc 기준 2pp 이상 진 cell 의 비율) 로 정의. threshold −0.02 는 val_size
200-1000 의 binomial std (약 1-3pp) 보다 큰 효과를 잡기 위한 선택. 이 rate 가 H1 의
outcome metric, 그리고 cell 단위 outcome 으로는 `gap_top_random_balanced` 의
연속값을 그대로 사용.

정량 falsifier (수정사항 13 의 effect size + p-value 결합): trainbal regime 의
small-budget cell 에서 (predictor = kl_S_val, outcome = gap_top_random_balanced) 의
*setting 단위 평균* (수정사항 14) Spearman r 이 |r| > 0.3 AND p < 0.0125 (Bonferroni
보정 α/4) 면 H1 confirm. |r| < 0.2 AND p > 0.1 면 reject. 그 사이 grey zone 은 cell
단위 mixed model (수정사항 11) 의 결과를 primary 판정으로 사용.

가설 H2 — **valbal regime 에서 A1 의 FreeShap/LR collapse 회복 강도는 P_train(y)
의 entropy H(P_train) 또는 majority probability max_c P_train(c) 와 monotone.**
즉 imbalance 가 깊을수록 A1 의 marginal benefit 이 커야 함. falsifier: valbal regime
dataset×ratio cell (21 setting) 에서 (x = max_c P_train(c), y = `acc_top_balanced[A1,
r=10%, sel=5] − acc_top_balanced[LR, r=10%, sel=5]`) 의 Pearson r 이 |r| > 0.3
AND p < 0.0125 면 confirm. n=21 의 small-n 에서 r=0.5 의 95% Fisher CI 가 약
[0.08, 0.77] 로 매우 wide 임을 인정 (수정사항 11 (a)), 임계값을 0.5 → 0.3 으로 완화.
*기존 plan 의 0.5 cutoff 는 false-negative 위험* 이라는 critic 지적 반영.

### §2.2 정확한 quantity 정의 (수식 출처 명시)

n 개 train sample, n_val 개 val sample, C class, ŷ : 𝒳 → {1,…,C} 가 임의의
classifier. label shift 의 가정은 P_train(x|y) = P_val(x|y) 이고 marginal P_train(y)
≠ P_val(y) 만 다르다는 것 (BBSE Sec. 2.1).

**(1) KL divergence, 방향 통일 (수정사항 1).**

  kl_val_train = KL(P_val(y) ‖ P_train(y)) = Σ_c P_val(c) · log(P_val(c) / P_train(c))     [main]
  kl_train_val = KL(P_train(y) ‖ P_val(y)) = Σ_c P_train(c) · log(P_train(c) / P_val(c))   [sensitivity]

label shift literature 의 convention — source → target (P_train → P_val) 의 KL 은
coverage error proxy, target → source (P_val → P_train) 의 KL 은 importance-weight
variance proxy. 우리 H1, H2 의 mechanism 은 weighted ERM 의 *variance* 가 핵심이
므로 `kl_val_train` (target → source) 을 H1-H4 의 primary predictor 로 통일.
`kl_train_val` 은 같이 산출하되 sensitivity check 로만 사용. cell 단위 selected-
subset 의 분포 P_S 에 대해서도 *동일 방향* 으로 `kl_val_S = KL(P_val ‖ P_S)` 가
primary, `kl_S_val` 은 secondary. (단 H1 의 직관적 표현 — selected subset 이 val
분포에서 멀어지면 risk 가 커진다 — 는 두 방향 모두에서 monotone 이라 결과가 같은
부호일 expect.)

**(2) BBSE importance weight (Lipton 2018 eq. (3)).**

  w(c) = P_val(c) / P_train(c) = q(y) / p(y)   (p = source = train, q = target = val)

방향이 Lipton 2018 의 표준과 *정확히* 일치. weighted ERM 의 sample weight 가 source
distribution 의 expectation 을 target distribution 의 expectation 으로 변환하는
importance ratio 라 이 방향이 표준. column `w_vector` (list of length C).

**(3) Confusion-matrix BBSE estimator (Lipton 2018 eq. (4)-(5), Theorem 3) — joint
vs conditional 명확화 (수정사항 2).** BBSE 의 confusion matrix 는 *joint* 분포

  Ĉ_{ij} = (1/n_train) · Σ_{k=1}^{n_train} 𝟙[ŷ_k = i, y_k = j]     [Lipton 2018 eq. (4)]

이고 (conditional `P̂(ŷ=i | y=j)` 가 아님), BBSE 가 직접 산출하는 것은 *importance
weight w 가 아니라* target marginal

  q̂_val(y) = Ĉ⁻¹ · μ̂_val(ŷ),   μ̂_val(ŷ)_i = (1/n_val) Σ_l 𝟙[ŷ_l = i]   [eq. (5)]

이며, w 는 후속 elementwise division `ŵ(c) = q̂_val(c) / p̂_train(c)`. consistency
bound (Lipton 2018 Theorem 3): `‖ŵ − w‖₂ = O((1/σ_min(C)) · √(C/n))`. 우리 setup 은
oracle P_val(y) 를 갖고 있으므로 estimator 자체는 *직접 사용 안 함*; 다만 oracle w
와 BBSE w 의 L1 차이 `bbse_err` 를 sanity check column 으로 산출.

추가로 (수정사항 3) — BBSE 의 conditioning 자체가 wins 패턴과 상관되는지 점검:
column `sigma_min_C_hat` (Ĉ 의 smallest singular value), `cond_C_hat = σ_max/σ_min`.
binary (C=2) 에서는 BBSE 가 *수학적으로 trivial 한 게 아니라* oracle P_val 을 우리가
알기 때문에 *실용적으로 redundant* 인 것뿐이고, multi-class (MNLI C=3, AG News C=4)
에서는 conditioning 이 작으면 (모형이 weak 일수록 작음) RLLS (Azizzadenesheli, Liu,
Yao, Anandkumar 2019, arXiv 1903.09734) 의 regularized estimator 가 안정. 우리는
RLLS 를 직접 적용하지 않지만, conditioning 이 cell-level wins 와 상관되면 BBSE
framework 의 multi-class 적용의 *경계조건* 을 발견한 것.

**(4) Weighted classifier risk + finite-sample variance 인정 (수정사항 4).**

  R_w(ŷ) = (1/n_val) Σ_{(x,y) ∈ val} w(y) · 𝟙[ŷ(x) ≠ y]

balanced accuracy 가 R_w 의 special case (`w(c) = 1/(C · P_val(c))`) 이므로,
acc_balanced 와 acc_naive 의 gap 으로 weighted risk gap 의 proxy 가 가능. 다만
*명시 한계*: oracle w-weighted risk 는 *label shift 만 있을 때의 reachable best
predictor 의 risk* 와 동일하지 *않음*. weighted ERM 이 asymptotically (n_val, n_train
→ ∞) optimal 인 것과 finite-n 의 best 가 일치하지 않고, n_val ≤ 1000 의 small budget
에서 Lipton 2018 Theorem 3 의 `O(√(C/n_val))` order variance 가 risk 추정에 직접
들어감. 따라서 weighted-balanced risk 는 *importance-weighted estimator 의 risk
proxy* 일 뿐, lower bound 가 아님을 plan 본문 단계에서 명시.

**(5) Train-sample composition 의 KL (Garg 2020 의 excess risk bound 근거; 수정사항
5).** 각 cell (method, rank, sel) 에서 top-sel 의 train index 가 sidecar 의
`top_results` 와 분리되어 별도 `indices/*.txt` 에 저장됨 (없으면 sidecar 에서 직접
추출). selected subset S 의 class distribution P_S(y) = (1/|S|) Σ_{i ∈ S} 𝟙[y_i =
c]. column

  kl_val_S = KL(P_val(y) ‖ P_S(y))   [primary, target → S 방향]
  kl_S_val = KL(P_S(y) ‖ P_val(y))   [secondary, S → target]

`kl_val_S` 의 *quality proxy* 근거: Garg, Wu, Smyl, Lipton (NeurIPS 2020, "A Unified
View of Label Shift Estimation", arXiv 2003.07554) 의 §3 에서 weighted ERM 의
excess risk 가 `O(‖P_S − P_val‖_TV²)` order 로 bounded. TV 와 KL 의 Pinsker
inequality (`TV(P, Q) ≤ √(KL(P ‖ Q)/2)`) 로 KL 도 same order. 이게 H1 falsifier 의
임계값 0.3 의 order-of-magnitude 근거. [B-Garg20] 으로 bibliography 추가.

**(6) Weighted balanced risk gap.**

  Δ_w = R_w(ŷ_random) − R_w(ŷ_top)

양수면 selection 이 random 보다 weighted risk 측면에서 우월. grand_df 의
`gap_top_random_balanced` 가 acc-side 의 naive balanced gap (= Δ_w 의 unweighted
case = 1/(C · P_val(c)) 가중치), true weighted risk gap 과의 차이가 H1 의 핵심.

### §2.3 검증 procedure

분석 단위 — **cell** = (dataset, regime, train_ratio, val_ratio, method, rank_pct,
sel). 분석 group — (i) regime 별 (valbal vs trainbal), (ii) sel 별 (small ≤ 5%,
mid ∈ [10, 20], large ≥ 50; **추가로 sel=1% 도 별도** — 수정사항 22), (iii) method
별, (iv) imbalance_level 별 (mild vs extreme — 수정사항 21).

H1 의 검증 (수정사항 11, 13, 14):
- *primary (setting 단위 marginal)* — trainbal regime 의 each setting 에 대해 sel
  ∈ {1, 2, 3, 5} 의 4 cell 평균을 한 점으로 (within-setting 종속성 처리). 이렇게
  18 setting × 1 point per setting → (x = kl_val_S, y = gap_top_random_balanced)
  의 Spearman + Pearson + 95% Fisher CI.
- *secondary (cell 단위 mixed model)* — `gap_top_random_balanced ~ kl_val_S +
  max_c P_val(c) + (1 | dataset)` 의 statsmodels MixedLM 의 fixed β (kl_val_S 의
  coefficient) + 95% CI. zero 를 cross 안 하면 더 strong.
- *tertiary (dataset 별 within-dataset Spearman)* — 각 dataset 의 setting 내
  Spearman 의 median + IQR.
- *stratification* — imbalance_level ∈ {mild, extreme} 별 separate primary
  Spearman 도 산출. mild-only 에서도 r 가 같은 부호이면 robustness 의 증거.

falsifier — primary Spearman 의 |r| < 0.2 AND p > 0.1 AND mixed model β 의 CI 가
zero 를 cross 하면 reject. primary |r| > 0.3 AND p < 0.0125 AND mixed model β > 0
이면 confirm. 그 사이는 partial.

H2 의 검증: valbal 의 21 setting 에서 (x = max_c P_train(c), y = `acc_top_balanced[A1,
r=10%, sel=5] − acc_top_balanced[LR, r=10%, sel=5]`). primary/secondary/tertiary 3-tier
동일 절차. n=21 의 Fisher CI 가 wide 함을 인정하고 임계값을 |r| > 0.3 (기존 0.5 →
완화) 으로 잡되 mixed model β 와의 결합 판정. partial Spearman 의 covariate 로
`H(P_train)` 도 같이 둠.

(Q1) 의 mechanism-level 답변 — valbal vs trainbal 의 collapse 양상 차이를 *single
metric* 으로 잡습니다: `delta_weighted_balanced_risk = E[acc_balanced − acc_naive]`
of FreeShap. valbal 에서는 acc_balanced ≈ acc_naive (val 이 balanced), trainbal 에서는
acc_balanced ≪ acc_naive (val 이 imbalanced 라 naive acc 가 majority bias 의 reward).
두 regime 에서 collapse 의 *측정 metric 자체가 다른 것* 이 (Q1) 의 답이고, lens 1
의 BBSE-weighted framework 이 이 차이를 정량화.

### §2.4 자료적 한계

label shift framework 의 가장 강한 가정 — P_train(x|y) = P_val(x|y) — 이 우리
controlled imbalance setup 에서 *정확히* 성립합니다. valbal 의 train 은 GLUE-original
의 stratified subsample, val 은 동일 GLUE-original 의 balanced subsample 이라 P(x|y)
가 *같은 generator* 에서 나옴 — covariate shift 항이 zero. trainbal 도 동일. 우리
setup 의 *행운* — BBSE framework 이 oracle gauge 로 사용 가능.

다만 (i) train sample size 가 작음 (n ≤ 5000), (ii) sel ≤ 5% 의 sub-sample 은 n_S
≤ 250 으로 더 작아 importance weight 의 분산이 큼, (iii) C=2 binary 에서는 BBSE
estimator 가 실용적으로 redundant (oracle P_val 을 갖고 있어서). multi-class (MNLI
C=3, AG News C=4) 에서 BBSE confusion matrix 의 conditioning 이 의미 있음 — §2.2(3)
의 `sigma_min_C_hat` column 으로 모니터링.

해석상 가장 큰 risk 는 *label shift 이상의 mechanism* 이 trainbal random-loss 의
일부 case 에 작동할 가능성. H1 의 partial Spearman 이 약하다면 lens 2 (KTA) 가 잔
차를 잡을 수 있는지가 §3 의 task. lens 1 단독으로 trainbal random-loss 의 *완전한*
predictive metric 이 되리라 expect 하지 않습니다.

### §2.5 스크립트 spec — `experiments/lens1_label_shift.py` (수정사항 17, 27)

입력: §1 의 `grand_df.csv`, `grand_meta.csv`. selected-subset index 가 별도 파일에
있으면 그 path 도. NTK pickle 의 K 자체는 lens 1 에서는 불필요.

산출:

(A) `state/iteration_04/lens1_table.csv` — setting 당 한 row (39 settings + 일부
cell-level extension). column schema:

| column | 정의 |
|---|---|
| `setting_id` | `<dataset>_<regime>_<train_ratio_tag>_<val_ratio_tag>` 의 unique key |
| `dataset, regime, imbalance_level` | (수정사항 21 의 stratification key) |
| `C, n, n_val` | class 수 + sample size |
| `P_train_y, P_val_y` | length-C list |
| `kl_val_train` (primary), `kl_train_val` (sensitivity) | 수정사항 1 |
| `w_vector` | length-C list, w(c) = P_val(c)/P_train(c) |
| `H_P_train, H_P_val, max_P_train, max_P_val` | entropy + majority prob |
| `sigma_min_C_hat, cond_C_hat` | BBSE conditioning (수정사항 3) |
| `bbse_err` | oracle w 와 BBSE w 의 L1 차이 |
| `risk_weighted_proxy` | (acc_naive − acc_balanced) of FreeShap at sel=100, sel=10, sel=5 — proxy of weighted risk gap |
| `kl_val_S_at_sel<S>_method<M>` | cell-level KL of selected subset (sel ∈ {1, 2, 3, 5, 10}, method ∈ {A1, LR, FreeShap}) — wide format 또는 별도 cell-level file |

(B) `state/iteration_04/lens1_corr.csv` — H1, H2 검증 table. row = (hypothesis_id,
predictor, outcome, regime, imbalance_level, n_settings, Pearson r, Pearson r 95%CI,
Spearman r, Spearman p, mixed_model_beta, mixed_model_beta_95CI, decision ∈ {confirm,
partial, reject}). lens2_corr.csv 와 동일 schema 로 통일.

(C) `state/iteration_04/lens1_figs/` — H1 scatter (cell 단위 + setting 평균), H2
scatter (setting 단위), regime 별 separate panel.

estimated cost: 1 CPU core, 20 분 (sidecar selected-index 추출 5 분 + KL 계산 5 분
+ mixed model statsmodels 10 분).

dependency: statsmodels (MixedLM), scipy (pearsonr, spearmanr, Fisher transformation).

## §3. Lens 2 — Spectral / kernel target alignment

### §3.1 motivation 과 hypothesis

iter_03 의 §3 에서 정리한 7-quantity (LC, PR(c²), Spearman(λ, c²), overlap_LR_A1,
train_majority, LR_predict_majority_frac, C) 중 *spectral-side* 의 핵심은 PR(c²)
와 LC(r). PR(c²) 가 작으면 label 정보가 top eigenmode 에 집중, LR top-r 이 그 mode
를 잡아 task 를 풉니다. LC(r) = (Σ_{i ≤ r} c²_i) / (Σ_i c²_i) 가 large 이면 동일
결론. iter_04 lens 2 의 새 contribution 은 (i) train-side cᵢ² 만 다룬 iter_03 분석에
*val-side projection* cᵢ²_val 을 Nyström extension 으로 추가, (ii) KTA 의 mode-wise
decomposition 으로 spread 와 alignment 를 동시에 측정.

가설 H3 — **valbal regime 에서 A1 의 recovery 강도는 train-Ỹ KTA 와 val-Ỹ KTA 의
*alignment gap* 과 monotone.** falsifier — 21 valbal setting 에서 (x = kta_gap_r10,
y = recovery_score) 의 primary Pearson + cell-level mixed model 의 결합 판정. |r| >
0.3 AND p < 0.0125 면 confirm.

가설 H4 — **trainbal regime 에서 LR collapse 는 train-Ỹ KTA 가 *높은데도* val-Ỹ
KTA 가 낮은 cell 에서 발생.** falsifier — trainbal 18 setting 에서 동일 3-tier 절차.

### §3.2 정확한 quantity 정의 (수식 출처 명시)

NTK gram K_train ∈ ℝ^{n×n} 의 eigendecomposition K_train = U Λ U⊤ (U = [u₁, …, u_n],
Λ = diag(λ₁, …, λ_n), 내림차순, ‖uᵢ‖₂ = 1). 이건 §1 의 eig_cache 에 있음.

**(1) Train label projection (iter_03 의 cᵢ²).**

  cᵢ²_train = ‖uᵢ⊤ Ỹ_train‖² = Σ_c (uᵢ⊤ ỹ_{train,c})²

Ỹ_train ∈ ℝ^{n×C} 는 column-wise mean centering (Ỹ_train[:, c] = 1[y_i = c] −
P_train(c)).

**(2) Val label projection via Nyström extension — 식 재유도 (수정사항 6).**

기존 plan 의 식

  cᵢ²_val = Σ_c ((1/λᵢ) · uᵢ⊤ K_{train,val} · ỹ_{val,c})² · (n_val/n)     [WRONG]

은 Nyström extension 의 √n factor 가 누락됐고, sample-mean form 으로 풀리지 않은
matrix-product form 이라 부정확. Williams & Seeger (NeurIPS 2000, "Using the Nyström
Method to Speed Up Kernel Machines") 의 eq. (4) 표기 (후속으로 Drineas & Mahoney
2005, JMLR vol. 6 의 eq. (5) 에서 standard form 정리) — train kernel 의 i-번째
eigenfunction ψ_i 의 out-of-sample value 는

  ψ̂_i(x_new) = (√n / λ_i) · uᵢ⊤ k(x_train, x_new)     [Williams-Seeger 2000 eq. (4)]

이며, train sample 에서 ψᵢ 의 평가값과 uᵢ 의 관계는 `uᵢ ≈ (1/√n) · (ψᵢ(x_1), …,
ψᵢ(x_n))⊤` (normalization). 따라서 val sample x_{val, j} 에서의 평가는

  ψ̂_i(x_{val,j}) = (√n / λᵢ) · uᵢ⊤ K_{train,val}[:, j]

이고, val side cᵢ_val 의 sample-mean form 은

  cᵢ_val, c = (1/n_val) · Σ_{j=1}^{n_val} ψ̂_i(x_{val,j}) · ỹ_{val,c}(j)
            = (√n / (λᵢ · n_val)) · uᵢ⊤ K_{train,val} · ỹ_{val,c}
  cᵢ²_val  = Σ_c cᵢ_val, c²

Ỹ_val ∈ ℝ^{n_val × C} 은 val one-hot 의 column-wise mean centering (P_val 로
centering). NTK pickle 의 `ntk` 가 (n+n_val) × (n+n_val) 의 full gram 이면
K_{train,val} = ntk[:n, n:n+n_val]. §1 의 `ntk_schema.md` 가 train-only 임을
보고하면 lens 2 의 val-projection 은 *제거* (수정사항 7).

numerical caveat — small λᵢ 의 division 에서 noise 큼. λᵢ < 10⁻⁶ 이면 Nyström 항을
0 으로 clip, `nystrom_clip_count` column 으로 보고.

**(3) Per-mode learnability score (A1 score 의 원형, BCP21).**

  sᵢ = (λᵢ / (λᵢ + ρ))² · cᵢ²_train,   ρ = 10⁻²   (sidecar 의 `eigen_lambda_`)

이 score 가 A1 의 mode 선택 기준. Bordelon, Canatar, Pehlevan (2021, Nature
Communications) [B-BCP21] 의 mode-wise generalization closed-form 의 학습 가능
quantity (`L_i = (λ_i / (λ_i + κ))² · α_i²`) 와 함수형이 일치. ρ 의 cross-sidecar
일관성은 §1 의 sanity check 로 확인 (수정사항 29).

**(4) Kernel-target alignment, full kernel (Cristianini et al. 2001 eq. (1); 수정
사항 8, 25).**

  KTA(K, Ỹ) = ⟨K, Ỹ Ỹ⊤⟩_F / (‖K‖_F · ‖Ỹ Ỹ⊤‖_F)   [Cristianini, Shawe-Taylor,
                                                     Elisseeff, Kandola, NIPS 2001
                                                     (proceedings vol. 14, 2002),
                                                     eq. (1)]

column `kta_train_full = KTA(K_train, Ỹ_train)`. centered KTA (Cortes, Mohri,
Rostamizadeh 2012, JMLR) 의 K-centering `K_c = H K H, H = I − (1/n) 11⊤` 는 *추가*
효과가 있음 (constant eigenmode 제거) — Ỹ 가 이미 mean-centered 라 *재중복은 아니
지만* K-centering 자체는 *redundant 가 아닌* 추가 자유도. 우리 stratified subsample
의 near-zero mean K eigenmode 가정 하에 KTA ranking 에 미치는 영향이 작다고 expect.
sanity check 로 K_c 의 KTA 도 함께 산출 → 두 ranking 의 Kendall τ ≥ 0.95 면
uncentered 사용. < 0.95 면 centered KTA 를 primary 로 전환.

KTA_val 는 K_val (NTK pickle 에 K_val 이 있다면, full gram 의 `ntk[n:, n:]`) 또는
K_train 위에 val label 을 project 한 형태로 산출. fallback definition —
*KTA_train_with_val_label* = KTA(K_train, Ỹ_train_via_val_y) — train sample 의
label 을 val side 분포로 centering 한 것. 두 KTA 의 차이가 *prior shift 가 label
alignment 에 미치는 영향* 의 정량.

**(5) Top-r KTA — self-contained derivation (수정사항 9, 26).**

기존 plan 이 "arxiv 2108.08752 chapter 3 eq. (5)" 로 인용했으나 이 arXiv ID 의 정확
한 paper / chapter 가 plan 단계에서 미확인. self-contained derivation 으로 대체:
top-r kernel `K_r = Σ_{i ≤ r} λᵢ uᵢ uᵢ⊤` 의 KTA 는 full KTA 의 분자·분모 모두 top-r
truncate.

  ⟨K_r, Ỹ Ỹ⊤⟩_F = Σ_{i ≤ r} λᵢ · uᵢ⊤ Ỹ Ỹ⊤ uᵢ = Σ_{i ≤ r} λᵢ · cᵢ²
  ‖K_r‖_F² = Σ_{i ≤ r} λᵢ²
  ⇒ KTA(K_r, Ỹ) = (Σ_{i ≤ r} λᵢ · cᵢ²) / (√(Σ_{i ≤ r} λᵢ²) · ‖Ỹ Ỹ⊤‖_F)

분모의 `‖Ỹ Ỹ⊤‖_F` 는 r 에 무관하게 full label 의 norm. 결과: top-r KTA 는 [0,
full_KTA] range (full KTA 만 [0, 1]), dataset-internal r-sweep 에만 사용. cross-
dataset 비교는 *ratio* (KTA(r) / KTA_full) 또는 *gap* (수정사항 9 의 일부).

column `kta_train_r<R>` for R ∈ {1, 5, 10, 15, 20, 25, 30}%.

**(6) Top-r KTA gap (train vs val).**

  gap_KTA(r) = KTA(K_r, Ỹ_train) − KTA(K_r, Ỹ_train_via_val_y)

H3, H4 의 main predictor variable. column `kta_gap_r<R>`.

**(7) Effective alignment ratio (iter_03 LC_LR 의 generalization).**

  LC_train(r) = (Σ_{i ≤ r} cᵢ²_train) / (Σ_i cᵢ²_train)
  LC_val(r)   = (Σ_{i ≤ r} cᵢ²_val) / (Σ_i cᵢ²_val)
  LC_gap(r)   = LC_train(r) − LC_val(r)

iter_03 의 LC_LR 은 LC_train(r) 에 해당. LC_val 이 새 quantity, LC_gap 이 H3 의
*partial verification* (KTA 의 unnormalised version).

**(8) Participation ratio (iter_03 와 동일, 수정사항 10).**

  PR_train = (Σᵢ cᵢ²_train)² / Σᵢ (cᵢ²_train)²
  PR_val   = (Σᵢ cᵢ²_val)² / Σᵢ (cᵢ²_val)²

train side 는 iter_03 의 `PR(c²)` column 그대로. *small-n caveat* — PR 의 분모
`Σ cᵢ⁴` 가 small cᵢ² 의 chi-square noise 에 sensitive, finite-n underestimation
bias 있음. iter_03 의 n=5000 와 iter_04 의 setting-별 n=2300-5000 비교 시 *절댓값*
비교는 보류, *ranking* (Spearman across datasets) 만 유효.

**(9) A1-LR overlap (iter_03 의 overlap_LR_A1).**

  overlap_r = |I_LR(r) ∩ I_A1(r)| / r,   I_LR(r) = top-r by λᵢ,
                                          I_A1(r) = top-r by sᵢ

### §3.3 검증 procedure — 3-tier 통계 절차 (수정사항 11, 12, 13, 19, 21, 22)

분석 unit — setting (한 (dataset, regime, train_ratio, val_ratio) 의 unique
combination, valbal 21 + trainbal 18 = 39 settings).

**통계 절차 (3-tier, lens 1 과 동일 구조)**:
1. *Primary (setting 단위 marginal)* — `outcome ~ predictor` 의 Pearson + Spearman
   + 95% Fisher CI. small-n (n=21 또는 18) 의 Fisher CI 가 wide 함을 본문에서 인정
   — n=21 에서 r=0.5 의 CI 가 [0.08, 0.77], n=18 에서 r=0.3 의 CI 가 [−0.20, 0.66]
   으로 zero 를 cross 가능. 임계값 0.5 → 0.3 으로 완화.
2. *Secondary (cell 단위 linear mixed model)* — `outcome ~ predictor + (1 | dataset)`
   의 statsmodels MixedLM fixed β + 95% CI. n_cell = setting × method × rank_pct ×
   sel ≈ 39 × 3 × 7 × ~10 = 약 8000, dataset 을 random intercept 로 처리해 dataset-
   level confounding 흡수. β 의 CI 가 zero 를 cross 안 하면 더 strong 한 evidence.
3. *Tertiary (dataset 별 within-dataset Spearman 의 분포)* — 7 dataset 각각의 within-
   dataset Spearman 의 median + IQR. dataset 간 heterogeneity 확인.

**Multiple-testing 보정 (수정사항 12)**: H1-H4 만 *primary pre-registered*, sub-group
analysis (regime, sel range, method, imbalance_level 별) 은 *exploratory*. primary
4-test 에 Bonferroni 보정 `α/4 = 0.0125` 적용 (또는 BH 의 step-up procedure, 4 test
중 작은 p 부터 정렬해 i-번째 p < (i/4) · 0.05 만족 검사). exploratory 는 raw p-value
와 effect size 만 보고하고 결론적 주장에 사용 안 함.

**Falsifier 임계값 재정의 (수정사항 13)**:
- *confirm* — primary |r| > 0.3 AND primary p < 0.0125 (Bonferroni) AND mixed model
  β > 0 AND β 의 95% CI 가 zero 를 cross 안 함.
- *reject* — primary |r| < 0.2 AND primary p > 0.1 AND mixed model β 의 95% CI 가
  zero 를 cross 함.
- *partial* — 둘 사이의 grey zone. 이 경우 mixed model β 의 결과를 우선 판정으로
  서술 + 본문에서 "small-n 의 Fisher CI 가 wide 라 setting 단위 marginal 만으로는
  결정 불가, cell 단위 mixed model 의 β 와 그 CI 를 일차 근거로" 명시.

**Stratification (수정사항 21)**: H1-H4 의 primary correlation 을 imbalance_level
∈ {mild, extreme} 별로도 separate 산출. pooled correlation 의 mechanical inflation
(extreme cell 이 KL 큼 + collapse 큼 → r > 0.5 의 trivial 발견) 점검.

**sel sweep 확장 (수정사항 22)**: H3 의 cell sweep 에 sel = 1 추가, sel ∈ {1, 3, 5,
10} 의 4 점 평균. sel-별 separate correlation 도 산출해 non-linear 의존성 (sel=1
의 극단 small-budget 에서만 collapse sharp) 확인.

**H3 의 검증**: valbal 21 setting 에서 (x = kta_gap_r10, y = recovery_score), 3-tier
+ stratification + sel sweep. recovery_score 정의 — `acc_top_balanced[A1, r=10%, sel
= 5] − acc_top_balanced[LR, r=10%, sel=5]`, 추가 robustness 로 sel ∈ {1, 3, 5, 10}
와 r ∈ {5, 10, 15} 의 12 cell average. confirm/reject 는 위 임계값에 따름.

**H4 의 검증**: trainbal 18 setting 에서 (x = kta_train_r10, y = − gap_top_random_
balanced[LR, r=10%, sel=5]). 결합 H4' — (x = kta_gap_r10, y = −gap_top_random_
balanced[LR, r=10%, sel=5]) 도 산출.

iter_03 Cond 2 (PR(c²) cause-side) 와의 명시 mapping — iter_04 의 KTA 분석은 이
표현을 *PR 큰데 KTA(r)/KTA_full 비율 작으면 collapse* 로 재진술하고 (이를 "spectral
*reformulation*" 또는 *complementary measure* 로 표현, 동치 주장 아님; 수정사항 15).
PR 의 *scalar* 진단을 r-별 *curve* (KTA mode-wise) 로 generalize 하는 게 lens 2 의
새 contribution.

falsifier (lens 2 종합) — H3, H4 가 둘 다 reject 면 lens 2 가 mechanism 핵심 변수
아님 → lens 3 (d_eff) 또는 lens 4 (variance) 또는 lens 1 단독 설명력으로 후퇴. 한
개만 reject 면 partial mechanism.

### §3.4 스크립트 spec — `experiments/lens2_kta_spectral.py`

입력: §1 의 `grand_df.csv`, `grand_meta.csv`, `eig_cache/*.npz`. NTK pickle 의
K_{train,val} block 도 필요 (eig_cache 산출 시 함께 저장하거나 이 스크립트에서
pickle 재로딩). `ntk_schema.md` 의 결과로 K_{train,val} 부재면 val-projection
column 들을 NaN 으로 채우고 H3, H4 의 val-side analysis 보류.

산출 — `state/iteration_04/lens2_table.csv` — setting 당 한 row, columns:
`setting_id, dataset, regime, train_ratio_tag, val_ratio_tag, imbalance_level, C, n,
n_val, kta_train_full, kta_train_with_val_label, kta_train_r{1,5,10,15,20,25,30},
kta_train_with_val_label_r{1...30}, kta_gap_r{1...30}, lc_train_r{1...30},
lc_val_r{1...30}, lc_gap_r{1...30}, pr_c2_train, pr_c2_val, d_eff_rho, d_eff_r{1...
30}, d_eff_ratio_r{1...30}, overlap_LR_A1_r{1...30}, recovery_score_A1_vs_LR_balanced
(sel=1,3,5,10 each), LR_loss_vs_random_balanced (sel=1,3,5,10 each, trainbal only),
nystrom_clip_count, kta_centered_kendall_tau`.

추가 산출 — `state/iteration_04/lens2_corr.csv` — H3, H4 검증 table, lens1_corr.csv
와 동일 schema: (hypothesis_id, predictor, outcome, regime, imbalance_level,
n_settings, Pearson r, Pearson r 95%CI, Spearman r, Spearman p, mixed_model_beta,
mixed_model_beta_95CI, decision).

추가 figure — `state/iteration_04/lens2_figs/` — H3, H4 scatter (primary setting
단위 + dataset color), stratified panel (mild vs extreme).

estimated cost: eig_cache 가 있으면 setting 당 ~ 5 초 (50 settings × 5 sec = 5 분,
단일 CPU). mixed model 추가 5 분.

### §3.5 직관적 sanity check 와 한계

KTA 의 absolute value 는 dataset 간 비교에 약함 (서로 다른 X 도메인의 ‖K‖_F scale
다름). *gap* 또는 *ratio* 만 cross-dataset correlation 에 사용 — top-r KTA 의 [0,
full_KTA] range 도 dataset-internal r-sweep 에만 (수정사항 9).

val-side projection 의 Nyström extension 이 mathematically 정확하지만 numerical
으로 small λᵢ 의 division 에서 noise 큼. λᵢ < 10⁻⁶ 이면 Nyström 항을 0 으로 clip
(column `nystrom_clip_count` 로 보고).

lens 1 (label shift) 과 lens 2 (KTA) 가 *상관* 변수일 가능성 — kl_val_train 큰
setting 에서 자연스럽게 kta_gap 도 크다. 두 lens 의 *분리 가능한 contribution* 측정
을 위해 multivariate 선형 회귀로 H3 의 recovery_score 를 (kl_val_train, kta_gap_r10,
pr_c2_train) 세 predictor 로 분해. 각 predictor 의 partial r² 가 lens 별 explained
variance.

PR(c²) 와 KTA(r)/KTA_full 의 redundancy 정량화 (수정사항 15) — 두 measure 의 cross-
dataset Spearman 을 산출, ρ_spearman > 0.8 이면 sufficiently 중복 (Cond 2 의 PR-
based 표현과 KTA-based 표현이 경험적으로 동치), < 0.5 면 두 measure 가 *다른* 정보.

## §4. Lens 3 — Effective dimension d_eff(ρ) — brief

Caponnetto-De Vito (2007, Foundations of Computational Mathematics, "Optimal Rates
for Regularized Least Squares Algorithm") 의 핵심 quantity — d_eff(ρ) = tr(K (K +
ρ I)⁻¹) = Σᵢ λᵢ / (λᵢ + ρ) — 는 KRR 의 *effective number of degrees of freedom*
이고, source condition 하 KRR generalization error 의 sharp 비율을 정합니다.
lens 2 의 sub-lens 인 이유: A1 score 의 첫 factor `(λᵢ/(λᵢ+ρ))²` 가 d_eff 의 mode-
wise contribution `(λᵢ/(λᵢ+ρ))` 의 제곱, 즉 spectral filter. lens 2 의 KTA(r) 가
*label* 측 정렬이라면, d_eff 는 *kernel* 측 effective complexity.

정의 (column 으로 lens2_table.csv 에 append):
- `d_eff_rho` = Σᵢ λᵢ / (λᵢ + ρ), ρ = 10⁻².
- `d_eff_r<R>` = Σ_{i ≤ r·n/100} λᵢ / (λᵢ + ρ), top-r mode 의 d_eff 기여분.
- `d_eff_ratio_r<R>` = d_eff_r<R> / d_eff_rho ∈ [0, 1].

검증 — H5 (mild, exploratory only): valbal regime 에서 d_eff_ratio_r10 가 작은
setting 일수록 A1 recovery_score 가 큼. exploratory 라 multiple-testing 보정 불필요,
raw r 와 p 만 보고. lens 2 와 강하게 collinear 할 가능성 큼 (단조감소 spectrum 가정
하에 d_eff_ratio 와 cumulative spectrum 비율은 거의 일치) — partial r² 로 평가.

해석상 lens 3 의 *core* contribution 은 *수치보다는 framework*: A1 의 score factor
가 우연이 아니라 KRR generalization 의 source condition 측 quantity 와 직결된다는
점을 명시 (BCP21 / Simon23 / Caponnetto-De Vito 의 lineage). 이 lens 는 sidecar
분석 자체보다 paper 의 §Theory 의 motivation 으로 들어갈 자료.

산출은 §3 의 `lens2_table.csv` 에 8 개 column 추가만. 별도 스크립트 없이 lens2
script 의 마지막 함수로. estimated cost: < 1 분.

## §5. Lens 4 — Shapley value variance — brief

TMC Shapley 의 marginal contribution 추정은 random permutation σ ∈ S_n 의 sample
mean 으로 계산. tmc_iter = 500 의 finite-sample variance 가 selection ranking 의
안정성을 결정. A1 score 가 LR score 대비 분산이 작은 mode 를 선호한다면 A1 의
ranking 이 더 안정 — 또는 그 반대.

정의:
- 각 method 의 final per-sample Shapley value φ̂_i ∈ ℝ^n 의 *seed-to-seed variance*
  Var_seed(φ̂_i) — single seed (2026) 만이라 직접 측정 불가. proxy 로 TMC 내
  permutation chunk 단위 partial mean 의 jackknife variance Var_jack(φ̂_i) 가
  가능. shapley pickle (data_selection_test/imbalance/shapley/.../*.pkl) 의 TMC
  trajectory 가 chunk 단위로 남아 있는지 점검 필요.
- trajectory 부재면 lens 4 는 *bound 형* 진술만 — TMC paper (Ghorbani-Zou 2019)
  의 Hoeffding bound: |φ̂_i − φ_i| ≤ ε with prob ≥ 1 − δ for ε = O(√(log n/T)).
  T = 500, n = 5000 → ε ≈ 0.04. small budget (sel = 5%) 의 top-100 ranking 변경에
  영향 가능. A1 vs LR 의 sensitivity 비교는 paper §Limitations 한 단락.

분석 procedure — lens 1-3 의 검증이 *모두 reject* 일 때만 lens 4 정밀 분석으로 진입.
산출 — `state/iteration_04/lens4_bound_note.md` (수식 + 두 paragraph, 별도 스크립트
없음).

## §6. 종합 — iter_03 framework 와의 mapping 및 (Q1)-(Q3) 답안 윤곽

iter_03 의 3-조건 framework 와 iter_04 의 4-lens 의 mapping 을 표로 정리 (수정사항
15, 16 의 mapping table 약화 반영):

| iter_03 quantity / Cond | iter_04 lens | 강화 방향 |
|---|---|---|
| LC_LR(r) (Cond 1 primary) | lens 2 의 LC_train(r) | val side LC_val(r) 와 gap 으로 확장. iter_03 §6 부록의 r-robustness 검증 그대로 |
| PR(c²) (Cond 2 cause-side) | lens 2 의 pr_c2_train + KTA_train(r)/KTA_full ratio | **complementary spectral measure** (동치 아님). PR ↔ KTA(r)/KTA_full 의 cross-dataset Spearman 으로 redundancy 정량 검증 (수정사항 15) |
| train_majority (Cond 2 verification) | lens 1 의 max_c P_train(c), kl_val_train | label shift framework 으로 위치 지정 |
| LR_predict_majority_frac (Cond 2 effect-side) | lens 1 의 BBSE-weighted balanced risk | confusion-matrix / weighted risk 의 *결과* observable |
| Spearman(λ, c²) (Cond 1 secondary) | lens 2 의 overlap_LR_A1, score-correlation | iter_03 와 동일 |
| overlap_LR_A1 | lens 2 의 overlap_LR_A1 | iter_03 와 동일 |
| C (Cond 3 primary) | **iter_04 lens 1-4 가 cover 하지 않음** (수정사항 16) | C ↔ KL 의 매핑은 *부정확* (binary balanced 와 binary imbalanced 모두 가능, multi-class balanced 와 imbalanced 도 모두 가능 — 두 mechanism 은 independent). next_directions item 1 (C-sweep 본 실험) 으로 이관 |

**Cond 2 cause-side ↔ lens 2 KTA 의 관계**: 두 측정 (PR, KTA(r)/KTA_full) 이 *원리적
동치* 라는 주장은 약하지만, *경험적으로* 강하게 상관할 expect (특히 spectrum 의 power-
law decay 영역에서). 동치성은 lens 2 의 검증 procedure (PR-KTA(r)/KTA_full 의 cross-
dataset Spearman 측정) 의 *결과* 로 결정될 정량 hypothesis 이지, plan 단계의 *가정*
아님. PR 큰데 KTA(r)/KTA_full 작은 setting 에서 prior shift (kl_val_train > 0) 가
추가되면 LR selection 이 majority class 의 common-feature mode 로 채워져 collapse —
이게 lens 2 의 narrative.

(Q1) — train ↔ val 의 imbalance 위치가 collapse 양상을 바꾸는 이유: valbal 은
balanced acc 가 acc_naive 와 일치 (val 이 balanced) → FreeShap collapse 의 직접 metric
이 train side mechanism (PR 큰데 KTA 작은) 의 직접 효과. trainbal 은 balanced acc
가 naive acc 와 *크게 다름* (val 이 imbalanced) → FreeShap naive acc 는 majority bias
의 reward 를 받아 절댓값 큼, balanced acc 만 보면 collapse. metric 의 reference 차이
가 가장 큰 분기 원인이고, 그 외에는 lens 2 의 train-val KTA gap 이 분기 강도를 조절.
lens 1 의 BBSE-weighted framework 이 이 metric 차이의 *수식적* 정량화.

(Q2) — A1 이 회복하는 quantity: lens 2 의 sᵢ = (λᵢ/(λᵢ+ρ))² · cᵢ²_train 이 BCP21
learnability 와 함수형 일치하는 mode-wise generalization 기여분. A1 의 top-r subspace
는 *label 변별력이 큰* mode 의 집합이라, 그 위에서의 ridge 가 minority class 의
prediction power 를 보존 → balanced acc 회복. mode-wise 표현으로는 LC_train(r) 에서
LR 이 빠뜨린 mode (small λ 에 흩어진 label 정보) 의 회복.

(Q3) — trainbal random-loss-rate (operational 정의: 수정사항 20 의 `gap_top_random_
balanced[A1, r=10%, sel ∈ {1, 2, 5}] < −0.02` cell 의 비율) 의 predictive metric:
lens 1 의 kl_val_S — 즉 selected subset 의 class distribution 이 val distribution 에서
멀어진 cell. lens 2 의 LR_loss_vs_random 과의 결합으로 *KTA_train(r) 큰데 kta_gap_r
도 큰* cell 이 가장 위험. 검증은 §2.3 의 H1, §3.3 의 H4.

종합 paragraph 는 (Q1)-(Q3) 답안 + lens 1-4 의 partial r² 분해 + iter_03 framework
의 spectral 강화 + 한계 (binary 의 BBSE 실용적 redundant, Nyström 의 small-λ noise,
single-seed, controlled imbalance 의 covariate shift zero 가정, n=21/18 의 small-n
Fisher CI wide) 를 prose 로 한 페이지 길이.

## §7. next_directions 의 윤곽

본 iter 의 critique 후 작성될 `state/iteration_04/next_directions.md` 의 candidate
items:

1. **multi-class C-sweep 의 본 실험 launch** — iter_03 §6 의 (ii) 우선 task. C ∈
   {6, 10, 14, 20} dataset (TREC, Yahoo Answers, DBpedia, 20Newsgroups) 의 NTK
   계산 + Shapley 실험. iter_04 lens 1-4 가 *cover 하지 않은* C ↔ Cond 3 mechanism
   (수정사항 16) 의 직접 검증. iter_04 의 H1-H4 가 multi-class 에서 strong evidence
   를 주는지도 부수 검증.

2. **valbal × trainbal 의 symmetric pair sweep** — 같은 dataset 에서 (train pos70,
   val balanced) 와 (train balanced, val pos70) 의 정확한 mirror pair. iter_04 의
   sidecar 는 일부만 mirror, 전체 21 ratio 에 대한 mirror pair 가 lens 1 label shift
   가설의 *causal* falsifier.

3. **A1 score factor decomposition 실험** — sᵢ = (λᵢ/(λᵢ+ρ))² · cᵢ² 를 두 factor
   ablation. score_spec = cᵢ² only, score_filt = (λᵢ/(λᵢ+ρ))² only, score_full = sᵢ.
   각 score 의 wins 비교. lens 3 의 d_eff 실증.

4. **paper draft 의 §Theory 한 페이지** — BCP21 learnability ↔ A1 score 의 함수형
   일치 증명. critique_priorart.md 의 C3 (필수).

현재 plan 시점의 가장 유력 우선순위는 H1, H3 confirm 시 (1) + (4), reject 시 (2) +
(3). critique 결과로 우선순위 재정렬.

## §8. 제약 재확인, 산출물 체크리스트, 2-라운드 critique 의 minimum input set

**제약 (재확인)**:
- dataset-specific 분석 금지. main claim 은 cross-dataset correlation 위에서.
- 추가 실험 launch 금지. NTK 재계산 금지 (eig_cache 까지만). shapley pickle 의
  trajectory 추출 OK.
- iter_03 framework (Cond 1/2/3, 7-quantity) 와의 explicit mapping 의무. lens 2 ↔
  Cond 2 의 spectral 정량화 관계 §6 에 명시 (단 *complementary*, 동치 아님).

**산출물 체크리스트**:
- [ ] `state/iteration_04/ntk_schema.md` (§1 사전 schema 확인, 가장 우선)
- [ ] `state/iteration_04/grand_df.csv` (§1, lens 1-4 공통 입력)
- [ ] `state/iteration_04/grand_meta.csv` (§1, kernel/spectral metadata)
- [ ] `state/iteration_04/eig_cache/*.npz` (§1, NTK eigendecomposition 캐시)
- [ ] `state/iteration_04/lens1_table.csv` (§2.5 spec, lens 1 quantity table)
- [ ] `state/iteration_04/lens1_corr.csv` (§2.5 spec, H1/H2 correlation summary)
- [ ] `state/iteration_04/lens1_figs/*.png` (H1, H2 scatter)
- [ ] `state/iteration_04/lens2_table.csv` (§3.4 spec, KTA + lens 3 d_eff column 포함)
- [ ] `state/iteration_04/lens2_corr.csv` (§3.4 spec, H3/H4 correlation summary)
- [ ] `state/iteration_04/lens2_figs/*.png` (H3, H4 scatter)
- [ ] `state/iteration_04/lens4_bound_note.md` (brief, 한 페이지)
- [ ] `experiments/build_grand_df.py`
- [ ] `experiments/lens1_label_shift.py`
- [ ] `experiments/lens2_kta_spectral.py`
- [ ] `state/iteration_04/critique.md` (2 번째 라운드 critique, executor 종료 후)
- [ ] `state/iteration_04/next_directions.md`
- [ ] `reports/iter_04.pdf` (full archive)
- [ ] `reports/iter_04_summary.pdf` (lens 1-2 깊게 다룬 reading copy)

**2-라운드 critique 의 minimum input set (수정사항 28)**: executor 완료 시 critic
이 받아야 할 4 + α 파일은

1. `state/iteration_04/grand_df.csv` — cell-level acc 와 모든 outcome metric.
2. `state/iteration_04/lens1_corr.csv` — H1, H2 의 confirm/partial/reject decision +
   3-tier 통계 (primary Pearson, mixed β, dataset-별 분포).
3. `state/iteration_04/lens2_corr.csv` — H3, H4 의 동일 schema.
4. `state/iteration_04/lens1_figs/H{1,2}_scatter.png`,
   `state/iteration_04/lens2_figs/H{3,4}_scatter.png` — 4 개 hypothesis scatter.

추가로 ntk_schema.md (lens 2 val-projection 가능 여부 보고), lens2_table.csv
(KTA-PR redundancy 정량) 도 critic 이 참조하면 더 강한 critique 가능. critic 의
primary outcome 은 H1-H4 의 decision (confirm/partial/reject) 과 그 decision 의
3-tier 통계적 robustness — Bonferroni 보정 후에도 p < 0.0125 인지, mixed model β
의 95% CI 가 zero 를 cross 안 하는지, dataset 별 within-Spearman 의 분포가 한
부호로 일관한지의 3 가지.

**Estimated total cost** — 단일 노드 CPU 만 사용, 6 시간 내외 (§1 build 30 분 +
NTK eigh 1 시간 + lens 1 분석 30 분 + lens 2 분석 30 분 + lens 3-4 5 분 + 리포트
빌드 30 분 + debugging buffer 2 시간 + statsmodels MixedLM 30 분). GPU 불필요.

## 참고

- BBSE: Lipton, Wang, Smola (2018) ICML, arXiv 1802.03916. eq. (3) — `w(y) =
  q(y)/p(y)`; eq. (4) — confusion matrix `Ĉ_{ij} = (1/n) Σ 𝟙[ŷ=i, y=j]` (joint);
  eq. (5) — `q̂ = Ĉ⁻¹ μ̂_q`; Theorem 3 — consistency bound.
  https://arxiv.org/abs/1802.03916
- RLLS: Azizzadenesheli, Liu, Yao, Anandkumar (2019), arXiv 1903.09734 — small-σ_min
  영역의 regularized BBSE.
- Garg, Wu, Smyl, Lipton (2020) NeurIPS, arXiv 2003.07554. weighted ERM excess risk
  bound 의 TV/KL form (Pinsker). [B-Garg20]
- Saerens, Latinne, Decaestecker (2002) Neural Computation. prior shift correction
  의 EM-based estimator — framework origin only, 직접 사용 안 함 (수정사항 23).
- KTA: Cristianini, Shawe-Taylor, Elisseeff, Kandola (2001) NIPS, proceedings vol. 14
  (2002 인쇄). eq. (1) — `A(K_1, K_2) = ⟨K_1, K_2⟩_F / √(⟨K_1, K_1⟩_F · ⟨K_2, K_2⟩_F)`.
  https://papers.nips.cc/paper/1946-on-kernel-target-alignment (수정사항 25 의 venue
  통일).
- Centered KTA: Cortes, Mohri, Rostamizadeh (2012) JMLR. K-centering `K_c = H K H,
  H = I − 11⊤/n` 의 추가 효과.
- Nyström out-of-sample extension: Williams & Seeger (2000) NeurIPS, eq. (4) —
  `ψ̂_i(x_new) = (√n / λ_i) · uᵢ⊤ k(x_train, x_new)`. 후속 표기 Drineas & Mahoney
  (2005) JMLR vol. 6 eq. (5). (수정사항 6 의 식 출처)
- d_eff: Caponnetto, De Vito (2007) Foundations of Computational Mathematics —
  d_eff(ρ) = tr(K(K+ρI)⁻¹).
- A1 score 의 learnability anchor: Bordelon, Canatar, Pehlevan (2021) Nature Comm
  [B-BCP21]; Simon, Dickens, Karkada, DeWeese (2023) TMLR [B-Simon23].
- top-r KTA mode-wise decomposition: self-contained derivation (수정사항 9, 26),
  본문 §3.2(5) 참조. (arxiv 2108.08752 인용 제거 — chapter/eq. 확인 불가능.)
- iter_03 framework: `state/iteration_03/report.md` §3, §4.
- prior-art 점검: `state/iteration_04/critique_priorart.md`.
- FreeShap = inv: Wang, Lin, Qiao, Foo, Low (2024) NeurIPS [B1].
- LR = lrfshap top-r: 저자=사용자 (2026) ICML workshop [B2].

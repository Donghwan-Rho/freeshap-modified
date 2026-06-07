# Iteration 04 plan critique — quantity 정의, falsifier 임계값, script spec 의 사후 점검

본 문서는 `state/iteration_04/plan.md` 의 *내부* 비평입니다. 5-단계 backbone
(데이터 통합 → lens 1 label shift → lens 2 spectral/KTA → lens 3 d_eff brief →
lens 4 Shapley variance brief → 종합) 의 골조는 사용자와 이미 합의된 것이라
유지하고, 각 단계 안의 quantity 정의의 수학적 정확성, 검증 procedure 의 타당성,
H1-H4 falsifier 임계값의 적절성, iter_03 framework mapping 의 정확성, 분석 스크립트
spec 의 충분성/누락을 정량 단위로 짚습니다. 마지막 §10 에 planner 가 받아 plan 을
수정할 수 있는 actionable 수정사항 목록을 둡니다. 분량은 본문 약 4000 단어.

## §1. Lens 1 — label shift quantity 정의의 수학적 정확성

plan §2.2 의 quantity 7 개 (KL, w(c), confusion-matrix BBSE, weighted risk, kl_S_val,
weighted risk gap, plus implicit Δ_w) 를 차례로 점검합니다.

**(a) KL divergence 의 방향.** plan §2.2(1) 는 `KL(P_train(y) ‖ P_val(y)) = Σ_c
P_train(c) · log(P_train(c) / P_val(c))` 를 정의하고 reverse `kl_val_train` 도
같이 둔다고 했습니다. 부호 자체는 맞고 두 방향 모두 두는 것도 좋은 판단입니다.
다만 *어느 방향이 H1, H2 의 predictor 로 사용되는지* 가 plan 에 없습니다. label
shift literature 의 convention 은 source → target (P_train → P_val) 방향의 KL
을 *coverage error proxy* 로 쓰고, target → source (P_val → P_train) 의 KL 을
*importance-weight variance proxy* 로 씁니다. 우리 문제에서는 H1 (trainbal 의
small-budget cell 에서 A1 이 random 한테 짐) 의 mechanism 이 "selected subset 의
class distribution P_S 가 P_val 과 멀어진다" 라서 `kl_S_val = KL(P_S ‖ P_val)` (=
selected → target) 이 자연스러운데, baseline 의 train 측 분포는 *target 인 val*
방향으로 가는 게 일관됩니다. 즉 main predictor 는 `kl_val_train = KL(P_val ‖
P_train)` (target → source) 로 통일하고, `kl_train_val` 은 같이 산출만 하되
sensitivity check 용으로 두는 게 좋습니다. plan 에 이 *방향 통일* 의 명시가
없으니 수정 필요 (§10 의 수정사항 1).

**(b) BBSE importance weight 의 방향 — 맞음.** plan §2.2(2) 의 `w(c) = P_val(c) /
P_train(c)` 는 Lipton-Wang-Smola 2018 (arXiv 1802.03916) 의 eq. (3) — `w(y) = q(y) /
p(y)`, p = source = train, q = target = val — 와 *정확히* 일치합니다. weighted
ERM 의 sample weight 가 source distribution 의 expectation 을 target distribution
의 expectation 으로 변환하는 importance ratio 라는 점에서 이 방향이 표준이고,
plan 의 표기는 정확합니다.

**(c) BBSE estimator — `Ĉ^{−1}` 만으로는 binary 에서 부족.** plan §2.2(3) 는
confusion-matrix BBSE estimator `Ĉ ∈ ℝ^{C×C}, Ĉ_{ij} = P̂_train(ŷ=i, y=j),
w_BBSE = Ĉ^{−1} P̂_val(ŷ)` 를 "참고만" 으로 두고, oracle w(c) 와의 L1 차이
`bbse_err` 만 산출한다고 했습니다. 두 가지 문제가 있습니다.

첫째, BBSE 는 *joint* 분포 `P̂_train(ŷ=i, y=j) = P̂_train(ŷ=i|y=j) · P̂_train(y=j)`
의 추정인데 plan 표기 `Ĉ_{ij} = P̂_train(ŷ=i, y=j)` 는 *joint* 인지 *conditional*
인지 모호합니다. 정확히는 BBSE eq. (3) 의 `Ĉ_{ij} = (1/n_train) · Σ_k 𝟙[ŷ_k = i,
y_k = j]` 이 joint 이고, 이 joint 의 `Ĉ^{−1} μ̂_val(ŷ)` 가 `q(y)` 를 직접 추정
(weight w 가 아니라 marginal q 를). w 는 `q̂(y) / p̂(y)` 의 elementwise division
으로 산출. plan 표기는 estimator 의 *직접 산출* 이 w 인 것처럼 읽혀 혼동 여지가
있습니다. 수정사항 2 — Ĉ 의 정의 (joint, not conditional) 와 BBSE 가 직접
산출하는 것이 q̂_val(y) 이고 w 는 그 후 elementwise division 임을 명시.

둘째, plan §2.4 에서 "C=2 의 binary 에서는 BBSE estimator 자체가 trivial" 이라
무력화시켰는데, 이건 *실용적으로는* 맞지만 *수학적으로는* 부정확합니다. binary
에서 BBSE 가 trivial 한 이유는 oracle P_val(y) 를 우리가 알기 때문이지, BBSE
자체가 비정의이거나 부정확해서가 아닙니다. multi-class (MNLI C=3, AG News C=4)
에서 BBSE 의 confusion matrix conditioning number `σ_min(Ĉ)` 가 *작으면* (모형이
weak 일수록 작아짐) RLLS (Azizzadenesheli, Liu, Yao, Anandkumar 2019,
arXiv 1903.09734) 의 regularized estimator 가 BBSE 보다 안정. plan 의 sanity
check (`bbse_err` L1) 는 oracle vs BBSE 의 *정확도* 만 측정하고 *conditioning*
은 측정하지 않습니다. 수정사항 3 — `bbse_err` 와 함께 `sigma_min_C_hat` (Ĉ 의
smallest singular value) 와 `cond_C_hat` 를 산출, multi-class setting 에서
conditioning 이 wins 패턴과 상관되는지 sanity check.

**(d) Oracle weighted balanced risk 의 "label shift 만 있을 때 reachable best"
가정.** plan §2.2(4) 가 weighted risk `R_w(ŷ) = (1/n_val) Σ_{(x,y)∈val} w(y) ·
𝟙[ŷ(x) ≠ y]` 를 정의하면서, 이 risk 가 acc_balanced 의 special case (`w(c) =
1 / (C · P_val(c))`) 라고 (정확히) 짚었습니다. 그러나 *plan 본문이 암묵적으로
가정* 한 것 — "oracle w-weighted risk 가 label shift 만 있을 때 reachable best
predictor 의 risk 와 일치" — 은 *틀립니다*. Sugiyama, Krauledat, Müller 2007 의
covariate shift importance-weighted ERM 의 *unbiased* 성격이 label shift 의 BBSE
세팅에서도 동일하게 성립한다는 점은 맞지만, *reachable best* 는 ERM 의 ridge
규제 선택 (우리는 ρ = 10⁻² 고정) 과 sample-size 의 finite-sample variance 에
의존합니다. weighted ERM 이 *asymptotically* (n_val, n_train → ∞) optimal 이라는
것과 finite sample 에서의 best 가 동일하지 않다는 점은 BBSE paper 의 Theorem 3
(weight estimation error bound `‖ŵ − w‖₂ = O(1/σ_min(C) · √(C/n))`) 에 의해
n_val ≈ 250 의 small-budget regime 에서 *명시적으로* 큰 variance 를 갖습니다.
plan §2.3 에서는 이 점이 "weighted risk gap" 의 측정 한계로만 짧게 언급되어
있는데 (acc 만 갖고 risk reconstruct 어려움), 사실은 더 큰 limitation 입니다 —
oracle weight w(c) 가 *known* 임에도 weighted-balanced risk 가 best reachable 의
proxy 일 뿐이라는 점이 본문에서 명시되어야 합니다. 수정사항 4 — §2.2(4) 끝에
"BBSE weighted risk 가 reachable best 의 lower bound 가 아니라 *importance-
weighted estimator* 의 risk proxy 일 뿐이고, finite sample (n_val ≤ 1000) 에서
는 `O(√(C/n_val))` order 의 variance 를 가짐" 한 줄 추가.

**(e) `kl_S_val` 의 quality proxy 근거.** plan §2.2(5) 의 `kl_S_val = KL(P_S(y) ‖
P_val(y))` 가 "selected subset 의 quality proxy" 라는 가정은 *직관적으로는* 맞
지만 *literature 근거* 가 plan 에 없습니다. 가장 가까운 근거는 Garg, Wu, Smyl,
Lipton (NeurIPS 2020, "A Unified View of Label Shift Estimation", arXiv 2003.07554)
의 §3 에서, weighted ERM 의 excess risk 가 `‖P_S − P_val‖_TV²` 의 order 로
bounded 라는 점입니다 (TV 와 KL 의 Pinsker inequality 로 KL 도 같은 order).
plan 의 H1 falsifier 임계값 (Spearman ≥ 0.3, |r| < 0.2 이면 reject) 의 *근거*
가 이 bound 의 *order of magnitude* 와 연결되어야 설득력 있습니다. 수정사항 5 —
§2.2(5) 에 "Garg 2020 의 weighted ERM excess risk bound 의 dominant term 이
TV(P_S, P_val) 임 (Pinsker 로 KL 도 same order)" 한 줄 + bibliography 에
[B-Garg20] 추가.

## §2. Lens 2 — spectral / KTA quantity 정의의 수학적 정확성

plan §3.2 의 quantity 9 개 (c²_train, c²_val, sᵢ, kta_train_full, kta_val,
kta_train_r, kta_gap_r, LC_*, PR_*, overlap) 를 점검합니다.

**(a) Nyström extension 으로 val projection — 식이 정확하지 않습니다.**
plan §3.2(2) 의 식
`cᵢ²_val = Σ_c ((1/λᵢ) · uᵢ⊤ K_{train,val} · ỹ_{val,c})² · (n_val/n)`
이 다음 두 점에서 부정확합니다.

첫째, Nyström extension 의 standard 식 (Williams & Seeger 2000, NeurIPS, "Using
the Nyström Method to Speed Up Kernel Machines"; 후속으로 Drineas & Mahoney 2005
JMLR vol. 6 의 eq. (5)) 은 train kernel 의 i-번째 eigenfunction ψ_i 의 out-of-
sample value 가

  ψ̂_i(x_new) = (√n / λ_i) · uᵢ⊤ K(x_train, x_new)

입니다 (여기서 uᵢ 는 train kernel 의 normalized eigenvector — `‖uᵢ‖₂ = 1`, λ_i
는 그 eigenvalue, K 는 일반 kernel function). plan 의 식에는 `√n` factor 가
빠져 있습니다. 그래서 `(1/λᵢ) · uᵢ⊤ K_{train,val}` 만 곱하면 magnitude 가
train side 의 `uᵢ⊤ Ỹ_train` 과 비교 가능한 scale 이 아닙니다. `n_val/n` 의 추가
factor 는 *partial* 보정이지만 (n_val ≠ n 일 때 sum 의 stochastic mean scale
조정), Nyström 의 √n factor 와는 *다른* 보정입니다.

둘째, plan 식은 ψ_i(x_val) 를 각 val sample 별로 평가한 vector 의 *내적* 형태가
아니라 `uᵢ⊤ K_{train,val} · ỹ_{val,c}` 의 matrix product 로 적었습니다. 정확히
풀면 `(uᵢ⊤ K_{train,val})` 가 길이 `n_val` 의 row vector 이고 (이게 `√n · λᵢ`
배 scaling 차이로 ψ_i(x_val) 의 sample 평가값과 일치), 이걸 `ỹ_{val,c}` 와 내적
하면 length-1 scalar — 이것이 *train side cᵢ ≡ uᵢ⊤ ỹ_train* 의 val-version 의
*proxy*. 따라서 정확한 식은

  cᵢ_val,c := (1/λᵢ) · uᵢ⊤ K_{train,val} · ỹ_{val,c} · (1/√n_val)
  cᵢ²_val   := Σ_c cᵢ_val,c²

이고, scaling factor `1/√n_val` (혹은 `√n / (√n_val · λᵢ)` 의 dimension match)
가 정확히 어떻게 들어가는지는 plan 이 의도한 *empirical estimator vs population
projection* 의 정합성에 의존합니다. 수정사항 6 — plan §3.2(2) 의 식을 Williams &
Seeger 2000 의 eq. (4) 표기 (`ψ̂_i(x_new) = (√n / λ_i) · uᵢ⊤ k(x_train, x_new)`,
`k` 는 length-n vector) 로 명시 *재유도* 하고, 그 후 `cᵢ_val,c = (1/n_val) Σ_j
ψ̂_i(x_val,j) · ỹ_{val,c}(j)` 의 sample mean form 으로 풀어 쓰기. NTK pickle 의
schema 와 무관하게 식 자체가 정확해야 lens 2 가 falsifiable.

**(b) NTK pickle 의 K_{train,val} block 의 존재 여부 — schema 가 plan 에 미명시.**
plan §3.2(2) 끝과 §3.4 의 입력 명세가 "NTK pickle 의 `bundle["ntk"].shape` 를
출력해 확인" 의 fallback 만 두고, *실제 schema 가 무엇인지* 사전 확인을 미뤘
습니다. `imbalance_ntk/.../*.pkl` 의 파일명에 `val<N>` 또는 `valbal<N>` 또는
`valimb<N>_...` 이 들어가 있고 두 개 (train + val) sample 의 size 가 함께
encoded 된 점 — sst2 의 `num5000_valbal856` 이면 5000+856 의 full kernel 가능
성이 큼 — 으로 보아 full (n+n_val) × (n+n_val) gram 일 가능성이 *높지만*, 
plan 단계에서 확정 필요. 수정사항 7 — §1 build script 의 첫 sanity check 로
`ntk` shape 검증 + `assert ntk.shape == (n+n_val, n+n_val)` 명시, 만약 train-only
면 lens 2 의 val-projection 은 *제거 not fallback* (proxy 가 부정확해 H3, H4 의
falsifier 가 무력해짐). FreeShap codebase 의 `compute_ntk` 함수 (`references/
Freeshap.pdf` Appendix C 또는 source) 를 직접 확인해 schema 를 미리 못 박는 것이
plan 단계에서 가능합니다.

**(c) KTA 정의 — Cristianini 2001 의 식과 일치.** plan §3.2(4) 의
`KTA_train = ⟨K_train, Ỹ_train Ỹ_train⊤⟩_F / (‖K_train‖_F · ‖Ỹ_train Ỹ_train⊤‖_F)`
는 Cristianini, Shawe-Taylor, Elisseeff, Kandola (NIPS 2001, "On Kernel-Target
Alignment") 의 eq. (1) (그들은 단순히 `A(K_1, K_2) = ⟨K_1, K_2⟩_F / √(⟨K_1, K_1⟩_F
⟨K_2, K_2⟩_F)`) 와 *정확히* 일치합니다. 분모의 normalization 도 맞고 mean-centering
에 대한 plan 의 추가 설명 (Cortes-Mohri 2012 의 centered KTA 가 redundant — Ỹ
가 이미 column mean centering 되어 있으므로) 도 정확합니다. 다만 *centered KTA*
의 표준 정의는 Y 의 centering 이 아니라 *kernel 의* centering — `K_c = H K H, H =
I − (1/n)11⊤` — 이므로, plan 의 진술 ("Ỹ 가 이미 mean-centered 라 redundant") 은
*절반만 맞습니다*. K 의 centering 은 *추가* 효과가 있습니다 (constant eigenmode
의 제거). 다행히 우리 setup 의 train data 는 stratified subsample 이라 mean
shift 가 작고, *uncentered* KTA 와 centered KTA 의 ranking 이 거의 동일할
가능성이 크지만, 이 점이 plan 에 명시되어야 합니다. 수정사항 8 — plan §3.2(4)
끝에 "K-centering 은 Cortes-Mohri 2012 의 eq. (5), 우리 stratified subsample 의
near-zero mean K eigenmode 가정 하에 KTA ranking 에 미치는 영향이 작다고 expect.
sanity check 로 K_c = H K H 의 KTA 도 함께 산출, 두 ranking 의 Kendall τ ≥ 0.95
면 uncentered 사용" 한 줄 추가.

**(d) top-r KTA 의 mode-wise decomposition — 분모 normalization 의 의심.**
plan §3.2(5) 가 `KTA_train(r) = (Σ_{i ≤ r} λᵢ · cᵢ²_train) / (sqrt(Σ_{i ≤ r}
λᵢ²) · ‖ỸỸ⊤‖_F)` 라고 적고 출처를 "arxiv 2108.08752 의 chapter 3 의 식 (5)"
라고 했는데, 이 식 자체의 두 가지 의심점이 있습니다.

첫째, KTA 의 *full kernel* 정의 `⟨K, ỸỸ⊤⟩_F / (‖K‖_F · ‖ỸỸ⊤‖_F)` 와 eigen
expansion `K = Σᵢ λᵢ uᵢ uᵢ⊤` 를 대입하면 `⟨K, ỸỸ⊤⟩_F = Σᵢ λᵢ · uᵢ⊤ ỸỸ⊤ uᵢ
= Σᵢ λᵢ · cᵢ²` 이고 `‖K‖_F² = Σᵢ λᵢ²`. 따라서 *full* KTA = `(Σᵢ λᵢ cᵢ²) /
(√(Σᵢ λᵢ²) · ‖ỸỸ⊤‖_F)`. plan §3.2(5) 의 top-r KTA 는 이 식의 분자만 top-r 로
truncate 하고 분모의 `√(Σ_{i ≤ r} λᵢ²)` 도 top-r 로 truncate 했는데, *분자와
분모의 truncate 가 일관* 하다면 이 식은 *top-r kernel `K_r = Σ_{i ≤ r} λᵢ uᵢ uᵢ⊤`*
의 KTA 가 됩니다. 이게 plan 의 의도면 정확하고, 출처 표기를 "arxiv 2108.08752"
보다는 직접적인 `K_r` 의 KTA 정의 — full KTA 의 top-r 절단판 — 으로 자기 충족적
으로 정의해도 됩니다. 수정사항 9 — plan §3.2(5) 의 출처 "arxiv 2108.08752 의
chapter 3" 가 정확히 어떤 chapter 인지 확인 (이 arXiv ID 가 *어떤* paper 인지
plan 에 명시되어 있지 않음 — Bach 2008 의 kernel methods chapter 일 가능성?). 
정확히 출처가 모호하면 *self-contained* derivation 으로 대체: "top-r KTA 는 K_r
= Σ_{i ≤ r} λᵢ uᵢ uᵢ⊤ 의 KTA 이며 분자·분모 모두 top-r 절단 (식 풀이는 부록)".

둘째, mode-wise decomposition 에서 *분모* 의 `‖ỸỸ⊤‖_F` 가 r 에 무관하게 full
label 의 norm 으로 고정된 점은 r 별 비교에서 *상대 scale* 만 보존하고 *KTA 가
[0, 1] range 에 있다는 성질* 은 깨집니다 (full KTA 만 [0, 1], top-r KTA 는
[0, full_KTA] range). 이는 H3 의 Pearson r 측정에서 monotone 관계만 잡으면
되므로 *큰 문제 아님* 이지만, plan §3.5 의 "absolute KTA 는 dataset 간 비교에
약함, gap 만 사용" 진술과 *일관* 시켜 "top-r KTA 도 dataset-internal r-sweep
에만 사용" 을 명시하는 게 좋습니다. 수정사항 9 의 일부로 함께.

**(e) PR(c²) 의 plan 정의 — iter_03 와 일치 확인.** plan §3.2(8) 의 `PR_train =
(Σᵢ cᵢ²_train)² / Σᵢ (cᵢ²_train)²` 는 iter_03 report §1 의 eq. (2) (`PR(c²) :=
(Σᵢ cᵢ²)² / Σᵢ cᵢ⁴`) 와 *정확히* 일치 (분모 `Σ (cᵢ²)² = Σ cᵢ⁴` 동일). 정의는
OK. 다만 iter_03 에서 산출한 `PR(c²)` 값 (MRPC 274.94, QQP 134.57 등) 이 n=5000
× 3-seed 평균이고, iter_04 의 grand_meta.csv 에서는 n 이 데이터셋·setting 별로
다르므로 (mrpc 의 valbal 은 n=2300, mr 의 valbal 은 n=4500) iter_03 의 값과
*직접 비교* 가능한지 plan 에 명시되어야 합니다. PR 자체는 normalization-free 라
n 에 무관하다고 *기대* 하지만, finite-n 의 noise floor `(C-1)σ²/n` 가 PR 의
denominator 에 들어가 작은 n 에서 PR 이 *overestimated* 됩니다 (작은 cᵢ⁴ 가
noise 로 부풀려져 분모가 커지고 PR 이 작아진다는 의미가 아니라, 정반대 — small
cᵢ² 의 chi-square noise 때문에 cᵢ⁴ 의 분포 tail 이 두꺼워져 PR 이 underestimated).
이 finite-n bias 의 방향을 plan 에서 한 줄 인정하고, iter_03 의 n=5000 값과 비교
시 *순서만* 안정한지 검증 절차 필요. 수정사항 10 — §3.2(8) 끝에 "PR 은 small-n
underestimation bias 있음, iter_03 (n=5000) 와의 비교는 *ranking* 만 유효, 절댓값
비교는 보류" 한 줄 추가.

## §3. H1-H4 falsifier 임계값의 적절성 — multiple testing 과 power

plan 의 H1 (Spearman ≥ 0.3), H2 (Pearson ≥ 0.5), H3 (Pearson ≥ 0.5), H4 (Pearson
≥ 0.3) 의 임계값과 cell/setting 수를 통계적 power 측에서 점검합니다.

**(a) cell 수 와 multiple testing.** plan §3.3 의 H3 는 21 valbal setting 의
cross-dataset correlation 이고 H4 는 18 trainbal setting. *setting 단위* 의 n
이 21 또는 18 으로 작아 Pearson r 의 95% CI 가 매우 wide: n=21 에서 r=0.5 의
CI 는 약 `[0.08, 0.77]`, n=18 에서 r=0.3 의 CI 는 `[−0.20, 0.66]` 으로 zero
를 cross 합니다 (Fisher transformation 기반). 따라서 plan 의 임계값 `r ≥ 0.3
면 confirm` 또는 `r ≥ 0.5 면 confirm` 은 *충분히 strict* 가 아닙니다 — 실제
mechanism 이 있어도 small-n 의 sample variance 만으로 r 이 0.3 부근으로 떨어질
가능성이 큼.

대안 — H3, H4 를 *cell 단위 hierarchical* 로 재정의하는 게 더 안전합니다.
*setting 단위* 의 cross-dataset correlation 외에, *cell 단위* (setting × method ×
rank_pct × sel) 의 paired comparison (e.g., 같은 setting 의 A1 cell vs LR cell 의
차이를 outcome 으로) 으로 가면 n 이 ~ 880 (valbal) 또는 ~ 750 (trainbal) 로
늘어나고, dataset 을 random effect 로 처리하면 hierarchical 적절. cross-dataset
correlation 의 *raw* r 만 보면 confounding (dataset-level fixed effect) 이 흡수
되지 않습니다.

수정사항 11 — H1-H4 의 검증을 (i) *setting 단위 marginal* Pearson/Spearman + 95%
CI (Fisher), (ii) *cell 단위 linear mixed model* — `outcome ~ predictor + (1 |
dataset)` 의 fixed coefficient β 와 그 95% CI — 두 형태로 *둘 다* 보고. linear
mixed model 의 β 가 zero 를 cross 하지 않으면 더 신뢰. lme4 의 R 또는 statsmodels
의 `MixedLM` 으로 구현 (analysis script 에 dependency 추가).

**(b) multiple-testing 보정.** plan 은 H1-H4 의 4 개 hypothesis 를 *동시에*
test 하고 추가로 sub-group 분석 (sel ≤ 5%, sel ∈ [10, 20], sel ≥ 50, regime 별,
method 별) 을 함. *나이브하게* 모든 sub-group 의 p-value 를 reported 하면
multiple testing 문제 — 4 main + 12 sub ≈ 16 test 의 Bonferroni 보정은 `0.05/16
≈ 0.003` 의 strict 임계값. Plan 에 이 보정의 *언급이 전혀 없음*. 수정사항 12 —
hypothesis 검증의 보정 정책 명시 — main 4 (H1-H4) 만 *pre-registered* primary,
나머지 sub-group 은 *exploratory* 로 표시. primary 의 Bonferroni 임계값 `α/4 =
0.0125` 또는 BH (Benjamini-Hochberg) 적용.

**(c) Pearson ≥ 0.5 의 false-negative risk.** plan 의 H2, H3 의 임계값 0.5 가
*overly strict* 이라는 비평이 정당합니다. n=21 의 setting 단위에서 true r =
0.4 의 mechanism 이 있어도 sample r 의 std 가 약 `(1 − 0.4²) / √(21 − 2) ≈ 0.193`
로 크고, observed r 이 0.3 부근으로 떨어질 확률이 25% 이상. *0.5 cutoff* 면
mechanism 이 약하지만 *real* 인 경우 reject 가 false. 더 적절한 정책은 (i)
임계값을 *effect size 만* 으로 보지 말고 p-value (multiple-testing 보정 후) 와
함께 판정, (ii) "reject" 의 정의를 `r < 0.2 AND p > 0.1` 같이 *명확한 null evidence*
로 강화. plan §2.1, §3.1 의 falsifier 정의를 이 두 조건 결합으로 재서술. 수정
사항 13 — falsifier 임계값을 effect size + p-value 의 결합으로 재정의.

**(d) H1 의 Spearman ≥ 0.3 — partial Spearman 의 자유도 손실 고려.** plan §2.3
의 H1 검증이 trainbal small-budget (sel ∈ {1, 2, 3, 5}, method=A1, rank_pct=10%)
의 cell 4 개 × 18 setting = *최대 72 cell* 입니다. covariate `train_majority +
val_majority` 의 partial Spearman 은 자유도가 `n − 3 = 69` 정도로 충분하나, 이
72 cell 안에서 *setting 당 4 cell* 의 within-setting correlation (같은 setting
의 sel = 1, 2, 3, 5 의 cell 은 *독립이 아님*) 이 무시되면 effective sample size
가 ~ 18 (setting 수) 까지 떨어집니다. 수정사항 14 — H1 의 partial Spearman 도
*setting 단위 평균* 으로 정의 (each setting 내에서 sel ∈ {1, 2, 3, 5} 평균을
한 점으로) 또는 mixed model 로 within-setting 종속성 처리.

## §4. iter_03 framework mapping 의 정확성

plan §6 의 mapping table 7 개 row 를 점검합니다.

**(a) Cond 1 ↔ LC_train(r).** "iter_03 의 LC_LR(r) ≡ iter_04 의 LC_train(r) = (Σ_
{i≤r} cᵢ²_train) / (Σ_i cᵢ²_train)" 의 동치 주장은 정확합니다 (iter_03 §1 의 LC
LR 정의가 정확히 이 형태). r=10% 외 다른 r 에서의 validity 도 iter_03 §6 부록
("LC_LR(r) r-robustness, n=5000, r=5~30% 에 ranking 유지") 으로 이미 검증됨. OK.

**(b) Cond 2 cause-side ↔ KTA train.** plan §6 의 "Cond 2 의 spectral 정량화는
*train kernel 의 top-r 이 train-label spread (PR_train large) 와 weak alignment
(KTA_train(r)/KTA_full small) 의 결합인 setting 에서 collapse*" 는 *heuristic
equivalence* 입니다 — KTA 와 PR 두 quantity 가 "label 정보의 top-r 흡수" 의
*dual measure* 라는 직관은 정당하지만, 두 측정이 *수학적으로 동치* 라는 증명은
없습니다. 사실 두 측정은 dimension 이 다릅니다: PR(c²) 는 cᵢ² 의 분포 spread
의 inverse 차원의 measure (Σ cᵢ² / max cᵢ² 의 일종), KTA(r)/KTA_full 은 r-mode
의 label-explanatory ratio. *경험적으로* 두 measure 가 강하게 correlate 할 수
있지만 (특히 spectrum 이 power-law decay 일 때), *원리적 동치* 의 주장은 약합
니다. 수정사항 15 — plan §6 의 mapping table 의 PR(c²) row 표현을 "spectral
*reformulation*" 또는 "*complementary* spectral measure" 로 완화, 동치 주장은
삭제. 본문에서도 "PR(c²) 와 KTA(r)/KTA_full 의 Spearman correlation 을 cross-
dataset 측정해 두 measure 의 redundancy 정량화" 라는 검증 procedure 를 추가하면
더 좋음.

**(c) C ↔ Cond 3 ↔ lens 1 의 KL(P_train ‖ P_val).** plan §6 의 마지막 row 가
"Cond 3 (multi-class √(C-1) SNR) ↔ lens 1 의 KL + √(C-1) SNR" 으로 mapping 했
는데, 이 매핑이 *부정확* 합니다. iter_03 의 Cond 3 mechanism 은 label 의 *number
of classes* C 와 noise reduction 의 관계인데, lens 1 의 label shift framework
은 *P_train(y) ≠ P_val(y)* 의 measure 입니다. 두 mechanism 은 *independent*: 
binary balanced (C=2, KL=0) 와 binary imbalanced (C=2, KL>0) 가 모두 가능하고
multi-class balanced (C=3, KL=0) 와 multi-class imbalanced (C=3, KL>0) 도 모두
가능. plan §6 의 표현은 두 mechanism 을 *혼동* 시킵니다. 수정사항 16 — mapping
table 의 C row 를 *삭제* 하거나 "Cond 3 (multi-class noise reduction) 은 lens
1-4 의 *어떤 lens 도 직접 cover 하지 않으며*, iter_04 의 분석 범위 밖. C-sweep
실험은 next_directions 의 item 1 로 이관" 으로 명시.

## §5. 분석 스크립트 spec 의 충분성

plan §1, §3.4 의 두 script (build_grand_df.py, lens2_kta_spectral.py) 와 implicit
한 lens1_label_shift.py 의 spec 을 점검합니다.

**(a) 3 개 script 가 모든 quantity 를 cover 하는가.** plan 의 §8 산출물 체크
리스트에서 lens 1 의 `lens1_table.csv, lens1_corr.csv` 가 명시되어 있는데, *생산
스크립트* 인 `lens1_label_shift.py` 의 명세가 §2 본문에 *암묵적* 으로만 존재하고
별도 §sub-section 으로 정리되지 않았습니다. lens 2 의 §3.4 만큼 명세 정리가 필요.
수정사항 17 — plan §2 에 §2.5 "스크립트 spec — `experiments/lens1_label_shift.py`"
sub-section 추가 (입력: grand_df.csv, grand_meta.csv; 출력: lens1_table.csv,
lens1_corr.csv; column schema; estimated cost).

**(b) NTK kernel pickle 의 위치와 format.** plan §1 의 grand_meta.csv 의
`ntk_path` column 은 *상대경로* 만 명시하고, pickle 의 *내부 schema* (`ntk`,
`per_class_counts`, `sampled_val_idx`, `kernel_eigvals` 등의 key 가 정확히 무엇
인지) 는 *추정* 으로 적혀 있습니다. plan §1 끝의 "캐시 schema: `npz` 에 `eigvals`,
`eigvecs`, `Y_tilde`, `Y_val_tilde`" 가 *우리가 만들 cache* 의 schema 이고,
*원본 pickle* 의 schema 는 모름. 수정사항 18 — plan §1 의 build script 첫 단계
로 "임의 pickle 한 개를 열어 `bundle.keys()` 출력 + 각 value 의 `type, shape,
dtype` 기록, 결과를 `state/iteration_04/ntk_schema.md` 에 인덱싱. 그 schema 가
plan 의 모든 quantity 산출에 충분한지 (특히 K_{train,val} block) 확인" 단계를
명시. 이게 lens 2 의 H3, H4 falsifiability 의 *전제조건* 이라 가장 우선되어야
하는 sanity check.

**(c) cell 단위 cross-dataset correlation 의 통계 절차.** plan 은 (§3.3) cell
단위 correlation 을 setting 수 21 또는 18 의 cross-dataset Pearson 으로 측정한
다고만 했고, *어떤 dataset 을 pool 하는지* (모든 7-dataset?), *dataset random
effect* 처리 정책 (fixed effect 로 빼는가, mixed model 의 random intercept 인가)
의 명세가 없습니다. 수정사항 19 — plan §3.3 끝에 "통계 절차: (i) primary —
`outcome ~ predictor` 의 cross-dataset Pearson + 95% Fisher CI, (ii) secondary —
`outcome ~ predictor + (1 | dataset)` 의 lme4-style mixed model 의 fixed β + 95%
CI, (iii) tertiary — 각 dataset 별 within-dataset Spearman 의 분포 (median, IQR)
로 dataset-heterogeneity 확인" 의 3-tier 명시.

## §6. 누락된 분석

**(a) trainbal random-loss 의 operational 정의.** plan 의 핵심 motivation 인
"trainbal 의 절반 case 에서 A1 이 random 한테 진다" 가 quantitative threshold 의
명시 없이 plan 전반에 사용됩니다. 정확한 정의 — `gap_top_random_balanced[A1,
r=10%, sel=5] < 0` (random 보다 A1 이 acc 낮음) 인 cell 의 비율 — 가 plan 에 명
시되지 않았고, "절반" 이라는 표현이 사용자 directive 의 *대략적 관찰* 이라 plan
에서 *정량적 outcome* 으로 변환 필요. 수정사항 20 — plan §6 의 (Q3) 답안 시작
부분에 "trainbal random-loss 의 operational 정의: `gap_top_random_balanced[A1,
r=10%, sel ∈ {1, 2, 5}] < −0.02` (−2pp 이상 random 에 짐) 인 cell 의 비율을
random-loss-rate 로 정의, 이 rate 가 H1 의 outcome metric" 한 줄 추가. threshold
−0.02 는 acc 측정 noise (val_size 200-1000 의 binomial std 약 1-3pp) 보다 큰 영
역을 잡기 위한 선택.

**(b) imbalance level 별 separately correlation.** plan 의 cross-dataset
correlation 은 모든 imbalance level (mild = 70/30 or cls55/15/15/15, extreme
= 90/10 or cls85/05/05/05) 을 *pooled* 로 처리합니다. 그러나 lens 1 의 KL 측정
이 mild 와 extreme 에서 *order-of-magnitude* 차이 (KL(0.7, 0.5) ≈ 0.09, KL(0.9,
0.5) ≈ 0.51) 가 나서, pooled correlation 이 *almost mechanical* (extreme cell 이
KL 큰 + collapse 큰 → r > 0.5 의 trivial 발견) 일 위험. 수정사항 21 — H1-H4 의
검증에서 imbalance level *별로* separate correlation 도 산출 — `level ∈ {mild,
extreme}` 의 stratified analysis. mild-only 에서도 H3, H4 의 correlation 이
significant 면 mechanism 의 robustness 더 강한 evidence.

**(c) sel% 별 non-linear effect (sel = 1% 만 collapse).** iter_03 와 iter_04 의
사이드카 관찰에서 *sel = 1% 의 극단 small budget* 만 LR collapse 가 가장 sharp
한 영역으로 알려져 있습니다 (plan §3.3 의 H3 검증 cell `sel ∈ {3, 5, 10}` 이 이
영역을 cover 안 함). 수정사항 22 — H3 의 cell sweep 에 `sel = 1` 도 포함 (즉
`sel ∈ {1, 3, 5, 10}` 의 4 점 평균), 그리고 *sel-별 separate* correlation 도
산출해 non-linear 의존성 확인.

## §7. 외부 자료 활용의 명시도

plan 의 §7 (참고) 와 본문 인용 의 정확도를 점검합니다.

**(a) Saerens 2002 의 인용.** plan §2.1, §2.2 에서 "Saerens-Latinne-Decaestecker
2002 의 EM-style estimator" 라고 brief 인용했는데, 정확히 *어떤 식* 이 우리 lens
1 의 어디서 사용되는지 명시 부족. Saerens 2002 의 *core contribution* 은 EM
algorithm 으로 P_val(y) 를 추정하는 *iterative procedure* (그들 paper 의 eq. (5)-
(8)). plan 의 lens 1 은 oracle P_val(y) 를 갖고 있어 *EM 이 필요 없는* setup —
즉 Saerens 2002 의 *iterative estimator* 는 *직접 사용 안 함*, 단지 *label shift
framework 의 origin* 으로만 인용. 수정사항 23 — plan §2.1 의 Saerens 2002 인용을
"prior shift correction 의 EM-based estimator (직접 사용 안 함, framework origin)"
로 한정. BBSE 의 *closed-form* estimator (Lipton 2018 eq. (3)) 가 우리 setup 에
서 실제로 쓰이는 식.

**(b) Lipton 2018 의 어떤 식.** plan §2.2(2) 가 `w(c) = q(y)/p(y)` 를 "BBSE Eq.
(3)" 으로 인용했는데 *정확한 BBSE eq.* 가 무엇인지 본문에서 직접 명시되지 않았
습니다 (arXiv 1802.03916 의 eq. (3) 가 importance weight 정의, eq. (4)-(6) 가
estimator). 수정사항 24 — plan §2.2 의 각 인용에 eq. 번호 명시. 구체적으로 (i)
w(c) 의 정의 → Lipton 2018 eq. (3), (ii) BBSE estimator → eq. (4)-(5) (또는
Theorem 3 의 식), (iii) consistency bound → Theorem 3.

**(c) KTA 의 Cristianini 2001 식 (1).** plan §3.2(4) 의 KTA 정의가 *어느 식*
에서 왔는지 plan §7 의 reference 에는 "NIPS 2002" 로 잘못 표기되어 있습니다
(실제는 NIPS 2001 — 12월 발표, paper 인쇄는 다음해 proceedings volume 14). 수정
사항 25 — plan §7 의 KTA reference 를 "NIPS 2001 (proceedings volume 14, 2002)"
또는 "Cristianini et al. 2001, NIPS" 로 통일. paper 의 어떤 eq. — KTA 의 정의
eq. (1) — 인지 본문 §3.2(4) 에 명시.

**(d) per-eigenvector KTA decomposition 의 출처 — arxiv 2108.08752.** plan §3.2(5)
가 "arxiv 2108.08752 의 chapter 3 의 식 (5)" 라고 인용했는데, 이 arXiv ID 가
어떤 paper / book chapter 인지 plan 의 어디에도 명시되지 않습니다. *제가 검색
해본 결과* — arXiv 2108.08752 는 Canatar, Bordelon, Pehlevan 2021 "Spectral bias
and task-model alignment explain generalization in kernel regression and infinitely
wide neural networks" 의 longer version 일 가능성이 있지만, *plan 의 chapter 3
식 (5)* 가 정확히 어디인지 plan 단계에서 확인되어야 합니다. 수정사항 26 — plan
§3.2(5) 의 출처를 직접 확인 (혹은 *self-contained derivation* 으로 대체, §2(d)
의 수정사항 9 와 동일).

## §8. Cycle progression — 2 번째 라운드 critique 를 위한 output schema

plan 은 §8 의 산출물 체크리스트에 `lens1_table.csv, lens2_table.csv` 등의 파일
명만 적었고, *2 번째 라운드 critique* (executor 결과 분석) 가 어떤 column 을
expect 할지 *output schema* 를 명시하지 않았습니다. 특히 lens2_table 의 column
schema 는 §3.4 에 약 30 개 column 으로 풀어 적혀 있어 OK이지만, *lens1_table* 의
schema 는 plan 본문에 *전혀 없음*. 그리고 *2 번째 라운드 critique* 가 받아 분석
할 *correlation table* (lens1_corr.csv, lens2_corr.csv) 의 schema 도 lens2 만
spec 됨 (`regime, hypothesis_id, predictor, outcome, Pearson r, Spearman r,
p-value, n_settings`), lens1 은 미명시.

수정사항 27 — plan §2.5 (수정사항 17 로 추가될 sub-section) 에 lens1_table.csv
의 column schema 명시. column 은 적어도 `setting_id (dataset + regime + train_
ratio + val_ratio), C, n, n_val, P_train_y, P_val_y, kl_train_val, kl_val_train,
w_vector, sigma_min_C_hat, cond_C_hat, bbse_err, kl_S_val (cell-level, 별도
파일?), risk_weighted_proxy`. lens1_corr.csv 도 lens2_corr.csv 와 같은 schema 로
통일.

수정사항 28 — *2 번째 라운드 critique 의 input* 으로 들어갈 *최소 output set* 을
plan §8 끝에 명시: "executor 완료 시 critic 이 받아야 할 4 파일 — grand_df.csv,
lens1_corr.csv, lens2_corr.csv, plus 4 개 figure (H1, H2, H3, H4 의 scatter)".
이 spec 이 없으면 2 번째 라운드 critic 이 어떤 *primary outcome* 을 보고 hypothesis
판정해야 하는지 모름.

## §9. 기타 minor 점검

**(a) ρ = 10⁻² 의 출처.** plan §3.2(3) 가 ρ = 10⁻² 를 `eigen_lambda_` (sidecar 의)
로 가정했는데, 이 값이 *모든 setting* 에서 동일한지, *method 별* (LR vs A1) 로
다른지 확인 필요. 만약 sidecar 별로 다르면 mode-wise learnability score `sᵢ =
(λᵢ/(λᵢ+ρ))² · cᵢ²` 의 ρ 가 *cell 별로* 다를 수 있어 cross-cell 비교의 일관성
훼손. 수정사항 29 — §1 build script 에서 `eigen_lambda_` 의 cross-sidecar 일관
성 확인, 만약 다르면 그룹별 separate 분석.

**(b) random_results 의 sanity check 의 정확성.** plan §1 의 (ii) sanity check
("random_results 가 method 간 거의 일치") 가 *seed mismatch 차이만 허용* 인데,
실제 random selection 의 *index* 가 method 별로 다를 수도 있습니다 (LRFShap 의
random vs A1 의 random 이 서로 다른 RNG seed 일 가능성). 수정사항 30 — sidecar 의
`random_seed_used` field 가 있는지 확인, 없으면 random_results 의 method-간
일치를 *strict* 가 아니라 *moderate* (Spearman ≥ 0.9 같은 약한 기준) 로 완화.

**(c) acc_at_f0 의 method-간 일치.** plan §1 의 (i) sanity check ("acc_at_f0
method 간 동일") 는 정확하지만, *실제 sidecar* 에서 method 별로 acc_at_f0 가
저장될지 (모든 train 사용 시는 method 무관해야 함) 코드 단계에서 검증 필요.

## §10. Actionable 수정사항 — planner 가 받아 plan 을 수정할 항목

이하 30 개 항목은 §1-§9 의 비평을 *plan 수정 가능 단위* 로 압축한 것입니다. 우선
순위 표기: **[필수]** = 다음 plan revision 에 반영 의무, **[권장]** = 가능하면
반영, **[선택]** = 시간 여유 시.

| # | 수정사항 | 우선 |
|---|---|---|
| 1 | plan §2.2(1): KL 방향 통일 — main predictor `kl_val_train = KL(P_val ‖ P_train)`, `kl_train_val` 은 sensitivity check 용. 두 방향 산출은 유지하되 H1-H4 의 primary predictor 가 어느 방향인지 명시. | 필수 |
| 2 | plan §2.2(3): BBSE `Ĉ_{ij} = (1/n) Σ 𝟙[ŷ=i, y=j]` 가 joint (not conditional) 임을 명시, BBSE estimator 의 직접 산출이 `q̂_val(y)` 이고 `w` 는 후속 elementwise division 임을 명시. Lipton 2018 eq. (3)-(5) 인용. | 필수 |
| 3 | plan §2.2(3): `bbse_err` 외에 `sigma_min_C_hat`, `cond_C_hat` 도 column 추가. multi-class setting 에서 conditioning 의 wins 패턴 상관 sanity check. | 권장 |
| 4 | plan §2.2(4): weighted-balanced risk 가 reachable best 의 *proxy* 일 뿐, finite sample `O(√(C/n_val))` order 의 variance 가짐을 한 줄 명시. | 필수 |
| 5 | plan §2.2(5): `kl_S_val` 의 quality proxy 근거로 Garg 2020 (arXiv 2003.07554) 의 weighted ERM excess risk bound (TV/KL of P_S, P_val) 인용. bibliography 에 [B-Garg20] 추가. | 필수 |
| 6 | plan §3.2(2): Nyström extension 식을 Williams & Seeger 2000 eq. (4) 표기 (`ψ̂_i(x_new) = (√n / λ_i) · uᵢ⊤ k(x_train, x_new)`) 로 재유도. plan 의 현재 식은 `√n` factor 누락. | 필수 |
| 7 | plan §1, §3.2(2): NTK pickle schema 의 *사전* 확인 — `bundle.keys()`, `bundle["ntk"].shape` 를 build script 의 첫 단계로 명시, 결과를 `state/iteration_04/ntk_schema.md` 에 인덱싱. K_{train,val} block 부재 시 lens 2 val-projection *제거* (fallback proxy 사용 금지 — H3, H4 falsifier 무력화 위험). | 필수 |
| 8 | plan §3.2(4): K-centering 의 추가 효과 (constant eigenmode 제거) 명시. uncentered vs centered K 의 KTA ranking 의 Kendall τ ≥ 0.95 sanity check. | 권장 |
| 9 | plan §3.2(5): top-r KTA 식의 출처 명시 (arxiv 2108.08752 가 어떤 paper / chapter 인지) 또는 *self-contained derivation* (K_r = Σ_{i ≤ r} λᵢ uᵢ uᵢ⊤ 의 KTA 의 직접 풀이) 으로 대체. top-r KTA 가 [0, full_KTA] range 이고 dataset-internal r-sweep 에만 사용함을 명시. | 필수 |
| 10 | plan §3.2(8): PR 의 small-n underestimation bias 명시, iter_03 의 n=5000 값과 iter_04 의 setting-별 n 값 비교는 *ranking* 만 유효. | 권장 |
| 11 | plan §3.3, §2.3: H1-H4 검증을 (i) setting 단위 marginal Pearson + 95% Fisher CI, (ii) cell 단위 linear mixed model (`outcome ~ predictor + (1 | dataset)`) 의 fixed β + 95% CI 두 형태로 *둘 다* 보고. lme4 또는 statsmodels MixedLM dependency 추가. | 필수 |
| 12 | plan §3.3, §2.3: multiple-testing 보정 정책 명시 — H1-H4 만 primary (Bonferroni `α/4 = 0.0125` 또는 BH 적용), sub-group (regime, sel range, method 별) 은 exploratory. | 필수 |
| 13 | plan §2.1, §3.1: falsifier 임계값을 effect size + p-value 결합으로 재정의 — "reject" = `|r| < 0.2 AND p > 0.1`, "confirm" = `|r| > threshold AND p < α_corrected`. 0.2-threshold 의 grey zone 은 mixed model 결과 우선. | 필수 |
| 14 | plan §2.3: H1 의 partial Spearman 을 *setting 단위 평균* (each setting 의 sel ∈ {1, 2, 3, 5} 평균을 한 점으로) 또는 mixed model 로 within-setting 종속성 처리. | 필수 |
| 15 | plan §6 mapping table: PR(c²) row 의 "spectral 정량화 동치" 주장을 "spectral *reformulation* / *complementary* spectral measure" 로 완화. PR(c²) 와 KTA(r)/KTA_full 의 cross-dataset Spearman 으로 redundancy 측정. | 필수 |
| 16 | plan §6 mapping table: C row (Cond 3 ↔ lens 1) 삭제 또는 "iter_04 lens 1-4 가 cover 하지 않으며, next_directions item 1 (C-sweep 실험) 으로 이관" 으로 재서술. C ↔ KL 의 혼동 제거. | 필수 |
| 17 | plan §2 에 §2.5 "스크립트 spec — `experiments/lens1_label_shift.py`" sub-section 추가. lens 2 의 §3.4 만큼 입력/출력/column schema/estimated cost 명시. | 필수 |
| 18 | plan §1: build script 첫 sanity check 로 임의 pickle 한 개의 `bundle.keys()` + 각 value 의 `type, shape, dtype` 출력, 결과 `state/iteration_04/ntk_schema.md` 인덱싱. (수정사항 7 과 결합). | 필수 |
| 19 | plan §3.3: 통계 절차 3-tier 명시 — (i) Pearson + Fisher CI, (ii) lme4-style mixed model fixed β + CI, (iii) 각 dataset 별 within-dataset Spearman 분포. | 필수 |
| 20 | plan §6 (Q3) 답안 시작: trainbal random-loss 의 operational 정의 — `gap_top_random_balanced[A1, r=10%, sel ∈ {1, 2, 5}] < −0.02` 인 cell 의 비율을 random-loss-rate 로. | 필수 |
| 21 | plan §3.3, §2.3: H1-H4 검증에 imbalance level *별 stratified* (mild=70/30 or cls55/15, extreme=90/10 or cls85/05) correlation 도 산출. pooled correlation 의 mechanical inflation 점검. | 필수 |
| 22 | plan §3.3: H3 cell sweep 에 `sel = 1` 추가 (`sel ∈ {1, 3, 5, 10}` 의 4 점), sel-별 separate correlation 도 산출해 non-linear 의존성 점검. | 권장 |
| 23 | plan §2.1: Saerens 2002 인용을 "prior shift correction 의 EM-based estimator (framework origin only, 직접 사용 안 함)" 으로 한정. BBSE 의 closed-form 이 실제 사용 식임을 명시. | 권장 |
| 24 | plan §2.2: 각 인용에 eq. 번호 명시 — w(c) 정의 → Lipton 2018 eq. (3), BBSE estimator → eq. (4)-(5), consistency bound → Theorem 3. | 필수 |
| 25 | plan §7 KTA reference: "NIPS 2002" → "Cristianini, Shawe-Taylor, Elisseeff, Kandola (2001) NIPS, proceedings vol. 14". paper 의 KTA 정의 eq. (1) 본문 §3.2(4) 에 명시. | 권장 |
| 26 | plan §3.2(5) 의 arxiv 2108.08752 인용 확인 — 정확한 paper title / chapter / 식 번호 검증 또는 self-contained derivation 으로 대체 (수정사항 9 와 결합). | 필수 |
| 27 | plan §2.5 (수정사항 17 로 추가될 sub-section): lens1_table.csv column schema 명시 — `setting_id, C, n, n_val, P_train_y, P_val_y, kl_train_val, kl_val_train, w_vector, sigma_min_C_hat, cond_C_hat, bbse_err, kl_S_val, risk_weighted_proxy`. lens1_corr.csv 도 lens2_corr.csv schema 와 통일. | 필수 |
| 28 | plan §8 끝: *2 번째 라운드 critique 의 input* 으로 들어갈 minimum output set 명시 — `grand_df.csv, lens1_corr.csv, lens2_corr.csv, plus 4 개 H1-H4 scatter figure`. critic 의 primary outcome 명확화. | 필수 |
| 29 | plan §1, §3.2(3): `eigen_lambda_` (ρ) 의 cross-sidecar 일관성 확인, 다르면 그룹별 separate 분석. | 권장 |
| 30 | plan §1 sanity check (ii): random_results 의 method-간 일치 기준을 *strict* (정확 일치) → *moderate* (Spearman ≥ 0.9) 로 완화, sidecar 의 `random_seed_used` field 확인. | 선택 |

위 30 개 항목 중 **[필수] 20 개** 는 planner 가 plan 의 다음 revision 에서 모두
반영해야 lens 1-2 의 falsifier 가 statistically valid 하고 reproducible 합니다.
특히 수정사항 6, 7 (Nyström 식 + NTK schema 사전 확인) 은 lens 2 의 *수학적 정확성과
실현 가능성* 의 전제이고, 수정사항 11, 12, 13 (mixed model + multiple-testing +
falsifier 임계값) 은 setting 수 21/18 의 small-n 에서 H1-H4 가 *통계적으로
meaningful* 한 결론을 내기 위한 최소 필요조건입니다. 수정사항 27, 28 (output
schema 와 critic 의 minimum input set) 은 2 번째 라운드 critique 의 *실행 가능성*
의 전제. 권장/선택 10 개는 plan 의 robustness 와 narrative 명확성을 강화하는
보조 항목입니다.

## §11. 외부 자료 참고 — 본 비평이 사용한 식의 출처

- BBSE (Lipton, Wang, Smola 2018), arXiv 1802.03916, ICML 2018 Proceedings.
  http://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf. importance weight
  `w(y) = q(y)/p(y)`, confusion matrix `Ĉ_{ij} = P̂(ŷ=i, y=j)`, estimator
  `q̂ = Ĉ⁻¹ μ̂_q`, consistency Theorem 3.
- RLLS / regularized label shift (Azizzadenesheli, Liu, Yao, Anandkumar 2019),
  arXiv 1903.09734. BBSE 의 small-σ_min(C) 영역에서 regularized estimator 의
  안정성 개선 — multi-class setting 에서의 alternative.
- Unified view of label shift (Garg, Wu, Smyl, Lipton 2020), NeurIPS 2020,
  arXiv 2003.07554. weighted ERM excess risk bound 의 TV (Pinsker 로 KL) 형태.
- KTA (Cristianini, Shawe-Taylor, Elisseeff, Kandola 2001), NIPS 2001 vol. 14.
  https://papers.nips.cc/paper/1946-on-kernel-target-alignment. KTA 정의
  `A(K_1, K_2) = ⟨K_1, K_2⟩_F / √(⟨K_1, K_1⟩_F · ⟨K_2, K_2⟩_F)`, eq. (1).
- Centered KTA (Cortes, Mohri, Rostamizadeh 2012), JMLR. K-centering `K_c =
  H K H, H = I − 11⊤/n` 의 추가 효과 (uncentered Y 보다 더 일반).
- Nyström out-of-sample extension (Williams & Seeger 2000) NeurIPS, eq. (4)
  `ψ̂_i(x_new) = (√n / λ_i) · uᵢ⊤ k(x_train, x_new)`. 후속 표기는 Drineas &
  Mahoney 2005 JMLR vol. 6 의 eq. (5).
- iter_03 framework: `state/iteration_03/report.md` §1, §3, §6.
- iter_04 directive: `state/directives/20260603_iter04.md`.
- prior-art critique (이전 작업, 본 critique 와 분리): `state/iteration_04/
  critique_priorart.md`.

---

# §9. Executor 1st-round 결과 분석 — H1-H5 판정의 사후 점검

본 section 은 plan v2 (`state/iteration_04/plan.md`, critic-반영본) 를 따라 executor 가
산출한 1st-round 결과 (`grand_df.csv` 79,700 cell, `grand_meta.csv` 52 setting,
`lens1_table.csv`, `lens1_corr.csv`, `lens1_cell_kl.csv`, `lens2_table.csv`,
`lens2_corr.csv`, `ntk_schema.md`, 4 개 H1-H4 scatter PNG) 를 §1-§8 의 사전 비평
관점에서 다시 검증합니다. 1st-round 의 형식적 요약은 다음과 같이 정리됩니다.

| 가설 | 가설 내용 (1줄) | n | r (Pearson) | p (Bonferroni α=0.0125) | 결정 |
|---|---|---|---|---|---|
| H1 | trainbal random-loss-rate ↔ kl_val_S | 14 | −0.001 | 0.997 | reject |
| H2 | A1 recovery (valbal) ↔ max_P_train | 21 | +0.71 | 3.2e-4 | confirm |
| H3 | A1 recovery (valbal) ↔ kta_gap_r10 | 21 | −0.71 | 3.5e-4 | confirm |
| H4 | LR loss (trainbal) ↔ kta_train_r10 | 14 | +0.99 | 7.6e-11 | confirm |
| H5 | recovery ↔ d_eff_ratio_r10 | 21 | +0.14 | 0.54 | reject (예상된 lens 3 collinearity) |

이 표가 "executor 가 합격" 으로 끝나면 안 되는 이유 — 즉 H4 의 r=0.99 가 어떻게
*physical* 인과 신호가 아니라 *cross-dataset confounding* 의 산물일 가능성이
큰지를 §9.1 에서 정확히 짚고, §9.2-§9.5 에서 H1-H5 각각의 사후 점검, §9.6 에서
plan §6 의 mapping 이 결과에 의해 어떻게 보강/약화 되었는지, §9.7 에서 mnli cls90
INV missing cell 의 영향, §9.8 에서 다음 round 의 우선 task 를 정리합니다.

## §9.1 H4 의 r=0.99 는 dataset-level confounding 의 결과 — circular 는 아니나 mechanism evidence 가 아님

executor 가 산출한 H4 (`kta_train_r10` ↔ `LR_loss_vs_random_balanced_sel5`,
n=14 trainbal setting, Pearson r=+0.987, p=7.6e-11) 가 *너무* 깨끗합니다. 보통
NLP / kernel 분석에서 cross-dataset Pearson 이 0.99 까지 올라가는 경우는 두 변수
중 하나가 *trivially* 다른 하나의 함수일 때이거나, 두 변수가 모두 *dataset
identity* 만으로 결정되는 surrogate 일 때입니다. 사용자가 비평 1 번에서 제기한
"hidden dependency" 가 정확히 이 두 가능성입니다. 1st-round 결과를 정밀히 들여다
보면 다음의 사실이 드러납니다.

먼저 `kta_train_r10` 의 정의 (`experiments/lens2_kta_spectral.py` 라인 152-153,
plan §3.2(5) 의 self-contained derivation) 는 `(Σ_{i ≤ r} λᵢ · cᵢ²_train) /
(√(Σ_{i ≤ r} λᵢ²) · ‖Ỹ_train Ỹ_train⊤‖_F)` 이고, `LR_loss_vs_random_balanced_sel5`
는 `grand_df.csv` 의 `gap_top_random_balanced[LR, r=10%, sel=5]` 의 부호 반전
입니다 (lens2 script 라인 219-222). 두 quantity 는 *직접 산술 의존* 이 없습니다 —
KTA 는 train kernel 의 spectrum × train label 의 spread 이고, LR_loss 는 LR top-r
selection 으로 뽑힌 5 sample 의 down-stream balanced acc 와 random selection 의
balanced acc 의 차이. 두 quantity 가 같은 train kernel 의 spectrum 정보를 *공유*
하는 것은 사실 (LR 의 top-r 선택 자체가 cᵢ² 의 ranking 에 기반) 이지만, KTA(r) 의
수치와 LR top-5 의 down-stream training-then-eval pipeline 의 결과 사이에는 *분명히*
nonlinear 단계가 여러 개 있습니다 (top-r index 선택 → 5 sample subset → BERT
classifier eval 의 logit → predicted label → balanced acc). 따라서 *circular
definition* 은 *아닙니다*.

그러나 결과의 *cross-dataset 분포* 를 보면 다른 의심이 강해집니다. `lens2_table.csv`
의 첫 두 trainbal row (ag_news cls25_25_25_25 trainbal, val_ratio = cls55 vs cls85)
의 `kta_train_r10` 값이 *둘 다* 0.03905545820668073 으로 **identical** 입니다.
이건 KTA 가 train kernel 의 spectrum 만으로 결정되고 *train ratio 가 동일* (cls25
4-uniform balanced) 한 두 setting 에서 K_train 도 같고 Ỹ_train 도 같기 때문에
당연한 결과입니다. trainbal 정의 자체가 *"train 은 balanced 로 고정, val 만 imbalance
변경"* 이므로, `kta_train_r10` 는 *dataset 내에서 trainbal 의 모든 val_ratio 변형*
에 걸쳐 *constant* 입니다. 즉 trainbal 14 setting 중 7-dataset × 2-val_imbalance =
14 cell 에서 `kta_train_r10` 은 효과적으로 **7 개의 unique 값** 만 갖고 (각 dataset
당 1 값), 그 7 개 unique 값은 dataset 별 train kernel 의 spectrum 차이만 반영합니다.
correlation `r(kta_train_r10, LR_loss)` = 0.99 는 본질적으로 `r(dataset_id_one_hot
≈ kta_train_r10, LR_loss_per_dataset)` ≈ 0.99 — 즉 dataset 정체성이 두 변수 모두를
강하게 결정하는 confounded correlation 입니다.

이 점이 plan §3.3 의 *Stratification* (수정사항 21) 의 의도와 정확히 부합하는
*mechanical inflation* 의 사례입니다. plan 은 imbalance level 별 stratification 만
요구했지만, *dataset-level stratification* 은 빠져 있었습니다. 사실 trainbal 에서
train ratio 가 고정이라 KTA train 측은 dataset 의 NTK structure 의 직접적 measure
일 뿐이고, mechanism-level "label spread × spectral concentration" 의 변동성은
*거의 없습니다* — 변동의 source 는 7 datasets 의 model-kernel 차이입니다. 이를 확인
하려면 (i) Spearman r (rank-based) 가 Pearson r 보다 *얼마나* 낮은지, (ii) dataset-
level fixed effect 를 제거한 within-dataset residual 의 correlation 이 얼마인지,
두 점검이 필요합니다.

executor 의 `lens2_corr.csv` 행 H4 의 Spearman r=+0.853 (Pearson 0.987 보다 낮음)
이 첫 번째 단서입니다 — Spearman 이 낮아진 건 trainbal 의 14 setting 중 같은 dataset
의 두 cell 이 KTA 가 *완전히 같고* (tie) LR_loss 는 다르기 때문에 rank 상관에서
loss 가 큽니다. 두 번째로, dataset-level confounding 을 정량화하려면 mixed model
`LR_loss ~ kta_train_r10 + (1 | dataset)` 의 fixed β 의 CI 를 봐야 하는데, executor
의 `lens2_corr.csv` 에서 H4 의 `mixed_beta` column 이 비어 있습니다 (NaN). plan
§3.3 의 secondary tier 가 *구현되지 않음* — `experiments/lens2_kta_spectral.py`
라인 264-288 의 `_add` 함수가 `mb=np.nan` 을 default 로 받고 mixed model 호출이
없습니다. 이건 plan 의 수정사항 11 (필수) 의 *명시적 incomplete implementation*
이고, H4 의 confirm 판정의 통계적 validity 의 *최대* hole 입니다.

요약하면 H4 의 r=0.99 는 (i) circular definition 아님, (ii) 두 변수 사이에 진짜
mechanism 신호가 있을 가능성도 있음, 그러나 (iii) trainbal 의 setup constraint 로
`kta_train_r10` 가 dataset 별 7 unique 값으로 *degenerate* 되어 있어, 이 high
Pearson 은 mechanism evidence 가 아니라 *dataset-level 분산 흡수* 입니다. 진정한
mechanism evidence 를 얻으려면 (a) within-dataset residual correlation, (b)
mixed model 의 fixed β + CI, (c) trainbal 외 valbal 에도 같은 predictor (kta_train_r10)
를 적용해 *regime-cross* 일관성 확인 — 셋 다 plan §3.3 의 secondary/tertiary tier
인데 executor 가 *primary 만* 실행했습니다. 수정사항 31 — 2nd-round 분석으로
mixed model + within-dataset 잔차 correlation 을 별도 script 로 산출.

## §9.2 H1 reject 의 해석 — kl_val_S 단독이 무의미한가? interaction effect 의 가능성

H1 의 reject (n=14, Pearson r=−0.001, Spearman r=−0.144) 가 사용자 비평 2 의
지적과 정확히 일치합니다. 우리 plan §2.1 의 H1 가설은 "trainbal small-budget 의
random-loss-rate ↔ kl_val_S" 단변량 monotone 관계였고, p=0.997 의 r=0.001 은
*완전한 null*. 그러나 lens 1 의 kl_val_S 가 무의미한 quantity 가 아니라, *interaction
effect* 또는 *non-monotone dependence* 일 가능성을 cell-level 데이터에서 점검해야
합니다.

`lens1_cell_kl.csv` 의 ag_news trainbal cls55 row 들을 보면 — sel ∈ {1,2,3,5,10}
의 모든 cell 에서 `kl_val_S` 가 11.25 로 *constant* 입니다. 그 이유는 selected
subset P_S 가 작은 sel 에서 거의 100% majority class (P_S = [1.0, 0, 0, 0]) 로
collapse 했기 때문이고, KL 의 무한대 방향 항이 약 19.34, mixed class 의 평균이 약
11.25 의 plateau 에 머무는 것 — 즉 small sel 에서 `kl_val_S` 의 *cross-cell 변동성
자체가 작음*. H1 의 Pearson 이 0 인 핵심 원인은 KL 값의 *between-setting 분산이
LR_loss 의 between-setting 분산보다 훨씬 작은* 것이지, KL 이 mechanism 측 신호가
아닌 것이 아닙니다.

interaction 의 직관 — kl_val_S 가 큰 setting 에서도 (1) kta_gap_r10 이 작으면
collapse 가 작고, (2) kta_gap_r10 이 크면 collapse 가 커지는 형태일 수 있음.
즉 H1 의 true 형태는 `gap_top_random_balanced ~ kl_val_S × kta_gap_r10` (multiplicative
interaction) 또는 `~ kl_val_S | conditioned on KTA tier`. 이 sub-analysis 가
executor 의 1st-round 에서 빠져 있고, plan §2.3 의 *primary tier (setting 단위
Pearson)* 만 실행되었습니다.

수정사항 32 — 2nd-round 분석에 (i) `gap_top_random_balanced` 를 outcome 으로,
predictors {kl_val_S, kta_train_r10, max_P_val, max_P_train} 의 multivariate
linear regression (statsmodels OLS) 과 partial r² (variance partitioning), (ii)
`gap ~ kl_val_S * kta_train_r10 + (1 | dataset)` 의 interaction model, (iii)
small-sel ∈ {1, 2, 3} 와 mid-sel ∈ {5, 10} 의 *별도* correlation. 만약 interaction
의 β_interaction 이 significant (95% CI 가 zero 를 cross 안 함) 이면 H1 의 reject
는 단변량 형태의 reject 였을 뿐 lens 1 의 KL 이 mechanism 의 *moderator* 로 작용
한다는 evidence 가 됨.

또 lens1_table 의 trainbal row 들의 `risk_w_proxy_sel5` 가 음수값 (−0.005 부터
−0.076 까지) 으로 *직접 random-loss 의 magnitude proxy* 이고, kl_val_train (target →
source 방향, plan §2.2(1) 의 primary) 과의 Spearman 을 따로 측정해보면 sub-signal
이 살아날 가능성 — executor 의 lens1_corr.csv 가 *kl_val_S 만* outcome으로 잡고
`kl_val_train` 은 빠져 있습니다. 수정사항 33 — 2nd-round 에 setting-level
`risk_w_proxy_sel5` ↔ `kl_val_train` (Lipton-Wang-Smola 2018, arXiv 1802.03916 의
KL convention) 의 Pearson + mixed model 도 추가. 이 게 *lens 1 의 본래 prediction*
이고 H1 의 cell-level 변형보다 dimension 이 일치.

## §9.3 H2 와 H3 의 r=±0.71 가 동일 신호의 두 측면일 가능성 — predictor redundancy 의 정량화

H2 (`max_P_train` ↔ A1-LR balanced acc gap, n=21 valbal, Pearson r=+0.709,
Spearman r=+0.719, p=3.2e-4) 와 H3 (`kta_gap_r10` ↔ recovery_sel5, n=21 valbal,
Pearson r=−0.705, Spearman r=−0.650, p=3.5e-4) 가 둘 다 confirm 인데 사용자 비평 3
이 지적하듯 두 predictor 가 *correlated* 라면 같은 신호의 다른 측면입니다.
정량화 — `max_P_train` 과 `kta_gap_r10` 의 cross-setting Spearman 을 직접 측정.
이는 `lens1_table.csv` 의 `max_P_train` column 과 `lens2_table.csv` 의 `kta_gap_r10`
column 의 valbal 21 setting join 으로 즉시 계산 가능 (mixed model 불필요).

cross-row 직관 검산 — lens2_table 의 valbal row 중 ag_news cls25 (balanced)
trainbal 가 아니라 *valbal* 에 해당하는 cls25 row 의 `kta_gap_r10 ≈ 0` (train +
val 둘 다 balanced 라 alignment 차이 없음, max_P_train = 0.25). 반면 ag_news
cls85 valbal 의 `max_P_train = 0.85`, `kta_gap_r10` 가 negative 큰 값. 즉 두 변수
가 *비슷한 방향* 의 함수 관계를 갖고 monotone 일 expected. 이 가설을 정량적으로
검증한 것이 executor 의 1st-round 에는 빠져 있습니다.

`max_P_train` 와 `kta_gap_r10` 의 Spearman ρ_predictor 가 |ρ| > 0.8 이면 두 가설
이 *같은 latent 변수* 의 두 측정. < 0.5 이면 *상보적* (complementary). plan §3.5 의
multivariate regression 의 partial r² 가 정확히 이 분해를 수행 — `recovery_sel5 ~
max_P_train + kta_gap_r10` 의 두 predictor 의 partial r² 합이 *각각 따로 측정한 r²
의 합* 보다 작으면 redundancy 큼.

paper draft 의 narrative 차원에서 이 점이 결정적입니다. lens 1 의 `max_P_train`
(label shift 의 단순 majority probability) 이 lens 2 의 `kta_gap_r10` (kernel
target alignment 의 train-val gap) 과 redundant 하면 *lens 1 의 정량 contribution
이 거의 없음* — 즉 paper 의 main story arc 는 "lens 2 가 lens 1 을 *흡수* 하고
mechanism 의 single best predictor 이다" 로 강화됩니다. 그렇지 않고 complementary
면 "lens 1 = global label distribution 의 effect, lens 2 = kernel-side spectral
effect, 두 lens 의 *합* 이 mechanism 의 full picture" 로 framing.

수정사항 34 — 2nd-round 분석에 (i) `corr(max_P_train, kta_gap_r10)` valbal 21
setting 의 Spearman 정량화, (ii) `recovery_sel5 ~ max_P_train + kta_gap_r10` 의
OLS partial r² 분해, (iii) 동일 분석을 `max_P_val` (trainbal 의 H1 replacement)
와 `kta_train_r10` (trainbal 의 H4) 에 대해 trainbal 14 setting 에서도.

## §9.4 H4 의 r=0.99 의 cross-regime 일반화 — kta_train_r10 가 valbal recovery 도 예측하는가?

H4 가 trainbal 14 setting 에서 r=0.99 라면, 같은 predictor `kta_train_r10` 를
valbal 21 setting 에 적용했을 때도 *비슷한* 강도의 prediction 이 나와야 mechanism
의 *regime-independent* evidence 가 됩니다. valbal 의 outcome 으로는 `LR_loss_vs_random_
balanced_sel5` 대신 `recovery_sel5` (A1 의 LR 대비 회복) 를 쓰는 게 자연 — H3 와
같은 outcome.

executor 의 lens2_corr.csv 에는 이 *regime-cross* test (H4' = valbal 의 kta_train_r10
↔ recovery_sel5) 가 없습니다. 사실 H3 와 H4 가 *별도* 라고 framing 한 것 자체가
이 cross-test 를 우회한 것 — H3 는 valbal 의 *gap* predictor, H4 는 trainbal 의
*train-only* predictor 로 갈라져 있습니다. *common predictor* (kta_train_r10) 의
*regime-cross* analysis 가 빠져 있어, H4 의 r=0.99 가 trainbal 의 *특수한 confounding*
인지 mechanism 의 *general property* 인지 결정 불가.

valbal 의 21 setting 에서는 train ratio 가 *변동* 합니다 (ag_news cls25/cls55/cls85,
sst2 pos50/pos70/pos90 등), 그래서 `kta_train_r10` 가 dataset-degenerate 되지 않고
21 unique 값을 갖습니다 — 따라서 valbal 의 r(kta_train_r10, recovery_sel5) 가 H4
의 r=0.99 만큼 높으면 mechanism 의 강한 evidence, 0.3 부근으로 떨어지면 trainbal
의 r=0.99 는 dataset confounding 의 산물.

수정사항 35 — 2nd-round 분석의 *최우선* task: H4-regime-cross — valbal 21 setting
의 `recovery_sel5 ~ kta_train_r10` Pearson + Spearman + Fisher CI + mixed model.
만약 valbal r > 0.6 이면 mechanism evidence 강화, < 0.4 면 H4 의 trainbal 결과는
confounding warning.

## §9.5 H5 의 reject 는 예상된 — d_eff_ratio 가 모든 setting 에서 거의 일정

H5 (`d_eff_ratio_r10` ↔ recovery_sel5, n=21 valbal, Pearson r=+0.142, p=0.54)
의 reject 는 사용자 비평이 지적한 *lens 3 의 본래 한계* 와 일치합니다. lens2_table
의 첫 4 row 의 `d_eff_ratio_r10` 가 모두 0.10000 으로 *동일* 합니다 (5-decimal
이상 일치). 이건 NTK spectrum 의 *power-law decay* 가 매우 안정적이고 (BCP21
Sec. 3, Bordelon-Canatar-Pehlevan 2021), top-r mode 가 d_eff 의 *r/n proportion*
을 거의 monotone 하게 흡수하기 때문 — r=10% 이면 d_eff_r10 / d_eff_rho ≈ r/n
× constant 가 거의 1:1 으로 비례. 따라서 d_eff_ratio_r10 의 *cross-setting 분산*
이 거의 0 이고, 이 변수로는 어떤 outcome 도 예측 불가.

이 점은 plan §4 의 "lens 3 는 *수치보다 framework*" 진술과 일치하고, paper draft
의 §Theory 에서 d_eff 가 *직접적 predictor* 가 아니라 *KTA(r) 의 normalization
denominator* (i.e., d_eff(ρ) 가 KRR 의 effective number of parameters 라 KTA(r)
의 r 의 *operational meaning* 을 정해주는 역할) 로 들어가야 함을 시사. lens 3 의
재정의 — `d_eff_ratio` 대신 `top-r KTA / d_eff_ratio_r` (effective alignment
density) 를 시도. 단 이 quantity 가 H3/H4 의 raw KTA gap 과 redundant 할 가능성
이 높아 추가 contribution 은 marginal.

수정사항 36 — paper draft 에서 lens 3 는 *framework section* 으로 격하 (별도 lens
가 아니라 KTA 의 normalizer 로). H5 의 reject 는 *결과 section* 의 short paragraph
로 처리.

## §9.6 plan §6 mapping table 의 결과-사후 보강

plan §6 (iter_04 plan v2 의 framework mapping) 의 7 개 row 중 결과로 *보강/약화*
된 것:

(i) **Cond 1 ↔ LC_train(r)** — 보강. lens2_table 의 `lc_train_r10` 가 trainbal/valbal
모두 dataset 마다 monotone 증가 (ag_news 0.876→1.0, sst2 와 mrpc 도 비슷) 로 iter_03
의 LC_LR(r) 의 functional form 과 일치.

(ii) **Cond 2 cause-side ↔ KTA train + PR(c²)** — 부분 보강 + 약화. KTA train 의
H4 r=0.99 가 (warning §9.1) trainbal-only confounding 일 가능성. PR(c²) ↔ KTA
ratio 의 *correlation* (plan §3.5 의 redundancy 정량화) 가 executor 결과에 없어
direct verification 불가. lens2_table 의 `pr_c2_train` column 이 11.5 (ag_news 4-uniform)
부터 30 부근까지 분포 — iter_03 의 PR ranking 과 ordering 정합성 점검 가능
(executor 가 안 함). 수정사항 37 — 2nd-round 에 `pr_c2_train` 와 `kta_train_r10`
의 valbal/trainbal cross-setting Spearman 측정, redundancy 정량화.

(iii) **Cond 3 ↔ C (multi-class)** — plan v2 에서 *mapping 삭제* 의 권고 (수정사항
16) 가 정확. executor 결과는 C=2 (binary) 와 C=3,4 (multi-class) 의 분리 분석을
하지 않아 직접 검증 불가. paper draft 의 §Limitations 한 단락.

(iv) **valbal A1 recovery ↔ kta_gap_r10** (H3) — 보강 (r=−0.71 confirm). mechanism
narrative — A1 의 score (lens 3 의 sᵢ = (λᵢ/(λᵢ+ρ))² · cᵢ²) 가 high-λ mode 의
*label-aligned* 부분만 선택하므로, train kernel 의 KTA 가 *낮은데* val 측에서
*다른* alignment 가 필요한 setting 에서 A1 이 LR 대비 더 큰 회복을 보인다는 결과
와 일치.

(v) **trainbal LR collapse ↔ kta_train_r10** (H4) — *형식적* 보강, *내용적* warning
(§9.1 의 confounding). 2nd-round 의 regime-cross + mixed model 결과 후 최종 판정.

## §9.7 mnli cls90 INV missing cell 의 영향 분석

mnli cls90_05_05 의 trainbal INV (FreeShap) 가 누락 (사용자 비평 4) 이라 lens2_corr
의 trainbal H4 n_settings 가 14 (15 이 정상). missing cell 의 candidate impact
방향을 lens1_table 의 mnli trainbal row 로부터 추정 — mnli `valimb1000_cls90_05_05`
의 row (lens1_table line 11) 에서 `risk_w_proxy_sel5, sel10, sel100` 가 *모두
NaN/empty* 인데 train ratio cls33_33_33 + val ratio cls90_05_05 의 *극단적* imbalance
이라 trainbal random-loss 가 가장 큰 cell 일 가능성 큼.

이 setting 의 추정 LR_loss_vs_random_balanced_sel5 는 (extrapolation 으로) ag_news
cls25 trainbal cls85 의 0.553 또는 sst2 trainbal pos90 의 0.4 보다도 클 expected.
H4 의 fit `r=0.99, slope ≈ 11` (lens2 figure scatter 추정) 에서 mnli cls90 의
`kta_train_r10` (lens2_table 에서 mnli cls33 trainbal cls60 row 의 값 사용) 와
LR_loss 의 expected 가 fit line 의 extreme 점에 위치하면 r=0.99 가 더 강화될
가능성. 반대로 outlier 면 r 이 0.85-0.9 부근으로 *떨어질* 가능성.

수정사항 38 — mnli cls90 INV cell 의 rerun 우선순위: *높음*. 누락 cell 이 H4 의
가장 extreme x-값 (kta_train_r10 가 mnli 의 spectrum 으로 결정됨) 에 해당하므로
inclusion 이 fit 의 robustness 검증의 핵심. executor 에게 rerun task 를 별도 round
로 할당.

## §9.8 통계 절차의 *구현 누락* — plan §3.3 의 secondary/tertiary tier

plan §3.3 의 statistical procedure 3-tier 중 primary (setting 단위 Pearson +
Fisher CI) 만 executor 가 구현했고, secondary (cell 단위 mixed model) 와 tertiary
(dataset 별 within-dataset Spearman 분포) 가 *모두 누락* 되었습니다. `lens2_corr.csv`
의 `mixed_beta, mixed_ci_lo, mixed_ci_hi` column 이 *모든 row 에서 NaN*.

이 누락의 영향이 H4 의 r=0.99 판정의 신뢰도에 직격탄입니다 (§9.1 의 dataset
confounding 점검의 primary 도구가 mixed model β). 또 H3 의 r=−0.71 의 valbal 21
setting 도 Fisher CI [−0.87, −0.39] (lens2_corr 의 pearson_ci_lo/hi) 로 wide
하고, mixed model 의 β 가 *변동성* (random intercept 흡수 후) 을 좁힐 수 있는데
숫자가 없습니다.

수정사항 39 — 2nd-round 의 *필수* 항목: `experiments/lens2_mixed_model.py` 신설,
statsmodels MixedLM (`outcome ~ predictor + (1 | dataset)`) 으로 H1-H4 의 secondary
tier 산출. 이 결과를 `lens1_corr.csv`, `lens2_corr.csv` 의 mixed_* column 에
back-fill. 비용 ~ 10 분 (cell 수 8000, single core).

수정사항 40 — tertiary tier (dataset 별 within-dataset Spearman 의 median + IQR)
도 같은 script 에서 산출. 이는 dataset heterogeneity 의 직접 evidence — 7 dataset
중 *몇 개* 에서 H3/H4 가 *각각 hold* 하는지 fraction 으로 보고.

## §9.9 외부 자료 비교 — KTA 의 cross-task generalization theory

WebSearch 로 확인한 최신 KTA 관련 자료:
- *Cortes, Mohri, Rostamizadeh 2012 JMLR* "Algorithms for Learning Kernels Based
  on Centered Alignment" (이미 plan §3.2(4) 인용). centered KTA 의 generalization
  bound — `R(f) − R̂(f) ≤ O(√(complexity / n) / KTA)` 의 inverse-KTA 의존. 우리
  H4 의 *KTA train 큰데 LR 가 random 한테 짐* 의 mechanism 은 이 bound 의 *kernel-
  target mismatch 가 작아도 selection-induced sample-imbalance 가 ERM 의 effective
  task 를 KTA 와 misalign* 한 시나리오. 즉 우리 KTA train 측정은 *full-train* alignment
  이고, *top-5 selected* subset 의 alignment 는 *훨씬* 낮을 expect — Garg 2020
  의 weighted excess risk bound 의 `‖P_S − P_val‖_TV²` 와 일관.
- *arXiv 2505.03617* (Liu et al. 2025) "Understand the Effect of Importance Weighting
  in Deep Learning on Dataset Shift" — importance weighting 의 효과가 *training
  iteration 초기* 에 강하고 *말기* 에 fade 함. 우리 lens 1 의 weighted-balanced
  risk proxy 의 finite-sample variance 논의 (plan §2.2(4)) 와 직결 — BERT 의 ~3-5
  epoch 짧은 fine-tuning 에서 importance weight 의 효과가 strong 한 regime 일 expect.
  이 paper 가 우리 분석의 *training dynamics dimension* 의 next-iter 확장의 직접
  reference. bibliography 추가.

WebSearch 의 *quantum KTA* (Sahin 2024) 와 *NTK alignment training* (arXiv 2105.14301)
은 우리 problem 과 직접 관련은 약함. 무시.

## §9.10 §9 의 종합 — Round 2 의 우선순위 task

위 §9.1-§9.9 의 *수정사항 31-40* 을 우선순위로 정렬:

| # | task | 위치 | 우선 | 비용 |
|---|---|---|---|---|
| 39 | mixed model (secondary tier) 산출 — H1-H4 의 (`outcome ~ pred + (1|dataset)`) β + CI | `experiments/lens_mixed_model.py` 신설 | **필수** | 10 분 |
| 35 | H4 regime-cross — valbal 21 setting 의 `recovery_sel5 ~ kta_train_r10` | mixed_model.py 에 추가 | **필수** | 5 분 |
| 34 | predictor redundancy — `corr(max_P_train, kta_gap_r10)`, `corr(pr_c2_train, kta_train_r10)`, OLS partial r² 분해 | mixed_model.py 에 추가 | **필수** | 5 분 |
| 32 | H1 interaction model — `gap ~ kl_val_S * kta_train_r10`, multi-predictor OLS | mixed_model.py 에 추가 | **필수** | 5 분 |
| 33 | lens 1 의 setting-level risk_w_proxy ↔ kl_val_train (Lipton primary 방향) Pearson | mixed_model.py 에 추가 | 권장 | 2 분 |
| 38 | mnli cls90 INV cell rerun (executor task) + H1-H4 의 rerun | 별도 round | **필수** | 1 day (학습 time) |
| 40 | tertiary tier — dataset 별 within-dataset Spearman median + IQR | mixed_model.py 에 추가 | 권장 | 3 분 |
| 31 | within-dataset residual correlation (Pearson 의 dataset fixed effect 제거 후) | mixed_model.py 에 추가 | 권장 | 5 분 |
| 36 | lens 3 의 paper-draft 격하 — d_eff 를 KTA normalizer 로 재서술 | next_directions.md | 권장 | (draft) |
| 37 | `pr_c2_train` ↔ `kta_train_r10` redundancy 정량화 | mixed_model.py 에 추가 | 권장 | 2 분 |

종합 — *1 시간* 짜리 `lens_mixed_model.py` 가 round 2 의 80% 를 cover 하고, *1 day*
짜리 mnli cls90 INV rerun 이 나머지 robustness check. 두 task 모두 GPU 충돌 없음
(mixed model 은 CPU, mnli INV 는 standard BERT fine-tune).

## §9.11 sub-question (Q1)-(Q3) 의 결과-사후 답안 윤곽

(Q1) train ↔ val imbalance 위치가 collapse 양상을 바꾸는 mechanism — 1st-round
결과로 *부분* 답 가능. valbal 의 경우 H3 (KTA gap r=−0.71) 가 main mechanism —
*train kernel 의 top-r 가 train label 에 잘 align 되어 있지만 val label 분포가
balanced 로 shift 하면 train-side KTA 가 val 측에 옮겨질 때 gap 발생, 이 gap 이
큰 cell 에서 LR collapse 가 크고 A1 가 더 회복*. trainbal 의 경우 H4 가 (warning
§9.1 후에 valbal-side cross-test §9.4 통과 시) main mechanism — *train kernel 의
top-r 가 train label 에 *너무 잘* align 된 setting (KTA train large) 에서 LR
top-r 가 *majority* 만 잡아 (`P_S ≈ majority class only`, lens1_cell_kl 의 19.34
plateau 가 직접 증거) random-loss 발생*.

(Q2) A1 이 *정확히 어떤* quantity 를 회복하는가 — H3 의 negative correlation 의
의미. A1 score `sᵢ = (λᵢ/(λᵢ+ρ))² · cᵢ²` 가 LR 의 *naive top-cᵢ²* 선택과 달리
spectral filter `(λᵢ/(λᵢ+ρ))²` 로 *high-λ mode* 만 강조. high-λ mode 는 *kernel
의 dominant direction* 이고, 이 direction 의 label projection 이 *train + val 모두*
유의미할 expected (Caponnetto-De Vito 2007 의 source condition 의 *spectral
decay* 가정과 일관). 즉 A1 은 *train kernel 의 spectral dominance* 와 *label-aligned
mode* 의 *교집합* 을 잡고, LR 은 *label-aligned mode* 의 *모든* 것 (low-λ 도
포함) 을 잡는다는 차이. KTA gap 이 큰 setting 에서 train-only KTA 가 misleadingly
high 이고 (top-r 가 train 의 majority class 만 잡음), A1 의 spectral filter 가
*그 정보를 demote* 해 더 *generalizable* 한 mode 를 잡음.

(Q3) trainbal 의 random-loss 절반 case 의 operational 기준 — plan §2.1 의 정의
`P(gap_top_random_balanced[A1, r=10%, sel ∈ {1, 2, 5}] < −0.02)`. 1st-round 결과의
이 rate 가 정확히 몇 % 인지 — executor 가 lens1_corr 에 보고하지 않았음. grand_df
에서 직접 계산 가능 (cell ≈ 14 setting × 3 sel × 1 method = 42, 그 중 −0.02 이하
fraction). 수정사항 41 — 2nd-round 에 이 rate 의 dataset-별, setting-별 분포 산출
(simple count). 사용자 directive 의 *"절반"* 표현의 정량 검증.

## §9.12 §9 의 결론 — round 1 의 *형식적* success, 통계적·해석적 *불완전성*

1st-round 의 H1-H5 판정 (H2/H3/H4 confirm, H1/H5 reject) 의 *형식적* 결과는 plan
v2 의 falsifier 기준 (Bonferroni α=0.0125) 을 만족합니다. 그러나 *통계적 신뢰도*
와 *mechanism 해석* 의 두 차원에서 1st-round 는 불완전합니다. 구체적으로:

- **통계적 hole**: secondary tier (mixed model β) 와 tertiary tier (within-dataset
  Spearman) 의 *전체 누락* — H4 의 r=0.99 가 dataset-level confounding 인지 mechanism
  인지 분리 불가. plan §3.3 의 *3-tier 절차* 의 1/3 만 실행.
- **해석적 hole**: H2 와 H3 의 predictor redundancy 정량화 부재 — `max_P_train`
  과 `kta_gap_r10` 의 cross-setting Spearman 측정 없이는 lens 1 의 정량 contribution
  이 lens 2 와 *별도* 인지 *흡수* 인지 결정 불가. H4 의 regime-cross test 부재 —
  valbal 에서도 r > 0.6 인지 모르고는 trainbal 의 r=0.99 의 generality 미확정.
- **데이터 hole**: mnli cls90 INV missing — H4 의 extreme x-value 가 빠진 fit.
  rerun 후 H4 의 fit robustness 재검증 필요.

Round 2 는 위 hole 셋을 cover 하는 *1 시간 mixed_model.py + 1 day mnli rerun* 으로
충분 (수정사항 39, 35, 34, 38). 이후 paper draft 의 main story arc 가 *lens 2 의
KTA 가 dominant mechanism, lens 1 의 label shift 는 moderator* 로 정착 (lens 2 가
lens 1 흡수면) 또는 *lens 1 과 lens 2 의 complementary partition* (redundancy 약함
일 때) 의 두 시나리오 중 한 쪽으로 결정. next_directions.md 에 두 시나리오 별 paper
narrative 의 outline 정리.

## §9.13 §9 의 외부 참고

- Liu et al. 2025 "Understand the Effect of Importance Weighting in Deep Learning
  on Dataset Shift", arXiv 2505.03617. importance weighting 효과의 training-stage
  의존성 — paper draft 의 limitation 단락의 직접 reference.
- Cortes, Mohri, Rostamizadeh 2012 JMLR "Algorithms for Learning Kernels Based on
  Centered Alignment". centered KTA 의 generalization bound `R(f) − R̂(f) ≤
  O(√(complexity/n) / KTA)` — H4 의 KTA-train 의 *full* alignment 측정 vs *selected
  subset* 의 alignment 차이의 theoretical framing.
- 1st-round 결과 파일: `state/iteration_04/grand_df.csv` (79,700 cell),
  `lens1_corr.csv` (3 rows), `lens2_corr.csv` (8 rows), `lens2_table.csv` (52
  settings), `ntk_schema.md`.
- plan v2 §3.3 의 statistical 3-tier procedure 의 미구현 evidence: `experiments/
  lens2_kta_spectral.py` line 287 의 `mb=np.nan` default.

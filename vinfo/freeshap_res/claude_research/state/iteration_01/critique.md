# Iteration 01 critique (round 1) — planner v1 review

planner v1 (`state/iteration_01/plan.md`) 을 한 번 통독하고, directive 의
의무 4 가지 (novelty / 인용 정확성 / 숨겨진 가정 / 반례) 를 따라 짚어
보겠습니다. 결론부터 말씀드리면, 큰 그림에서 알고리즘 A1–A3 의 핵심
아이디어 — eq. (9) 의 spectral identity 를 supervised score 로 환원해서
top-r 을 재선택 — 자체는 LRFShap 안에서 자연스럽고 prior work 와도 충돌하지
않는 직교적 contribution 으로 보입니다. 다만 몇 군데 인용·페이지 매핑이
어긋나 있고, eq. (9) 도출 과정에 깔린 가정 중 plan 이 surface 시키지 않은
것들이 있어, plan_v2 에서 정리해야 합니다.

## 1. Novelty — A1–A3 가 prior work 변형이 아닌가

가장 신경 써야 할 비교 대상은 directive 가 콕 집은 Liu et al. (label-aware
base kernels, AAAI 2014 / Neurocomputing 2016) 과 Bach 2013 의 ridge
leverage 입니다.

먼저 Liu et al. 부터. ScienceDirect 와 ResearchGate 의 abstract 를 다시
보면 (sciencedirect.com/science/article/abs/pii/S0925231215010796, AAAI
2014 ojs.aaai.org/index.php/AAAI/article/view/8958), 이 라인의 핵심은
"label 정보를 주입해서 *새로운 base kernel* 의 eigenvector 를 ideal kernel
eigenfunction 의 extrapolation 으로 만든다" 입니다. 즉 출력은 *수정된 kernel*
또는 *수정된 eigenfunction set* 이고, 이걸로 KTA 를 끌어올린 뒤 SVM 류에
넣는 흐름입니다. plan 의 A1–A3 는 정반대 방향 — kernel K(X_N) 은 고정하고,
*그 자체의* eigendecomposition U Λ U⊤ 안에서 어느 r 개를 keep 할지를 score
s_i = (λ_i/(λ_i+ρ))² · (u_i⊤Y)² 로 결정합니다. eigenvector 자체를 새로
만들거나 extrapolate 하지 않으니, Liu et al. 의 변형이라고 보기 어렵습니다
(plan.md p.4, §4 Track B 후반부 mapping 이 이 점을 정확히 짚고 있습니다).
여기서 중요한 포인트는 "supervised eigenvector selection from a *fixed*
kernel basis" 가 실질적 novelty 의 핵심이라는 것이고, plan_v2 에서 이
구분을 한 줄 더 강조해주시면 좋겠습니다.

다음으로 Bach 2013 / Alaoui–Mahoney 2015 의 ridge leverage score
ℓ_i(ρ) = [K(K+ρI)⁻¹]_{ii} 와의 비교. 이 쪽은 *row/column* sampling 도구
(Nyström) 로 (u_i⊤Y) 를 보지 않고 column 을 뽑습니다. Track A mapping
(plan.md p.4 ~p.5 첫 단락) 도 이걸 옳게 적었습니다. 굳이 trivial 이 아닌
관점을 하나 더 추가하면, ridge leverage 의 mode-wise 형태 λ_i/(λ_i+ρ)
는 우리 score s_i 의 *square root × label term 제외* 에 정확히 대응
한다는 것입니다. 즉 s_i = (ridge leverage at mode i)² · (u_i⊤Y)² 로
factorize 되니, A1–A3 는 "ridge leverage 의 supervised refinement"
라고 한 줄 정의해두면 prior work 와의 자리매김이 깔끔해집니다. 이 표현을
plan_v2 §4 Track A 에 끼워 넣어주십시오.

마지막으로 A2 (LC/FC ratio greedy) 가 약간 의심스럽습니다. r_i = (u_i⊤Y)²
/ (λ_i ‖Y‖²) 는 Canatar–Bordelon–Pehlevan 2021 (Nat. Comm.) 의 task-model
alignment α_i = (u_i⊤Y)² / λ_i 와 정확히 같은 양으로, plan §4 Track C 가
인정한 대로 α_i 자체는 이미 generalization 분석 도구로 잘 알려져 있습니다.
A2 는 "α_i 를 selection criterion 으로 쓰자" 인데, α_i 가 작은 λ_i 를 분모에
밀어 넣어 noise mode 를 과대평가할 수 있다는 우려는 plan §3 A2 의 "위험"
서술이 이미 짚었지만, novelty 측면에서는 "Canatar et al. 의 α_i 를 그대로
selection criterion 으로 사용" 이라는 한정된 contribution 이 됩니다. A1
(s_i 가 ρ-aware) 과 A3 (partial decomp) 가 진짜 신선한 부분이고, A2 는
보조적 ablation 으로 위치시키는 게 정직합니다.

## 2. 인용 정확성

세 가지 mismatch 를 발견했습니다.

첫째, **eq. (9) 의 페이지 번호**. directive (latest_directive.md L17) 는
"§5.5 의 eq. (9) (p.7)" 라고 명시했는데, plan.md §2 (line 25) 는 "lrfshap.pdf
§A, p.12, eq. (8)–(9)" 로 인용했습니다. 즉 "main text 의 §5.5 (p.7)" 인지
"appendix §A (p.12)" 인지가 plan 안에서 충돌합니다. lrfshap.pdf 본 문서를
저는 텍스트 추출 환경 제약으로 직접 페이지 매김을 재확인하지 못했습니다
(poppler 미설치로 raster 실패). 사용자가 directive 에서 명시한 p.7 을
정답으로 보면, plan_v2 에서 페이지 번호를 p.7 (또는 main text 안의 식 번호)
로 통일해주시고, appendix 에 같은 식이 다시 등장한다면 "main eq. (9), p.7;
appendix §A 에서 동일 식 재기술" 같이 두 위치를 모두 명기해주십시오.

둘째, **Simon et al. eigenlearning 의 venue**. plan.md §4 Track C (line 184)
가 "Simon et al. Nature Comm 2023, 'The eigenlearning framework: A
conservation law perspective'" 로 적었는데, arxiv.org/abs/2110.03922 와
OpenReview (openreview.net/forum?id=FDbQGCAViI) 를 보면 이 논문은 Nature
Communications 가 아니라 OpenReview-only / TMLR 계열로 게재된 paper
입니다 (Simon, Dickens, Karkada, DeWeese, 2021–2023). plan_v2 에서 venue 를
"arXiv 2110.03922 / OpenReview 2023" 정도로 정정해주시고, "Nature Comm" 는
빼야 합니다. 같은 §4 Track C 에서 Canatar–Bordelon–Pehlevan 2021 은 실제로
Nature Communications 가 맞으므로, 아마 두 paper 의 venue 가 머릿속에서
섞인 것 같습니다.

셋째, **Liu et al. 의 저자명**. directive 와 plan 모두 "Liu et al." 로 부르고
있지만, 검색에서 나오는 Neurocomputing 2016 paper "Enhancing semi-supervised
learning through label-aware base kernels" 의 저자가 Wang, Q., Zhang, K.,
Chen, Z., Wang, D., Jiang, G., Marsic, I. 로 표시됩니다 (sciencedirect link
의 메타데이터). AAAI 2014 paper "Improving Semi-Supervised Target Alignment
via Label-Aware Base Kernels" 도 같은 그룹일 가능성이 높습니다. 즉 "Liu et al."
은 사용자가 처음 호명한 것이라 directive 단계에서부터 잘못된 이름일 수
있습니다. plan_v2 작성 시 ojs.aaai.org/index.php/AAAI/article/view/8958
의 author 페이지를 한 번 더 확인해서 저자명을 정확히 옮겨 적어 주십시오
(저는 critic 의 WebSearch budget 안에서는 abstract 까지만 봤고, author
list 는 직접 확인하지 못했습니다).

이외 인용은 — Alaoui–Mahoney 2015, Cortes et al. 2012, Park et al. TRAK
ICML 2023, Kwon et al. DataInf ICLR 2024 — 모두 venue/연도가 일치합니다.

## 3. 숨겨진 가정 / trade-off

plan §2 가 eq. (9) 를 "임의의 인덱스 집합 I 에 대해 정의되는 spectral
identity" 로 일반화했는데, 이 일반화 자체가 명시되지 않은 가정 두 개에
의존합니다.

(a) **Basis 가 K(X_N) 자체의 eigendecomposition 으로 고정된다**. 즉 I 는
{u_i} 인덱스 집합으로 한정. plan 도 본문 (line 32) 에 한 줄 흘려 적었지만,
"임의의 r-차원 subspace V ⊆ ℝⁿ 에서 minimize" 까지 확장한 게 아니라는 점이
독자에게 분명히 보이게 §2 첫 단락에 한 줄을 더 박아주십시오. 이게 중요한
이유는, 만약 V 가 자유 변수면 supervised optimum 이 "PCA-of-Y" 같은 다른
basis 로 가버려서 알고리즘 자체가 달라집니다.

(b) **In-sample, S = N 한정**. eq. (9) 는 X_N 위의 predictor gap 이고,
TMC permutation 안에서는 ‖f_S^FNTK(X_S) − f_S^LRNTK(X_S)‖² 같은 *coalition
S 에 대한* gap 이 진짜 신경 써야 할 양입니다. plan 이 §2 마지막 단락에서
"Prop 4.2 의 Shapley bound 가 ‖K̃ − K‖₂ 를 통해 predictor gap 의 proxy 로
나타난다" 고 정성적으로 연결했는데 (line 50–53), 이 연결이 정량적으로
보장되는 것은 S = N 한정이고, |S| = k < n 인 sub-coalition 에서 spectrum
이 다르게 잘리면 in-sample I\* 가 더 이상 optimal 이 아닐 수 있습니다.
plan_v2 §2 끝에 "S = N 의 minimizer 가 모든 |S| < n coalition 에 대해서도
average-case 로 유의한 개선을 줄 거라는 가정은 실험적으로 verify 해야
한다" 라고 한 단락 추가해 주십시오. (이건 다음 iter 의 실험 design 을
구체화할 때도 의미가 있습니다.)

(c) **ρ > 0 가정과 ρ → 0 한계**. plan §2 (line 41–46) 가 ρ → 0 에서 score
가 (u_i⊤Y)² 로 수축한다고 정확히 적었지만, ρ = 0 에서는 (λ_i/(λ_i+ρ))² 가
모든 i 에서 1 로 saturate 되니 score 가 *λ-independent* 가 됩니다. 그러면
A1 = "argmax over i of (u_i⊤Y)² 만으로 r 개 뽑기" 가 되는데, 이 경우
λ_i = 0 인 null direction (예: K 가 low-rank 일 때) 도 들어와 Φ_S⊤Φ_S + ρI
가 0 으로 나누는 사고가 생깁니다. lrfshap.pdf 가 ρ = 1e-3 같은 값에서
실험한다고 plan 이 적었지만, 실제 codebase (`../label_concentration/` 등)
가 ρ 를 어떻게 설정하는지를 한 번 확인하셔서 plan_v2 §3 A1 의 "위험" 에
"ρ 가 너무 작으면 null mode 가 selection 에 들어올 수 있어 minimum
λ-threshold ε 를 두는 게 안전" 이라고 한 줄 추가하시는 게 좋겠습니다.

(d) **Prop 4.2 의 Shapley bound 가 supervised selection 하에서도 그대로
유효한가**. 저는 Prop 4.2 의 statement 를 lrfshap.pdf p.5 에서 직접
재확인하지 못했지만 (위와 같은 raster 제약), plan §2 (line 50) 가 "‖K̃ − K‖₂
자리는 사실 in-sample predictor gap 의 proxy" 라고 *해석* 한 부분은
strong claim 이라 출처가 필요합니다. lrfshap 의 증명이 ‖K̃ − K‖₂ 를
operator norm 으로 다루는지 (그렇다면 supervised selection 에서도 norm
자체는 valid), 아니면 trace-norm / nuclear-norm 으로 다루는지에 따라 bound
의 형태가 달라집니다. plan_v2 에서 Prop 4.2 의 정확한 statement 한 줄
("‖φ̂_i − φ_i‖ ≤ … · ‖K̃ − K‖_? · …") 을 lrfshap 본문에서 옮겨 적고,
operator-norm 가정이라면 supervised selection 도 같은 norm bound 를
유지한다는 점을 명시해 주십시오.

(e) **Multi-class y 의 well-definedness**. plan §5 (line 200–207) 가
sum-of-squares Σ_c (u_i⊤y_c)² 형태로 통일하기로 결정했고, 이건
lrfshap.pdf §D.2 의 single-logit eNTK 구조와 정합적입니다. 다만 한
가지 확인할 점은, lrfshap §5.5 의 *원래* LC(r) 정의가 binary 인지
multi-class Frobenius 인지에 따라 numerator/denominator 의 normalization
이 달라진다는 것입니다 (binary: ‖P_r y‖² / ‖y‖²; multi-class Frobenius:
‖P_r Y‖_F² / ‖Y‖_F²). plan_v2 에서 lrfshap §5.5 의 정의 한 줄을 *그대로*
인용하고 (페이지·식 번호 포함), 우리 통일 정의가 그것의 multi-class
extension 임을 명시해 주십시오.

(f) **Partial decomp 의 cost 추정**. plan §3 A3 (line 131–133) 의 cost
0.24 n³ 추정은 randomized SVD 의 *flop count* 로는 맞지만, GPU 위에서는
dense O(n³) eigh 가 이미 매우 잘 최적화돼 있어 wall-clock 이 randomized
방식보다 빠른 영역이 n = 5000~10000 사이에 자주 있습니다 (plan §5 line
210–212 가 이 점을 정확히 인정함). 그렇다면 A3 의 "약 4× 빠르다" 라는
주장은 *flop 기준* 으로 한정해서 적고, wall-clock 비교는 "다음 iter 실험
에서 검증" 으로 명시하는 편이 안전합니다.

## 4. 반례 / falsifier

두 가지 시나리오에서 supervised selection 이 ER 을 *낮출* 수 있습니다.

**Falsifier #1: degenerate small-λ + large-(u⊤Y) mode**. spectrum 이
λ_1 ≥ … ≥ λ_n 으로 빠르게 떨어지다가 tail 끝에 λ_k ≈ 1e-8 인 mode 가
하나 있는데, 우연히 (u_k⊤Y)² 가 ‖Y‖²/n 수준으로 큰 경우. ρ = 1e-3 일 때
score s_k = (1e-8/(1e-8 + 1e-3))² · (u_k⊤Y)² ≈ 1e-10 · (u_k⊤Y)² 라
score 자체는 작아서 selection 안에 안 들어옵니다. 그런데 ρ = 1e-5 같이
더 작은 영역에서는 (1e-8/(1e-8+1e-5))² ≈ 1e-6 라 여전히 작긴 하지만, plan
A2 의 r_i = (u_k⊤Y)²/(λ_k‖Y‖²) 는 λ_k 가 분모라 *터집니다*. r_k ≈ 1e8 ·
(normalized) 라 A2 는 이 mode 를 무조건 1순위로 뽑고, Φ_tr 의 k-th column
이 √λ_k = 1e-4 norm 인 noise 로 들어가 ridge solve 의 effective conditioning
을 망가뜨립니다. 즉 **A2 는 small-λ degenerate 에서 ER 을 낮출 가능성**
이 명확히 있습니다 — plan §3 A2 "위험" 의 정성적 우려가 정량적으로 falsifier
가 됩니다. 검증 실험: synthetic setup 에서 K 의 spectrum 에 인위적 tail spike
를 박은 뒤 A1 vs A2 vs top-r-by-λ 의 ER 을 측정. tail spike 가 강해질수록
A2 의 ER 이 baseline 아래로 떨어지면 reject.

**Falsifier #2: Y 가 noise-aligned with low-eigenvalue modes** (label noise
시나리오). label 의 일부가 random flip 으로 망가져 있을 때, 큰 (u_i⊤Y)² 가
사실은 *noise component* 의 alignment 일 수 있습니다. supervised selection
은 이 noise mode 를 "유의한 supervised score" 로 선택해서 학습하지만,
unsupervised top-r by λ 는 큰 λ 만 보니까 *우연히* noise mode 를 거를
수 있습니다 (plan §3 A1 "위험" iii 가 이걸 짚었지만 정량적 시나리오로
정리하지 않음). lrfshap.pdf Fig. 2, p.6 에서 SST-2 1% selection 이
LRFShap 이 baseline 을 *초과* 한 것이 정확히 이 메커니즘일 가능성 — low-rank
truncation 이 implicit denoiser 였다면 supervised selection 은 그 효과를
잃습니다. 검증 실험: SST-2 의 train label 5–20% 를 random flip 시켜 놓고
A1 vs top-r-by-λ 의 ER 을 비교. flip rate 가 높을수록 A1 의 ER 이
baseline 아래로 떨어지면 partial confirm. 이 실험은 다음 iter 의 핵심
robustness check 으로 추가하는 게 좋겠습니다.

## 5. 정리

요약하면, plan v1 은 이론 도출 (§2) 과 알고리즘 구조 (§3 A1, A3) 가 견고
하지만, (a) eq. (9) 의 페이지 인용 (p.7 vs p.12 충돌), (b) Simon et al.
의 venue 오기, (c) Liu et al. 저자명 재확인, (d) Prop 4.2 statement 의
원문 인용, (e) S = N 한정 가정의 명시화, (f) ρ → 0 에서 null mode 회피
규약, 이 여섯 가지가 plan_v2 에서 반드시 보강돼야 합니다. A2 는 novelty
가 약하고 falsifier #1 에 취약하니 ablation 으로 위치 재조정. Falsifier
#1, #2 는 다음 iter 실험 design 에 명시적으로 들어가야 합니다.

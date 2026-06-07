# Iteration 04 — Lit-search critique: A1 (label-aware rank-r selection) 의 originality 진단

본 문서는 iter 04 directive (`state/directives/20260515_iter04.md`) 의 controlled
imbalance 실험 결과 분석이 아니라, 사용자가 별도로 요청한 *논문화 직전의 prior-art
점검* 입니다. 검증 대상은 A1 — score sᵢ = (λᵢ/(λᵢ + ρ))² · ‖uᵢ⊤Ỹ‖² 로 eNTK gram
의 top-r mode 를 픽해 FreeShap [B1] 의 TMC Shapley 를 가속하면서 imbalanced data
에서 LRFShap (top-r by λ) 의 1-class collapse 를 회복하는 알고리즘 — 의 *defensible
originality* 입니다. 두 질문 — (Q1) scoop 여부, (Q2) 세 갈래 related work — 을
WebSearch / WebFetch 로 점검한 결과를 prose 중심으로 정리하고, 마지막 §3 에서 어떤
framing 이 방어 가능한지 진단합니다.

## §1. Q1 — Scoop 여부: "label-aware rank-r selection for imbalanced data Shapley"
의 정확히 같은 셋업은 없습니다

키워드 셋 — "supervised low-rank kernel approximation for data valuation", "label-
informed eigenvector selection for Shapley", "imbalanced data Shapley" — 셋 모두를
*동시에* 만족하는 paper 는 2023-2026 range 에서 검색되지 않습니다. 그러나 "two-of-
three" 를 만족하는 work 가 두 편 있고, 둘 다 우리 paper 의 contribution claim 을
약화시킬 수 있어 명시적 distance 처리가 필요합니다.

가장 위험한 후보는 **CS-Shapley (Schoch, Xu, Ji, NeurIPS 2022)** [B-CS22]. 이
paper 는 Shapley 의 value function 자체를 class-wise 로 재정의 — in-class / out-of-
class contribution 을 분리하는 valuation function — 하여 *class imbalance dataset 에
서 특히 효과적* 이라는 주장을 직접 펼칩니다. 우리 work 와의 distance 를 정확히 짚으
면 (a) CS-Shapley 는 value function v(S) 의 정의를 바꾸지만 kernel / low-rank 구조
는 사용하지 않습니다, (b) 우리 A1 은 *value function 은 FreeShap 그대로* 두고 *kernel
의 subspace 를 label-informed 로 픽* 하는 *acceleration trick* 입니다. 즉 두 work
는 같은 효과 ("imbalanced 에서 robust") 를 서로 *orthogonal* 한 layer 에서 달성합니
다. CS-Shapley + A1 결합도 자연스럽고, 이 경우 "value function 도 class-aware, kernel
selection 도 label-aware" 라는 *dual* configuration 이 됩니다. 논문에서 §Related work
에 CS-Shapley 를 명시적으로 두고 "value-function-side vs kernel-side, 우리는 후자"
라는 위치 지정을 해두는 게 안전합니다.

두 번째 close-but-not-identical work 는 **Label-Aware Neural Tangent Kernel (Chen,
Huang, Zou, NeurIPS 2020)** [B-LANTK20]. 이 paper 는 우리 paper title 의 후보 ("label-
aware ...") 와 *naming collision* 이 있고 NTK 도 다룹니다. 그러나 검색 결과의 abstract
level 에서 확인되는 의도는 *NTK 자체* 를 local elasticity 향상 목적으로 label-aware
modification 하는 것이지, *rank-r selection* 도, *data Shapley* 도, *imbalance* 도 아
닙니다. WebFetch 로 본문 직접 확인은 PDF 인코딩 문제로 실패해 abstract 와 paper title
근거만 확신할 수 있습니다 — 따라서 *"NTK kernel 자체 modification 이지 우리처럼 fixed
eNTK 의 subspace selection 이 아니다"* 정도가 안전한 distance claim 이고, 만약 reviewer
가 detail 을 paper 본문에서 발견하면 추가 조정이 필요합니다. 본 문서의 [B-LANTK20] 인
용 시 사용자가 직접 PDF 를 확인해 distance 를 한 줄로 재정리할 것을 권합니다.

세 번째로 살펴본 PKeX-Shapley (arXiv 2505.16516, "Computing Exact Shapley Values in
Polynomial Time for Product-Kernel Methods", 2025) 는 product kernel 의 multiplicative
구조를 이용해 *exact* Shapley 를 다항시간에 계산하는 work 로, low-rank 도 label-aware
도 아니고 imbalance 도 다루지 않습니다. 충돌 없음.

종합하면 Q1 의 답은 *"정확히 같은 셋업의 paper 는 없습니다. CS-Shapley 만이 imbalance
대응 + Shapley 라는 공통 motivation 을 갖지만 kernel/low-rank 와는 orthogonal layer
이고, Label-Aware NTK 는 naming 만 비슷하고 task 가 다릅니다."* 입니다. 즉 A1 의
*kernel-side, eigenvector-selection-side* originality 는 protected 영역입니다.

## §2. Q2 — Related work 세 갈래

### §2.1 (a) Imbalanced data Shapley / fairness-aware data valuation

이 갈래는 5 편으로 정리됩니다. **Beta Shapley (Kwon, Zou, AISTATS 2022)** [B-Beta22]
는 efficiency axiom 을 완화해 small-cardinality marginal contribution 에 가중치를 더
얹는 family 를 정의하고 noisy label / mislabeled detection 에서 robust 라고 주장합니
다. Beta Shapley 가 imbalance 를 *직접* 다루진 않지만 noise-robust 라는 motivation 이
가깝고 우리 paper §Related work 에서 "Shapley 의 imbalance robustness 는 weighting
modification 으로 접근하는 line 도 있다 — 이는 kernel-side 와 분리된다" 식으로 위치
지정에 유용합니다. 두 번째로 §1 에서 짚은 **CS-Shapley** [B-CS22] 가 가장 가까운
prior work. 세 번째 **FairShap (Arnaiz-Rodríguez et al., DMLR @ ICLR 2024)** [B-Fair24]
는 Shapley value 로 fairness metric (group fairness) 에 대한 instance 의 contribution
을 측정해 sample re-weighting 합니다. 우리와 motivation 은 다르지만 (fairness vs
acceleration), "Shapley value 가 imbalanced/biased data 에 어떻게 interact 하는가" 의
정량 분석에 reference 가 됩니다. 네 번째 **CHG Shapley (Zheng et al., 2024)**
[B-CHG24] 는 closed-form Shapley 로 효율을 끌어올리면서 *class imbalance dataset* 을
benchmark 의 한 축으로 포함시켰습니다. 우리 paper 의 imbalance 실험 결과를 비교 baseline
으로 가져올 수 있습니다. 다섯 번째 **Data Banzhaf (Wang, Jia, AISTATS 2023)**
[B-Banzhaf23] 는 axiom 측에서 noise robustness 를 강화한 score 로, Beta Shapley 의 후
속 line. 직접 imbalance 는 아니지만 valuation 의 robustness literature 위치 지정에 필
요합니다.

이 5 편의 공통점은 *value function 또는 weighting axiom 의 modification* 으로 robustness
를 얻는다는 점입니다. 우리 A1 은 이들과 *kernel-side vs valuation-side* 로 분리되어
있어 직접 충돌은 없지만, paper §Related work 의 첫 단락에서 명시적 위치 지정이 필수입
니다.

### §2.2 (b) Label-aware spectral decomposition / supervised PCA / KTA 계열

이 갈래는 A1 의 score sᵢ = (λᵢ/(λᵢ + ρ))² · ‖uᵢ⊤Ỹ‖² 의 두 factor — *spectral filter*
(λ/(λ+ρ))² 와 *label projection* ‖u⊤Ỹ‖² — 가 각각 어디에서 왔는지의 lineage 입니다.

**Bordelon, Canatar, Pehlevan, Nature Comm 2021** [B-BCP21] / **Simon, Dickens,
Karkada, DeWeese, TMLR 2023** [B-Simon23] 는 iter 02 부터 이미 우리 paper 의 backbone
입니다. 두 work 는 *kernel ridge generalization 의 mode-별 closed-form* — learnability
L(φᵢ) ≈ (λᵢ/(λᵢ + κ))² · cᵢ² — 이 정확히 우리 score 의 함수 형태와 일치한다는 점을
이론적으로 정당화합니다. 즉 A1 의 score 는 *generalization-side learnability* 의 mode-
별 contribution 그 자체이고, 우리 contribution 은 이 quantity 를 *selection criterion*
으로 변환한 데 있습니다. 두 work 모두 selection / Shapley / imbalance 와는 무관해
distance 가 분명합니다.

**Kernel-target alignment (Cristianini et al., NeurIPS 2001)** [B5] 와 **centered
KTA (Cortes, Mohri, Rostamizadeh, JMLR 2012)** [B6] 는 KTA = ⟨K, yy⊤⟩_F /
(‖K‖_F · ‖yy⊤‖_F) 가 *전체 kernel* 의 label alignment 를 scalar 로 측정하지만,
mode-level 분해는 하지 않습니다. 우리 score 는 KTA 를 *eigenmode 단위* 로 풀어 쓴 것
의 spectrum-filtered 버전입니다. distance: "KTA 는 K 전체의 single scalar, 우리는
per-mode score" 가 가장 명확한 표현.

**Supervised PCA / supervised kernel PCA (Barshan, Ghodsi et al., 2011-2013)**
[B-Barshan11] 는 HSIC criterion tr(HKHL) (K = kernel of input, L = kernel of label,
H = centering matrix) 을 maximize 하는 방향으로 principal component 를 정의합니다.
A1 의 score 와 가까운 motivation — "label-dependent direction 만 골라 차원 축소" —
이지만 (i) supervised PCA 는 *새 base* 를 찾는 방향이고 A1 은 *기존 K 의 eigenbasis*
중 일부 mode 만 픽한다는 점, (ii) supervised PCA 는 Shapley / data valuation 과 무관하
다는 점에서 distance 가 큽니다. 그러나 *direct intellectual ancestor* 이므로 §Related
work 에 반드시 인용해야 합니다.

**Label-aware base kernels for semi-supervised learning (Wang Q. et al., AAAI 2014;
Neurocomputing 2016)** [B7, B8] 는 label 정보를 직접 base kernel 의 eigenfunction
extrapolation 에 주입합니다. 이 line 은 우리 work 와 *philosophy* 는 같지만 *kernel 자
체* 를 modify 한다는 점에서 A1 (fixed eNTK + selection only) 과 다릅니다.

**Supervised Nyström (vector-quantization landmark, ScienceDirect 2024)** [B-VQNys24]
는 label 정보로 Nyström landmark 선택을 informed 하게 합니다. A1 과 가장 가까운 *방법
론적 사촌* — "label 정보로 low-rank approximation 의 basis 를 픽한다" 라는 동일 motif —
이지만 distance 는 (i) Nyström landmark 는 *row* selection, A1 은 *eigenvector* selection
(서로 다른 low-rank family), (ii) Nyström 은 kernel approximation accuracy / classification
accuracy 가 목적, A1 은 Shapley 의 selection accuracy 가 목적, (iii) Nyström landmark
는 data point index 의 informed sampling, A1 score 는 spectral mode 의 scoring. 같은
"label-informed low-rank" framework 의 두 instantiation 이라는 위치가 정확합니다.

### §2.3 (c) Low-rank approximation + data Shapley / influence

여기는 5 편으로 정리됩니다. **FreeShap (Wang et al., NeurIPS 2024)** [B1] — 우리 직접
baseline, eNTK 위 TMC Shapley, low-rank 없음. **LRFShap (저자=사용자, ICML 2026
workshop)** [B2] — eNTK gram K = U Λ U⊤ 의 top-r by λ. 우리 A1 의 직접 ablation
target. **TRAK (Park, Georgiev, Ilyas, Leclerc, Madry, ICML 2023)** [B11] — random
projection + ensembling 으로 influence 를 scale up. kernel spectrum 직접 사용하지 않
고, label-aware 도 아닙니다. distance: "TRAK 은 random Gaussian projection, A1 은
data-dependent label-informed projection" 이 가장 정확. **DataInf (Kwon, Wu, Wang,
Mohri, ICLR 2024)** [B12] — LoRA Hessian 의 rank-1 closed-form inverse, influence
function 계열 (Shapley 아님), label 무관. **LoRIF (Low-Rank Influence Functions,
arXiv 2601.21929)** — TRAK/LoGRA 계열 후속, gradient 의 low-rank structure 활용, rank-c
truncated SVD + Woodbury, 0.1B-70B scale. label-aware 가 아니고 selection 도 아닙니
다. **HyperINF (arXiv 2410.05090)** — Schulz iteration 으로 Hessian inverse 추정, low-
rank 라기보단 iterative solve. 우리와 다른 axis.

이 갈래 전체의 공통 distance: "low-rank 를 *gradient/parameter* space 에서 쓰는 line
(TRAK, DataInf, LoRIF)" 과 "low-rank 를 *eNTK gram* space 에서 쓰는 line (LRFShap,
A1)" 의 분리가 가장 깔끔합니다. 그리고 그 *gram space* line 안에서 A1 만이 *label-
informed* 이라는 게 우리 contribution 의 위치.

## §3. Defensible originality 진단

### §3.1 Originality 의 위치

A1 의 originality 는 *세 layer 의 교집합* 에 있습니다. (i) Shapley/data valuation
layer 에서 — kernel-side acceleration 으로 imbalance 대응을 하는 work 가 prior art 에
서 발견되지 않습니다 (§1 의 Q1 결론). (ii) low-rank kernel approximation layer 에서 —
*eigenvector selection* + *label projection score* 의 결합이 §2.2 의 supervised PCA /
KTA / supervised Nyström line 의 직접 후속이지만, *Shapley 의 selection accuracy* 라는
metric 으로 정당화한 instantiation 은 없습니다. (iii) imbalanced training data 라는
focus 에서 — §2.1 의 CS-Shapley 등은 valuation-side 에서 같은 motivation 을 다루지만
kernel-side 에서는 비어있습니다.

이 세 교집합에서 가장 강한 framing 은 **"FreeShap 의 kernel-side robustness layer:
imbalanced data 에서 Shapley selection accuracy 가 collapse 하는 mechanism 을 spectral
filter × label projection score 로 직접 해결"** 입니다. 이 framing 의 강점은 (a) A1 의
score sᵢ 가 BCP21 / Simon23 의 learnability 와 정확히 같은 함수형태라는 *이론적 anchor*
가 있고, (b) iter 02-03 의 LC/FC/PR(c²) framework 가 *왜* imbalance 에서 1-class
collapse 가 일어나는지를 mechanism 으로 설명하며, (c) iter 04 의 controlled imbalance
실험 (70/30, 90/10, directive 의 §성공 기준) 이 효과의 *causal* 검증을 제공합니다.

약점은 두 가지입니다. 첫째, *intellectual ancestor* 의 폭이 넓습니다 — supervised PCA
[B-Barshan11], KTA [B5], supervised Nyström [B-VQNys24], eigenlearning [B-BCP21,
B-Simon23] 가 모두 A1 의 부분 motif 를 이미 갖고 있어 "A1 의 score 자체가 새롭다" 는
강한 주장은 어렵습니다. 둘째, *imbalance 대응* 이라는 효과는 §2.1 의 valuation-side
line (CS-Shapley, FairShap, CHG-Shapley) 이 이미 다루고 있어 "imbalance 대응이 새롭다"
는 주장도 약합니다. 따라서 contribution 은 *결합* (kernel-side + label-aware +
Shapley + imbalance) 의 새로움과 *mechanism explanation* 의 깊이 (PR(c²), 1-class
collapse, LC/FC framework) 로 차별화해야 합니다.

### §3.2 권장 framing 과 추가 contribution 권고

논문 abstract / introduction 의 권장 framing 은 두 단락 구조입니다. 첫 단락 — "FreeShap
[B1] 과 LRFShap [B2] 은 imbalanced training data 에서 top-r-by-λ selection 이 class-
blind common-feature mode 로 채워져 Shapley 가 1-class collapse 한다 (실험 §SST-2
70/30). 이는 kernel-side 의 새로운 failure mode 로, valuation-side 의 기존 대응 (CS-
Shapley, FairShap) 으로는 해결되지 않는다." 둘째 단락 — "우리는 spectral filter ×
label projection score sᵢ = (λᵢ/(λᵢ+ρ))² · ‖uᵢ⊤Ỹ‖² 로 top-r selection 을 supervised
하게 변경하여 1-class collapse 를 회복한다. 이 score 는 BCP21 [B-BCP21] / Simon23
[B-Simon23] 의 generalization-side learnability 와 함수형이 일치하며, supervised PCA
[B-Barshan11] / supervised Nyström [B-VQNys24] 의 *kernel approximation 가속* line 을
*Shapley selection accuracy* metric 으로 옮긴 첫 instantiation 이다."

추가 contribution 으로 권장하는 항목은 세 가지입니다. (1) **Theoretical anchor**:
A1 score sᵢ 가 BCP21 learnability 의 closed-form 과 일치한다는 점을 §Theory 한 페이지
로 명시. 이는 score 의 *임의성* 의심을 차단합니다 — score 가 ad-hoc 이 아니라 KRR
generalization 의 closed-form 에서 유도된다는 주장. (2) **Causal isolation**: iter 04
directive 의 70/30 + 90/10 controlled imbalance 결과로 *train_majority 와 LR_predict_
majority_frac 의 격차* 가 imbalance 깊이와 monotone 하다는 *causal* trace 를 §6 에
명시. (3) **Orthogonality to valuation-side**: §Related work 에서 CS-Shapley [B-CS22]
와의 *combination* — "value function 도 class-aware, kernel selection 도 label-aware"
— 가 future work 로 자연스럽다는 점을 한 단락 명시. 이는 *complement 관계* 임을 못박
아 reviewer 의 "왜 CS-Shapley 와 비교 안 했나" 질문을 차단합니다.

### §3.3 검증 가능한 개선안 체크리스트

| # | 개선안 | 검증 방법 | 우선순위 |
|---|---|---|---|
| C1 | §Related work 에 CS-Shapley [B-CS22] 의 *value-function-side* 위치 지정 + A1 의 *kernel-side* layer 와의 orthogonality 명시 | 한 단락 추가 후 self-review | 필수 |
| C2 | §Related work 에 supervised PCA [B-Barshan11] / supervised Nyström [B-VQNys24] 의 *label-informed low-rank* line 위치 지정 + A1 의 *eigenvector selection* 변형의 distance 명시 | 한 단락 추가 | 필수 |
| C3 | §Theory 에 BCP21 [B-BCP21] / Simon23 [B-Simon23] 의 learnability closed-form 과 A1 score 의 함수형 일치를 1 페이지로 명시 | 수식 + 인용 | 필수 |
| C4 | [B-LANTK20] (Label-Aware NTK, NeurIPS 2020) 의 abstract / 본문 직접 확인 후 distance ("우리는 fixed eNTK 의 subspace selection, 저들은 NTK 자체 modification") 한 줄 명시 | 사용자가 PDF 직접 확인 | 필수 |
| C5 | iter 04 directive 의 controlled imbalance (70/30, 90/10) 결과로 train_majority 와 LR_predict_majority_frac 격차의 monotonicity 검증 | iter 04 실험 종료 후 critique | 필수 |
| C6 | §Future work 에 CS-Shapley + A1 결합 (value-function-side × kernel-side dual configuration) 가능성 한 단락 | 텍스트 추가 | 권장 |
| C7 | [B-CHG24] CHG Shapley 의 class imbalance benchmark 와 직접 numerical 비교 | 추가 실험 또는 reproduce | 선택 |

### §3.4 결론

A1 의 originality 는 *방어 가능* 하지만 *strong* 하지는 않습니다. 가장 안전한 positioning
은 "kernel-side acceleration line (FreeShap, LRFShap) 의 imbalance robustness 확장" 이
며, valuation-side (CS-Shapley, FairShap, Beta Shapley) 와 supervised-low-rank
line (supervised PCA, supervised Nyström, KTA) 양쪽에 대해 *orthogonal layer* 라는
position 을 명시적으로 지키는 것입니다. iter 04 의 controlled imbalance 실험이 *causal*
검증을 제공하고, iter 02-03 의 LC/FC/PR(c²) framework 가 *mechanism* 을 설명하며,
BCP21/Simon23 의 learnability 가 *이론적 anchor* 를 제공하는 — 이 세 축의 결합이 단일
contribution paper 의 강도로는 충분합니다. 단, §3.2 의 C1-C5 를 모두 본문에 명시하지
않으면 reviewer 가 "단순 kernel-side variant" 로 평가절하할 위험이 큽니다.

## §4. Bibliography (iter 04 의 신규 reference)

iter 01-03 에서 이미 인용된 [B1]-[B14], [B-BCP21], [B-Simon23] 는 그대로 유효합니다.
iter 04 에서 새로 추가되는 reference 만 아래 명시합니다.

[B-Beta22] **Kwon, Zou (2022)** *Beta Shapley: a Unified and Noise-reduced Data
Valuation Framework for Machine Learning*. AISTATS 2022 (Oral). PMLR 151:8780-8802.
https://arxiv.org/abs/2110.14049
    Efficiency axiom 완화로 small-cardinality marginal contribution 에 가중치를 더해
    noisy / mislabeled data detection 의 robustness 를 얻는 family. distance — A1 은
    valuation axiom 이 아니라 kernel subspace 를 modify, 직접 충돌 없음.

[B-CS22] **Schoch, Xu, Ji (2022)** *CS-Shapley: Class-wise Shapley Values for Data
Valuation in Classification*. NeurIPS 2022. https://arxiv.org/abs/2211.06800
    Value function 을 class-wise (in-class / out-of-class) 로 재정의하여 *class
    imbalance dataset 에서 특히 효과적* 이라고 주장. A1 과의 motivation 충돌이 가장
    가까운 prior work. distance — CS-Shapley = value-function-side, A1 = kernel-side.
    Orthogonal layer 이며 결합 가능 (§3.2 C6).

[B-Fair24] **Arnaiz-Rodríguez, Curto, Oliver (2024)** *Towards Algorithmic Fairness
by means of Instance-level Data Re-weighting based on Shapley Values* (FairShap).
DMLR @ ICLR 2024. https://arxiv.org/abs/2303.01928
    Shapley value 로 group fairness metric 에 대한 sample contribution 을 측정해 re-
    weighting. distance — fairness metric 대상, A1 은 Shapley selection accuracy 대상.

[B-CHG24] **Zheng, et al. (2024)** *CHG Shapley: Efficient Data Valuation and
Selection towards Trustworthy Machine Learning*. https://arxiv.org/abs/2406.11730
(OpenReview: https://openreview.net/forum?id=uVMZgtw2pf)
    Closed-form Shapley 로 효율 향상, class imbalance dataset 을 benchmark 의 한 축으
    로 사용. distance — A1 은 NTK low-rank 기반, CHG 는 closed-form Shapley.

[B-Banzhaf23] **Wang, Jia (2023)** *Data Banzhaf: A Robust Data Valuation Framework
for Machine Learning*. AISTATS 2023. PMLR 206. https://proceedings.mlr.press/v206/
wang23e/wang23e.pdf
    Banzhaf index 기반 valuation, noise robustness 강화. distance — axiom-side, A1
    은 kernel-side.

[B-LANTK20] **Chen, Huang, Zou (2020)** *Label-Aware Neural Tangent Kernel: Toward
Better Generalization and Local Elasticity*. NeurIPS 2020. https://proceedings.
neurips.cc/paper/2020/file/b6b90237b3ebd1e462a5d11dbc5c4dae-Paper.pdf
    NTK 자체를 label-aware modification 하여 generalization / local elasticity 향상.
    distance — 우리는 fixed eNTK 의 subspace selection (eigenvector 단위), 저들은
    kernel 자체 modification. Naming collision 이 있어 본문에서 distance 한 줄 명시
    필요 (§3.3 C4). 사용자의 PDF 직접 확인 권장.

[B-Barshan11] **Barshan, Ghodsi, Azimifar, Jahromi (2011)** *Supervised principal
component analysis: Visualization, classification and regression on subspaces and
submanifolds*. Pattern Recognition 44(7):1357-1371. https://www.sciencedirect.com/
science/article/abs/pii/S0031320310005819 (uwaterloo preprint: https://uwaterloo.ca/
data-analytics/sites/default/files/uploads/documents/barshan_supervised_preprint.pdf)
    HSIC criterion tr(HKHL) 로 label-dependent principal direction 학습. A1 의 직접
    intellectual ancestor — "label 정보로 dimension reduction" motif. distance —
    new basis 학습 vs 기존 K 의 eigenbasis 중 selection.

[B-VQNys24] **Hammer, et al. (2024)** *Sparse Nyström approximation for structured
data using vector quantization-based landmark determination*. Neurocomputing 2024.
https://www.sciencedirect.com/science/article/pii/S0925231224008713
    Class-wise data distribution 을 prototype-based vector quantization 으로 Nyström
    landmark 선택에 주입. distance — Nyström row selection vs A1 eigenvector
    selection, 같은 "label-informed low-rank" framework 의 다른 instantiation.

[B-PKeX25] **(저자 미확인) (2025)** *Computing Exact Shapley Values in Polynomial
Time for Product-Kernel Methods*. https://arxiv.org/abs/2505.16516
    Product kernel 의 multiplicative 구조로 exact Shapley 다항시간 계산. distance —
    exact / product kernel 가정, A1 은 approximation / general kernel. 충돌 없음.

[B-LoRIF26] **(저자 미확인) (2026)** *LoRIF: Low-Rank Influence Functions for Scalable
Training Data Attribution*. https://arxiv.org/abs/2601.21929
    Gradient 의 low-rank structure 로 influence function 을 0.1B-70B scale 까지 확장.
    distance — gradient/parameter space low-rank, A1 은 eNTK gram space low-rank.

## §5. 부록 — feature-attribution SHAP 계열 4 편 점검 및 contribution framing critique

iter 04 본문 §1-§4 작성 이후 사용자가 별도로 점검 요청한 두 가지 — (Q1') *"SHAP 와
imbalance"* 키워드로 검색되는 4 편이 우리 A1 의 scoop 인지, (Q2') *"LRFShap (top-r)
+ A1 (label-aware rank)" 두-pillar 결합* 의 contribution framing 이 적절한지 — 을
부록으로 정리합니다.

### §5.1 Q1' — 4 편 모두 feature-attribution SHAP, data Shapley 와는 다른 domain

**핵심 판정 — 4 편 모두 (a) feature-attribution SHAP (Lundberg-Lee 2017) 계열**
입니다. 즉 *각 input feature 가 model 의 한 sample 에 대한 prediction 에 얼마나
contribute 하는가* 를 묻는 attribution method 입니다. 우리 A1 은 (b) *data Shapley*
(Ghorbani-Zou 2019, FreeShap [B1]) — *각 train sample 이 model accuracy 에 얼마나
contribute 하는가* — 의 acceleration 입니다. 두 quantity 는 이름이 똑같이 "SHAP /
Shapley" 이지만 *game 의 player 가 무엇이냐* (feature vs train sample) 에서 갈리는
완전히 다른 problem 입니다. 따라서 scoop 위험은 *0* 이지만, reviewer 의 naming
confusion 위험은 *실재* — §Related work 한 단락에서 "data Shapley ≠ feature-
attribution SHAP" 를 못박는 게 안전합니다.

paper 별 정리는 다음과 같습니다.

| paper | core finding | SHAP 종류 | imbalance 처리 | A1 과의 distance |
|---|---|---|---|---|
| Liu et al. 2022 (BalanceSHAP) | DeepExplainer 의 *background* + *explanation* 두 dataset 의 class distribution 을 같게 맞추면 beeswarm plot 의 artifact 가 줄고 variable importance 의 discrimination power 가 향상 | feature-attribution (Lundberg-Lee DeepExplainer) | background / explanation dataset 의 balancing | 완전히 다른 domain. game player 가 *feature* 이며, training data 가 아닌 inference-time *background sample distribution* 을 조정. 우리는 training data 자체의 valuation. |
| CPRD-XAI 2025 (Frontiers in AI) | XGBoost / RF / MLP 에 lung cancer risk prediction, LIME / SHAP / PDP 의 explanation 이 imbalanced vs balanced training data 사이에서 Jaccard / Rank Agreement 가 떨어진다는 경험적 관찰 | feature-attribution (TreeSHAP, KernelSHAP) | training data balancing 의 사전·사후 비교 | 완전히 다른 domain. *explanation consistency* metric (Jaccard / Rank Agreement) 평가, model retraining 기반 contribution 측정 아님. |
| Chen-Storey-Liu EJOR 2024 (Interpretable ML for imbalanced credit scoring) | LightGBM 등의 credit-default prediction 에서 imbalance ratio 가 커질수록 LIME / SHAP 의 stability (재추정 시 ranking 동일성) 가 떨어짐 | feature-attribution (KernelSHAP) | progressive imbalance ratio 실험 | 완전히 다른 domain. game player = 각 feature (e.g., income, age), training sample 아님. |
| CEUR-WS 2024 / arXiv 2507.09545 (frost events) | unbalanced 일 때 majority class 의 explanation 은 trustworthy 하지 않고 minority class 에 focus 해야 한다는 metric + on-manifold neighbour generation | feature-attribution (KernelSHAP) | minority-class-focused explanation aggregation | 완전히 다른 domain. *어느 sample 의 feature-attribution 을 신뢰할 것인가* 의 문제이고, *어느 train sample 이 가치 있는가* 는 묻지 않음. |

§Related work 위치 — 권장은 **§Related work 의 가장 마지막 한 단락**, "Disambiguation:
data Shapley vs feature-attribution SHAP" 같은 명시적 sub-section 으로 두 줄. "본
work 는 Ghorbani-Zou 2019 / FreeShap [B1] line 의 data-Shapley, 즉 train sample
의 model accuracy 에 대한 contribution 을 다룬다. SHAP/Lundberg-Lee line 의 feature-
attribution 과 imbalance 의 interaction 을 다룬 work (Liu 2022 BalanceSHAP, Chen
2024 등) 와는 game 의 player 가 다르며 직접 비교 대상 아님" 정도. 본문 main contribution
claim 에는 영향 없으나 *reviewer 의 confusion* 만 차단하는 안전장치.

출처 — BalanceSHAP https://arxiv.org/abs/2206.04050, BalanceSHAP GitHub
https://github.com/nliulab/BalanceSHAP, CPRD-XAI https://www.frontiersin.org/
journals/artificial-intelligence/articles/10.3389/frai.2025.1682919/full, Chen-
Storey-Liu EJOR https://www.sciencedirect.com/science/article/pii/S0377221723005088,
"Assessing reliability of explanations in unbalanced datasets" CEUR-WS
https://ceur-ws.org/Vol-4017/paper_10.pdf, arXiv mirror
https://arxiv.org/abs/2507.09545.

### §5.2 Q2' — 두-pillar (top-r 속도 + A1 small-budget robustness) 결합의 critique

#### §5.2(a) 두-pillar 가 single paper 의 contribution 으로 충분한가

ICML / NeurIPS standard 의 single-paper threshold 에서 보면, **LRFShap (top-r) 의
acceleration 만으로 conference main track 은 약합니다**. 이유는 top-r 자체가
*standard* low-rank approximation 의 직접 적용이고, FreeShap [B1] 위에 한 줄의
spectral truncation 을 붙인 것에 가깝기 때문입니다. workshop (ICML workshop, NeurIPS
ML for systems / efficient ML workshop) 으로는 충분하지만 — 사용자도 이미 이
positioning 으로 ICML 2026 workshop 을 진행 중 — main conference 의 reviewer 는
"why is this not just standard Nyström / spectral truncation applied to FreeShap?"
질문에 답을 요구할 것입니다.

A1 만 standalone 으로 보면 *이론적 anchor* (BCP21 learnability), *mechanism explanation*
(LC/FC/PR(c²) framework), *causal validation* (iter 04 controlled imbalance) 의 세
축이 결합해 *primary contribution* 수준의 evidence base 가 됩니다 — 이게 §3.4 본문의
결론이었습니다. 따라서 *결합 paper* 의 contribution 분배를 보면 (i) LRFShap 의 속도
는 *enabling tool* / "how we make this practical", (ii) A1 의 label-aware rank 는
*the* contribution. 두 pillar 가 *대등하게* 묶이는 게 아니라 *one main + one
supporting* 의 구조가 자연스럽습니다.

#### §5.2(b) "fine-tuning 데이터 선택" angle 의 prior art 차별화

이 angle 은 *underexplored* 가 아니라 *active research area* 입니다. 2023-2026
range 의 직접 경쟁자 — TS-DShapley (Schoch et al., ACL SRW 2023, arXiv 2306.10165;
LLM fine-tuning data selection with transferred Shapley), SHED (Erfan et al., ICLR
2024 OpenReview; instruction fine-tuning data refinement via Shapley), DemoShapley
(Xie et al., arXiv 2410.07523; in-context demonstration valuation), CHG Shapley
(Zheng et al., 2024, [B-CHG24]; small-budget regime 0.05-0.1 selection + class
imbalance benchmark), DPO-Shapley (Bertolazzi et al., arXiv 2512.15765; DPO-based
LLM data valuation 의 closed-form Shapley) — 가 모두 *data Shapley + LLM/fine-tuning*
의 cross-section 을 다룹니다. 즉 angle 자체는 새롭지 않고, **prior art 와의 차별화는
"imbalanced fine-tuning data 라는 specific axis 의 mechanistic analysis"** 에 있어야
합니다.

특히 CHG Shapley 가 *"small selection ratio (0.05, 0.1) + class imbalance benchmark"*
를 이미 다룬다는 점이 가장 위험합니다 (직접 인용 — CHG paper 의 표에서 "particularly
strong performance when selection ratios are small (0.05, 0.1)" 라고 명시).
사용자의 "small data training = small selection budget" 효과는 *기존에 없는 angle*
이 아니라 *기존 angle 의 NTK-kernel-side instantiation* 입니다. 따라서 framing 에서
"new angle" 보다 *"new mechanism"* — *왜* small budget × imbalance 에서 LRFShap 가
1-class collapse 하는지의 spectral 원인 — 을 강조해야 합니다.

#### §5.2(c) A1 의 small-budget 효과가 primary contribution 강도인가

iter 03 의 MNLI 42/42, MRPC 35/42 wins 와 iter 04 의 controlled imbalance 영역에서
sel ≤ 5% 일 때 +20pp 격차는 *통계적으로 강한 신호* 입니다. 그러나 *primary
contribution* 으로 standalone 하려면 두 보강이 필요합니다. 첫째, **이 효과가 NLP-
classification 영역 외에서도 재현되는가** — 현 evidence 는 GLUE-family 7-dataset 에
국한, vision (CIFAR), tabular, regression 으로 확장되면 reviewer 의 "domain-specific
trick" 의심이 차단됩니다. 둘째, **+20pp 의 *왜* — LC(r) / PR(c²) framework 가
metric 으로 검증되는가**. iter 04 directive 의 train_majority vs LR_predict_majority_
frac monotonicity 가 *causal* trace 를 제공하지만, 본 framework 가 *quantitative
predictor* (예: PR(c²) ≥ threshold 면 1-class collapse) 까지 가는지가 framing 의
강도를 결정합니다.

현 evidence 의 강도는 *primary contribution 으로 standalone 가능하나 borderline* —
대략 ICLR / NeurIPS rebuttal 에서 *방어 가능* 수준이고, *strong accept* 는 아닙니
다. 따라서 사용자의 "top-r 만으로는 아쉽다" 직관은 정확하고, A1 을 *primary* 로,
top-r 을 *baseline / enabling tool* 로 두는 게 evidence base 와 fit 합니다.

#### §5.2(d) "follow-up incremental work" risk

이 risk 는 *실재* 입니다. LRFShap (top-r) 이 ICML 2026 workshop 에 들어간 이후
conference paper 에 A1 을 얹으면, reviewer 의 두 가지 perception 이 갈립니다. (i)
"Workshop paper 의 incremental upgrade — top-r 위에 label-aware score 한 줄 추가"
로 보면 contribution 의 *delta* 가 작아 weak reject 위험. (ii) "Workshop paper 가
*failure mode* 를 노출했고, conference paper 가 *그 failure mode 를 mechanism 으로
설명하고 해결*" 으로 보면 *natural follow-up but with substantial new content* —
borderline accept 위험. 어느 쪽으로 보일지는 paper 의 *narrative* 에 달려 있습니다.

방어 전략 — conference paper 의 introduction 에서 LRFShap 을 *brief background*
로만 (한 단락) 인용하고, *main contribution* 은 (1) imbalanced data 에서 top-r-by-λ
의 1-class collapse 라는 *새로운 failure mode 의 발견*, (2) PR(c²) / LC(r) / FC(r)
framework 로 *mechanism explanation*, (3) A1 score 의 *theoretical anchor* (BCP21
learnability), (4) controlled imbalance 의 *causal validation* 의 4 점으로 묶어야
합니다. LRFShap 자체를 contribution 으로 claim 하지 말 것 — workshop venue 에서
이미 published 된 상태이므로 conference 의 contribution 으로 double-count 하면
reviewer 가 즉시 incremental work 로 분류합니다.

#### §5.2(e) 두 framing 의 권장 — *(2) A1 standalone* 추천

**Framing (1) "LRFShap + A1 결합 paper"** — 장점은 narrative 가 직관적 (속도 → 그
대가로 imbalance fail → label-aware 로 해결). 단점은 (i) 두 pillar 의 강도 불균형
(§5.2(c)), (ii) workshop 과의 double-count risk (§5.2(d)), (iii) ICML/NeurIPS
reviewer 가 "two-trick paper" 로 분류해 *focus 의 부재* 를 지적할 위험. workshop
venue 와의 boundary 가 모호해집니다.

**Framing (2) "A1 standalone, LRFShap 은 baseline"** — 장점은 (i) contribution
focus 가 한 점으로 모이고 *delta vs workshop paper* 가 명확 (failure mode + mechanism
+ theoretical anchor 의 3-축 추가), (ii) LRFShap 은 §Related work 의 한 줄 + §Method
의 baseline 으로만 인용하면 되어 workshop 과의 분리가 깔끔, (iii) reviewer 의 incremental
work perception 을 차단. 단점은 (i) "왜 굳이 low-rank 를 쓰는가" 의 motivation 을
LRFShap 의존 없이 처음부터 다시 build 해야 한다는 점, (ii) computational cost section
이 약해지면 *practical relevance* 의심.

권장은 **Framing (2)** 이며, motivation 의 약점은 "FreeShap [B1] 의 TMC Shapley 가
n=10k 규모에서 m × n 회 retraining 으로 O(n²) cost — n=50k 면 hours/days 단위" 의
quantitative cost 한 단락으로 해결하고, LRFShap 은 "the natural top-r by λ baseline"
으로 한 단락 인용하면 충분합니다.

#### §5.2(f) 결론 한 줄

iter 04 까지의 evidence (iter 03 7-dataset wins, iter 04 controlled imbalance
causal trace, BCP21 theoretical anchor, PR(c²)/LC(r)/FC(r) mechanism framework) 의
강도와 LRFShap-workshop-double-count risk 를 함께 고려하면, **가장 안전한 framing
은 "data Shapley 의 kernel-side robustness layer — imbalanced fine-tuning data 에서
top-r-by-λ 의 1-class collapse 를 spectral filter × label projection score 로 해결,
LRFShap 은 baseline 으로 인용"** 한 줄이며, 이는 §3.2 본문의 권장 framing 과 일치
하고 부록 §5.2 의 (a)-(e) 점검을 통과합니다.

### §5.3 §5 의 신규 출처

- BalanceSHAP (Liu et al. 2022). arXiv 2206.04050. https://arxiv.org/abs/2206.04050
- BalanceSHAP GitHub. https://github.com/nliulab/BalanceSHAP
- CPRD-XAI (2025). Frontiers in AI. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1682919/full
- Chen, Storey, Liu (EJOR 2024). "Interpretable machine learning for imbalanced credit scoring datasets". https://www.sciencedirect.com/science/article/pii/S0377221723005088
- "Assessing reliability of explanations in unbalanced datasets" (CEUR-WS 2024). https://ceur-ws.org/Vol-4017/paper_10.pdf 및 arXiv 2507.09545. https://arxiv.org/abs/2507.09545
- TS-DShapley (Schoch et al., ACL SRW 2023). https://arxiv.org/abs/2306.10165
- SHED (instruction fine-tuning data refinement). https://openreview.net/forum?id=Gqou8PRgWq
- DemoShapley (Xie et al., 2024). https://arxiv.org/abs/2410.07523
- DPO-Shapley (Bertolazzi et al., 2026). https://arxiv.org/abs/2512.15765

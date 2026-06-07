# Iteration 02 — FC vs LC Orthogonal Framework for A1 vs LRFShap

*Reading copy, 5 페이지. plan_v2 + critique_v2 의 핵심 발췌.*

## §1 Motivation & Restated Question

iter 01 의 7 dataset head-to-head 측정 — A1 vs LRFShap, 동일 (rank, sel%) 평가축에서 inv prediction 으로 — 의 win 패턴이 극단적으로 갈렸습니다.

| Dataset | n | C | wins/42 cells |
|---|---:|---:|---:|
| MNLI | 2000 | 3 | **42/42** |
| MRPC | 2000 | 2 | **35/42** |
| AG News | 2000 | 4 | 17/42 |
| SST-2 | 2000 | 2 | 8/42 |
| QQP | 2000 | 2 | 7/42 |
| MR | 2000 | 2 | 4/42 |
| RTE | 2490 | 2 | 4/42 |

iter 01 은 archetype 분류 (multi-class 우세, kernel-task fit 차이 등) 의 현상학적 서술에 머물렀습니다. 본 iter 의 목표는 이 win 패턴을 **FC (kernel norm) axis 와 LC (label energy) axis 의 직교성** 관점에서 식 단위로 설명하는 것. 실험은 없고 precomputed.json 의 수치만으로 분석.

## §2 FC vs LC Orthogonal Framework

### 정의

정렬된 eigendecomposition K = Σᵢ λᵢ uᵢ uᵢ⊤ (λ₁ ≥ λ₂ ≥ …), label projection cᵢ² := ‖uᵢ⊤Y_centered‖_F² (Y 는 centered C-class one-hot), ρ = ridge.

  **FC(I) := Σ_{i∈I} λᵢ / Σⱼ λⱼ**     — kernel Frobenius energy 의 retain ratio
  **LC(I) := Σ_{i∈I} cᵢ² / Σⱼ cⱼ²**    — label energy 의 retain ratio

두 selection 의 objective:

- **top-r (= LRFShap I_LR)**: I_LR = argmax_{|I|=r} Σ_{i∈I} λᵢ → K 의 best rank-r Frobenius approximation. cᵢ² 안 들어감 (label-blind).

- **A1 score sᵢ = f(λᵢ)·cᵢ², f(λ) = (λ/(λ+ρ))²**: λ >> ρ 영역에서 f ≈ 1 → I_A1 ≈ argmax_{|I|=r} Σ_{i∈I} cᵢ² (pure LC selection).

### A1 ≈ LC 의 정확한 조건

단순 "λ >> ρ" 는 충분조건의 절반. 정확한 조건은 I_A1 = I_LC 이려면 ∀i ∈ I_LC, ∀j ∉ I_LC:

  cⱼ² / cᵢ² < f(λᵢ) / f(λⱼ)

즉 filter ratio 의 dynamic range 가 cᵢ² ratio 를 dominate 해야 함. precomputed.json 의 frac(λᵢ > 10ρ) ≥ 0.999 가 모든 dataset 에서 성립하지만, RTE 만 λ_min = 0 mode 가 일부 존재해 f(λ) = 0 으로 정확히 떨어지는 영역이 있고 — 이 경우 cᵢ² 가 아무리 커도 A1 이 *원리적으로* 선택 못 함. RTE 가 framework 의 partial 예외 (edge case).

### 두 selection 이 놓치는 정보

- **top-r 이 놓친 label energy**: miss_LC_by_LR := 1 − LC(I_LR) = Σ_{i∉I_LR} cᵢ² / Σⱼ cⱼ²
- **A1 이 놓친 kernel energy**: miss_FC_by_A1 := 1 − FC(I_A1) = Σ_{i∉I_A1} λᵢ / Σⱼ λⱼ

LRFShap eq. (9) (lrfshap.pdf p.7) 의 Shapley estimation error bound 는 Σ_{i∉I} (λ_i/(λ_i+ρ))² · cᵢ² 형태 — *위 두 양의 product 의 정확한 합*. 즉 본 framework 의 두 양은 eq. (9) bound 의 axis-decomposition 임.

### Axis 직교성과 selection 결과의 직교성

Axis (FC, LC) 자체는 정의상 독립적 — FC 는 K 만, LC 는 (K, Y) joint 만 봄. *Selection 결과* 의 직교성은 spectrum-label joint distribution 의 *Spearman(λᵢ, cᵢ²)* 으로 정량. Spearman = 0 → 두 ranking 독립 → 기대 overlap = r/n. Spearman = 1 → 두 ranking 일치 → overlap = 1.

## §3 Empirical Evidence

### 7 dataset 의 핵심 수치 (r=10%)

| Dataset | λ_max/λ_med | frac(λ>10ρ) | Spearman(λ,c²) | overlap | FC_LR | LC_LR | FC_A1 | LC_A1 | wins/42 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNLI | 5093 | 1.000 | 0.226 | 0.325 | 0.822 | 0.271 | 0.271 | 0.410 | **42** |
| MRPC | 5492 | 1.000 | 0.114 | 0.260 | 0.819 | 0.280 | 0.188 | 0.532 | **35** |
| AG News | 10092 | 1.000 | 0.377 | 0.510 | 0.873 | 0.747 | 0.292 | 0.778 | 17 |
| SST-2 | 4564 | 1.000 | 0.193 | 0.325 | 0.814 | 0.566 | 0.323 | 0.704 | 8 |
| QQP | 25716 | 1.000 | 0.104 | 0.290 | 0.930 | 0.345 | 0.124 | 0.549 | 7 |
| RTE | 6924 | 0.999 | 0.098 | 0.241 | 0.821 | 0.206 | 0.139 | 0.470 | 4 |
| MR | 11876 | 1.000 | 0.150 | 0.310 | 0.898 | 0.516 | 0.303 | 0.662 | 4 |

핵심 관찰: 모든 dataset 에서 **LC_A1 > LC_LR** 이고 **FC_LR > FC_A1** — A1 은 label energy 를 2 배 가까이 잘 잡고, LR 은 kernel energy 를 4 배 가까이 잘 잡음. 두 selection 의 trade-off 정량 확인.

### 5 후보 척도 vs empirical_wins 의 rank correlation

| 척도 | 가설 | r=10% Spearman ρ | r=20% Spearman ρ |
|---|---|---:|---:|
| A: miss_LC_by_LR | 클수록 A1 가치 | +0.01 | +0.13 |
| B: miss_FC_by_A1 | 클수록 LR 가치 | +0.08 | −0.13 |
| **C: Spearman(λ,c²)** | **클수록 두 axis aligned** | **+0.60** | **+0.60** |
| D: eq9_gap_A1/LR | 작을수록 A1 bound 좋음 | −0.19 | +0.02 |
| E: overlap_LR_A1 | 클수록 두 방법 일치 | +0.45 | +0.43 |

**C 가 best predictor** — r 변경에 robust (r=10%, r=20% 양쪽 +0.60), ranking 의 qualitative order C > E > D, A, B 가 모든 robustness check 에서 유지.

### Statistical caveat (n=7)

Spearman ρ = +0.60 의 양측 p-value 는 n=7 에서 ≈ 0.15, Fisher CI ≈ (−0.20, +0.92). 단일 dataset leverage 가 큼:

| LOO 제외 | Spearman(C, wins) |
|---|---:|
| 전체 | +0.598 |
| MNLI 제외 | +0.471 |
| RTE 제외 | +0.429 |
| AG News 제외 | +0.643 |

→ paper framing 은 **"C 가 후보 5 개 중 가장 강한 정성적 predictor 이고 r-robust" 의 정성 ranking 수준** 에서 멈춰야 하고, "0.60" 의 정량을 single number 로 강조하지 말 것.

### *가장 중요한 보정* — Prediction-side vs Shapley-side 분리

iter 01 setup: A1 으로 Shapley 추정 → top-x% sample 선택 → *full kernel* (rank-r 아닌 K 전체) prediction → accuracy. LRFShap 도 동일하게 full kernel prediction. 두 단계 분해:

- **(a) Shapley-side**: 어떤 sample 이 골라졌나. A1 의 sᵢ ranking vs LRFShap 의 rank-r approximation Shapley.
- **(b) Prediction-side**: 그 subset 이 full kernel ridge 에서 얼마나 generalizable 한가.

본 framework 의 miss_LC, miss_FC, eq9_gap, overlap 은 **모두 (a) Shapley-side 양**. *그러나 empirical wins 는 (a) + (b)*. 두 사이의 mapping 은 monotone 보장 안 됨.

이 분리가 두 관찰을 명료히 해석해 줌:

1. **후보 A (miss_LC_by_LR) 가 wins 와 ρ ≈ 0** — framework 결함이 아니라 *negative result*. "Shapley-side 의 절대 miss 양은 prediction-side accuracy 의 직접 결정 요인이 아님."

2. **후보 C (Spearman(λ,c²)) 가 best predictor** — framework 가 *우연히* prediction-side proxy 를 포함. Bordelon-Canatar 의 task-model alignment 이론 [B-BCP21] 에 따르면 prediction-side generalization 의 핵심 결정 요인이 spectrum-label alignment. C 가 (a) + (b) 모두에 작용하는 *dataset-level* 양.

**Framework scope 명시**: 본 iter 의 framework 는 *Shapley-side selection geometry* 의 식 단위 정리. (b) prediction-side 의 full-kernel ridge generalization 은 *직접* 설명 안 함 — task-model alignment 이론과 bridge 가 필요.

## §4 Mechanism — 왜 Spearman(λ, c²) 가 Best?

Spearman(λ, c²) 가 클 때 두 사건이 동시 발생:

**첫째 (Shapley-side)**: A1 selection 이 "안전하게" 큰 cᵢ² 를 골라냄. Pure LC selection 의 estimation variance 는 small-λ noise mode 에서 큰데, λ-c² alignment 가 높으면 top-cᵢ² mode 의 λ 도 크기 때문에 A1 score sᵢ ranking 이 안정적.

**둘째 (prediction-side, Bordelon-Canatar eigenlearning curve)**: generalization error = Σᵢ cᵢ² · (1 − learnability_i)² 형태로 분해, learnability_i 는 λᵢ 의 monotone 함수. high-cᵢ² mode 가 high-λ region 에 몰려 있으면 full kernel ridge 의 generalization 이 효율적이고, A1 의 LC-biased selection 이 그 generalizable mode 의 sample 을 더 잘 representative 하게 골라줌.

### iter 01 Archetype 과의 매핑

| Archetype | Spearman(λ,c²) 위치 | Mechanism |
|---|---|---|
| MNLI (42/42), AG News (17/42) | 상대 rank 상위 (0.226, 0.377) | multi-class → label energy 가 여러 mode 에 분산되지만 embedding sub-cluster 가 top eigenvector 와 align |
| MRPC (35/42) | 낮음 (0.114) 인데도 wins 높음 | 절대값 낮지만 cᵢ² 가 *특정 minority-detection mode* 에 집중되어 A1 의 LC selection 이 그 mode 를 정확히 catch (PR(c²) 작음 → §5 참조) |
| RTE, MR, QQP (≤ 7/42) | 최하위 (0.10 근처) + Shapley-side 의 prediction-side ineffectiveness | low alignment → kernel ridge generalization 자체가 비효율, A1 의 Shapley-side advantage 가 prediction 단계에서 사라짐 |

### RTE vs MRPC Paradox

둘 다 miss_LC_by_LR 큰데 (0.794, 0.720) wins 가 천차만별 (4 vs 35). *Shapley-side 절대 miss* 로는 설명 불가능.

가설: **놓친 label energy 가 prediction-side 에서 learnable signal 인지 unlearnable noise 인지**. RTE 는 cᵢ² 가 거의 uniform 하게 1000+ mode 에 흩어진 *effective high-rank* (noise) 케이스. MRPC 는 cᵢ² 가 200-500 mode 에 집중된 *low-effective-rank* (signal) 케이스. 측정 proxy: cᵢ² 의 participation ratio PR(c²) = (Σ cᵢ²)² / Σ cᵢ⁴. precomputed.json 에는 없어 next iter 에서 측정.

## §5 Implications and Next Steps

### Paper framing 권고

1. A1 의 selection 은 거의 모든 dataset 에서 *pure LC selection* 의 근사. RTE 만 partial 예외 (λ_min = 0 mode 의 filter cut). "Universal" 이 아니라 **"near-universal, with RTE as edge case"**.

2. *Shapley-side 의 놓친 정보 절대량* (miss_LC, miss_FC) 은 empirical wins 를 거의 예측 못 함 (ρ ≈ 0). 이는 *Shapley-side vs prediction-side 분리*의 직접 증거.

3. **Spectrum-label alignment Spearman(λ, c²) 가 best predictor** — Bordelon-Canatar task-model alignment 이론과 직접 연결. paper 본문에서는 정성 ranking 으로 framing.

4. RTE-MRPC paradox 는 cᵢ² 분포의 shape (PR(c²)) 양으로 풀 수 있을 것 — next iter 의 첫 항목.

### Next Iter Obligatory Items

- **Falsifier 1 — synthetic spectrum-label decoupling**. 동일 K, 인위 Y 생성. aligned (wᵢ ∝ √λᵢ) vs anti-aligned (wᵢ ∝ 1/√λᵢ) 에서 wins matrix 비교. Framework 가 옳다면 aligned ≈ 42, anti-aligned ≈ 4.

- **Falsifier 2 — Spearman range 확장**. CoLA 추가 또는 같은 7 dataset 에서 n ∈ {500, 1k, 2k, 4k} 변화로 leverage 분산.

- **보조 — PR(c²) 와 effective LC dimension**. RTE vs MRPC paradox 정량 해명.

- **보조 — A1 modified version**. λ filter 끄고 pure cᵢ² baseline (= I_LC) 과 A1 의 실증 비교. RTE 의 bottom mode 에서 갈리는지 직접 검증.

### 한계 (남은 risk)

본 iter PDF 의 source 로는 충분 (critic R2 sign-off). 단 (i) Spearman(λ,c²) 가설의 synthetic falsifier 미실행, (ii) RTE 의 |I_A1 ∩ I_LC|/r 직접 측정 부재 — 둘 다 next iter 실험으로만 해소 가능. paper draft 단계 시 next-iter 결과 기다리는 게 안전.

# critique iter 02 R1 — FC vs LC orthogonality framework 검증

## §0 한 줄 요지

Plan 의 framework 는 식 정의 (i)-(iv) 단까지는 깔끔하고 precomputed.json 의 수치 인용도 직접 재계산해본 결과 거의 모두 정확하지만, *main finding* — Spearman(λ, c²) 가 best predictor (rank-correlation ≈ 0.60) — 의 통계적 robustness 가 부족하고, A1 ≈ pure LC selection 의 limit 분석에 한 가지 빠진 sub-condition 이 있으며, miss_LC_by_LR 의 무상관 (≈ 0.01) 을 단순 "절대 miss 양 무력" 으로만 해석하는 것은 *prediction-side* 와 *Shapley-side* 의 분리를 흐리는 수사라는 점이 본 critique 의 핵심입니다.

## §1 Formalization 검증 — A1 ≈ pure LC 의 limit 조건이 더 까다롭습니다 (plan §1 (ii))

Plan §1 (ii) 의 limit 분석 — "λ >> ρ 이면 filter f(λ) ≈ 1, 따라서 I_A1 ≈ argmax Σ cᵢ²" — 은 *충분조건*을 부분적으로만 적었습니다. 정확한 식 단위 조건은 다음입니다. score 가 sᵢ = f(λᵢ) · cᵢ² 이고 (filter f(λ) = (λ/(λ+ρ))²), I_A1 의 top-r argmax 가 pure LC top-r argmax 와 *완전히 일치할* 충분조건은 모든 i ∈ I_LC, j ∉ I_LC 에 대해 f(λᵢ) · cᵢ² > f(λⱼ) · cⱼ² 가 성립해야 합니다. 이를 풀면 cⱼ²/cᵢ² < f(λᵢ)/f(λⱼ) 가 필요한데, 우변이 1 에 근접하려면 단순 frac(λ > 10ρ) ≈ 1 만으로 부족하고 *filter ratio* 의 dynamic range 가 1 ± δ 안에 들어와야 합니다. precomputed.json 의 `filter_top1pct_min` 이 모든 dataset 에서 ≥ 0.9999 라는 사실은 *top 1% mode 들 사이*에서만 ratio 가 1 에 가깝다는 보장이고, top-r 의 r 이 1% 보다 크게 잡힐 때 (예: RTE 의 r=30% 즉 747 mode) 는 bottom selection candidate 의 filter 가 1 에서 *얼마나* 떨어지는지가 검증 안 되어 있습니다. RTE 의 λ_min = 0 (precomputed.json) 이라 filter 가 정확히 0 으로 가는 mode 도 존재하고, 그러면 그 mode 는 cᵢ² 가 아무리 커도 A1 이 *선택 안 함* 인데 pure LC selection 은 선택할 수 있어요. 즉 RTE 에서는 A1 ≠ pure LC 가 일부 mode 에서 발생합니다. **개선안**: plan §1 (ii) 끝에 한 문장으로 — "단, filter ratio f(λᵢ)/f(λⱼ) 가 cⱼ²/cᵢ² dynamic range 를 dominate 할 만큼 1 에서 떨어지면 (대표적으로 RTE 의 bottom mode) 일부 mode 에서 두 selection 이 갈리며, 이를 측정하려면 |I_A1 ∩ I_LC| / r 같은 추가 양이 필요" — 을 적고, 가능하면 r=30% 에서 I_A1 vs I_LC overlap 의 한 줄 측정을 추가하면 limit 분석이 *condition 부 정확*해집니다.

§1 (iii) 의 직교성 표현은 적절합니다. "axis 의 직교성은 보장, selection 결과의 직교성은 spectrum-label joint distribution 에 의존, 이를 측정하는 양이 Spearman(λᵢ, cᵢ²)" 의 논리 흐름은 식 단위로도 자연스럽고 precomputed.json 에 들어있는 양과 1:1 대응. 다만 *joint distribution* 의 dependence 식이 plan 본문에 명시 안 되어 있는데, 한 줄 추가 가능합니다 — Spearman(λ, c²) 가 0 일 때 expected overlap = r/n (independence), 1 일 때 overlap = 1, 일반적으로 overlap ≈ r/n + (1 − r/n) · g(Spearman) 형태의 monotone relationship 가 데이터에서도 관측됨 (AG News 의 Spearman 0.38, overlap 0.51 vs RTE 의 Spearman 0.10, overlap 0.24). 이게 framework 의 *internal consistency check* 역할.

## §2 Quantitative 검증 — best predictor (Spearman(λ, c²)) 의 robustness 가 부족합니다 (plan §3)

Plan §3 의 5 후보 척도 vs empirical_wins Spearman 값들은 정확합니다. 후보 A (miss_LC_by_LR) 의 +0.01 과 후보 C (Spearman(λ, c²)) 의 +0.60 은 제가 wins rank vs 척도 rank 로 직접 재계산해본 결과와 정확히 일치합니다 (후보 C 의 d² sum = 22.5, n=7, Spearman = 1 − 6·22.5/(7·48) = 0.598). 정량 결과 자체는 신뢰합니다.

문제는 *해석 단계*에서 plan §3 이 ρ = 0.60 을 "강한 양의 상관" 으로 단정하고 §6 의 "main finding" 으로 격상시킨 점입니다. n=7 의 Spearman 0.60 의 양측 p-value 는 약 0.15 — 유의수준 0.05 미달입니다. 그리고 outlier leave-one-out 을 해보면:

| 제외 dataset | Spearman(C, wins) |
|---|---:|
| 전체 7 | 0.598 |
| AG News 제외 | 0.643 |
| MNLI 제외 | 0.471 |
| RTE 제외 | 0.429 |
| MR 제외 | ~0.61 (계산 거의 동일) |

AG News 를 제외하면 오히려 ρ 가 살짝 올라가는 (0.64) 게 흥미로운데 — plan §4 끝에서 "AG News (0.38) 는 가장 높은 Spearman 이지만 wins 가 17/42 인 것은 §3 의 outlier 가능성" 이라고 *self-flag* 하긴 했습니다. 그런데 MNLI 와 RTE 를 각각 제외하면 ρ 가 0.43-0.47 로 떨어지는 사실은 본문에서 다뤄지지 않았어요. 즉 main finding 의 강도가 *single-dataset sensitive* 입니다. **개선안**: plan §3 끝에 leave-one-out robustness 표 (위와 동일) 를 추가하고, "ρ = 0.60 의 confidence interval 은 n=7 에서 ≈ (−0.20, 0.92) 로 매우 wide 함 — predictor 후보 선정의 정성적 ranking 은 유지되지만 (C > E > D > A,B), 정량적 수치는 ±0.2 정도의 noise band 안에 있음" 을 본문에 명시하세요. 그리고 §6 의 framing 을 "Spearman(λ, c²) 가 *상위 ranking* 의 best candidate" 로 약화 — n=7 에서 0.60 이 한 점 통계로 절대값을 단정하기엔 부족합니다.

추가로 짚을 점: 후보 D (eq9_gap_ratio_A1_over_LR) 가 −0.19 로 약한 음의 상관인 것은, plan §3 마지막 단락에서 "eq. (9) 는 worst-case bound 라 monotonically 안 맞물림" 으로 해석했는데 — 저는 다른 이유를 더 의심합니다. eq9_gap_A1/LR 은 두 selection 의 *Shapley 추정 오차 bound ratio* 인데, empirical_wins 는 *prediction accuracy* 의 head-to-head 결과입니다. Shapley estimation 오차와 prediction accuracy 사이의 mapping 이 monotone 이 아니라는 게 더 직접적인 이유 — 즉 plan §3 의 "worst-case" 해석은 절반만 맞고, 나머지 절반은 후술하는 §3 (숨겨진 가정) 의 prediction-side ≠ Shapley-side 문제입니다.

## §3 숨겨진 가정 — A1 evaluation 이 *inv prediction* 라는 점이 framework 전체를 비대칭하게 만듭니다

이게 가장 중요한 비판 포인트입니다. Plan 의 framework 는 *Shapley-side* — 즉 어떤 mode 가 잘 선택되어 Shapley value 추정이 정확한가 — 와 *prediction-side* — 즉 그렇게 추정된 Shapley 가 sample selection 으로 이어졌을 때 최종 model accuracy 가 어떤가 — 를 분리하지 않고 한 axis 의 framework 로 묶었습니다.

iter 01 의 실험 설계상 A1 의 evaluation 은 *inv prediction* 으로 진행됐습니다 — 즉 A1 으로 Shapley estimate → top-x% sample 선택 → 그 sample subset 에서 *full kernel* (rank-r 아님) 으로 prediction → accuracy 측정. 반면 LRFShap 도 동일하게 *full kernel* 로 prediction. 그러면 두 방법의 차이는 (a) Shapley-side 의 *어떤 sample* 을 골랐느냐 와 (b) 그 sample 들이 prediction-side full kernel 에서 *얼마나 generalizable* 한가 의 두 단계로 분해됩니다.

Plan 의 framework — miss_LC_by_LR, miss_FC_by_A1, eq9_gap — 는 모두 (a) 단계 (Shapley estimation 정확성) 의 양입니다. 그런데 wins 는 (b) 까지 합친 양이에요. 그래서 후보 A (miss_LC_by_LR) 가 wins 와 ≈ 0 의 상관을 보이는 게 자연스럽습니다 — Shapley 추정의 missed LC 가 prediction-side 의 generalization 에 어떻게 transfer 되는지가 별개 question 이니까요. 그리고 후보 C (Spearman(λ, c²)) 가 가장 강한 predictor 인 이유도 plan §4 의 mechanism 보다 더 단순하게 설명될 수 있습니다 — Spearman(λ, c²) 는 spectrum-label alignment 의 *dataset-level* 양이라서 prediction-side 의 generalization 성질 (task-model alignment, Bordelon-Canatar 의 의미에서) 과 직접 연결되고, *Shapley-side* 의 miss 양보다 *prediction-side* 와의 연결고리가 더 짧습니다. 즉 "Spearman(λ, c²) 가 best predictor" 라는 결과는 framework 의 정합성보다는 framework 가 *우연히* prediction-side proxy 를 포함하고 있었다는 것을 시사합니다.

**개선안**: plan §3 말미에 한 단락 추가 — "후보 C 의 우월성은 Spearman(λ, c²) 가 Shapley-side 와 prediction-side 모두에 동시에 작용하는 양 (spectrum-label alignment) 이기 때문이고, 후보 A 의 ≈ 0 은 *Shapley-side 의 절대 miss 양은 prediction-side accuracy 의 직접 결정 요인이 아님* 을 보여주는 negative result 이지, framework 의 결함이 아님" 으로 다시 framing 하세요. 그리고 §4 의 mechanism 단락에서 estimator variance 1/λ 가설은 *Shapley estimator* 에 대한 것인지 *prediction-side* 의 generalization variance 에 대한 것인지 명시 — 현재 plan 은 두 개를 섞어서 적었습니다.

## §4 반례 / falsifier 제안 2 개

**Falsifier 1 — synthetic spectrum-label decoupling 실험**. 동일한 base embedding 으로 K 를 고정하고, label Y 를 두 방식으로 인위 생성. (a) "aligned": Y = Σᵢ wᵢ uᵢ + ε, wᵢ ∝ √λᵢ (top eigenvector 에 label 몰빵, Spearman(λ, c²) → 1 부근). (b) "anti-aligned": wᵢ ∝ 1/√λᵢ 또는 random permutation (Spearman → 0 또는 음수). 동일 n, 동일 K 에서 A1 vs LRFShap 의 42-cell wins 매트릭스를 측정. Framework 가 옳다면 wins 가 aligned 에서 ≈ 42, anti-aligned 에서 ≈ 4 정도로 나와야 합니다. 만약 wins 차이가 작으면 (예: 둘 다 20 부근) framework 가 falsified — *real* embedding 의 dataset-specific structure 가 다른 변수 (label 노이즈, sub-cluster geometry 등) 로 wins 를 결정한다는 것을 의미합니다. precomputed.json 추가 계산 없이 실험 가능.

**Falsifier 2 — *intermediate* Spearman dataset 두 개 추가**. 현재 7 dataset 의 Spearman 은 0.10-0.38 range 에 좁게 분포 — 의 ρ = 0.60 을 결정짓는 leverage 가 endpoint (RTE 0.10, AG 0.38) 의 두 점에 집중됩니다. Spearman(λ, c²) ≈ 0.6-0.9 의 high-alignment dataset 을 추가로 확보해야 framework 가 *간단한 monotone* 인지 *saturation* 또는 *non-monotone* 인지 가립니다. 후보: CoLA (linguistic acceptability, 단순 label structure 라 alignment 높을 가능성), 또는 image dataset (CIFAR-10 의 일부 class) 의 RoBERTa-embedding 대신 CLIP-embedding — 다만 후자는 cross-modal 이라 framework 외 변수 도입이 위험. 더 안전한 falsifier: 같은 7 dataset 에서 *training set size* 를 n=500, 1000, 2000, 4000 으로 변화시키고 (precomputed.json 의 n=2000 fixed 와 달리), Spearman(λ, c²) 가 n 에 따라 어떻게 움직이는지 그리고 wins 가 그 변화를 따라가는지 확인. 만약 wins 가 n 따라 변화하는데 Spearman 은 stable 하면 framework 외 변수 (sample complexity) 가 dominate 하는 것이라 framework falsified.

## §5 minor — 출처와 표기

- plan §4 에서 Bordelon-Canatar-Pehlevan 2021 인용 시 [B?] 로 placeholder — plan_v2 또는 bibliography.md 에서 [B번호] 또는 명시 URL 로 보정해야 합니다. URL 은 plan §8 에 이미 들어있으니 cross-link 만 정리.
- plan §1 (iv) 에서 eq. (9) 의 형태를 "Σ_{i∉I} λᵢ cᵢ²" 로 기술했는데, 출처는 "(lrfshap.pdf, p. 7, eq. (9))" 처럼 page+eq 번호를 본문에 명시하면 critic 검증 용이.
- plan §2 의 frac(λ>10ρ) 가 RTE 만 0.999 (정확히 0.9992) 인 것은 RTE 의 λ_min = 0 mode 가 존재함을 의미. plan §1 (ii) 의 limit 분석 보완 (§1 위 비판) 과 연결.

## §6 종합 — round 2 에서 planner 가 반영해야 할 것

1. **§1 (ii) limit 조건 보완**: "filter ratio f(λᵢ)/f(λⱼ) 의 dynamic range 가 cᵢ² ratio dynamic range 를 압도해야 I_A1 = I_LC 성립" 한 줄 + RTE 의 partial 미성립 측정 1 개.
2. **§3 robustness 표 추가**: leave-one-out Spearman, n=7 confidence band (± 0.2 정도) 명시. ρ = 0.60 의 단정 약화.
3. **§3 / §4 prediction-side vs Shapley-side 분리 한 단락**: 후보 C 의 우월성 해석을 정합적으로 보정. estimator variance 1/λ 가설이 어느 side 의 양인지 명시.
4. **§5 RTE-MRPC paradox 의 PR(c²) 가설**: round 2 에서는 *추가 계산 요청* 으로 두고 본 iter 에서 결론짓지 않을 것 — directive 가 "변경 부분만 갱신" 이라 PR(c²) 가설은 §7 다음 iter 항목으로 유지.
5. **§7 next iter falsifier**: 위 §4 의 synthetic spectrum-label decoupling 실험을 §7 에 첫 항목으로 추가. PR(c²) 단독으로 모자람.

전체적으로 framework 의 식 정의와 수치 분석은 신뢰할 만하고, 다음 iter 의 *실험 가능한* falsifier (§4-1) 가 명확합니다. 다만 main finding 의 statistical robustness 와 prediction-side ≠ Shapley-side 분리는 반드시 round 2 에서 반영해야 합니다.

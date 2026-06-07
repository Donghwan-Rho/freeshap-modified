# critique iter 02 R2 — plan_v2 final sign-off

## §0 한 줄 요지

plan_v2 는 round 1 의 5 발견 사항 중 4 개를 식·표·단락 단위로 충실히 반영했고, 남은 1 개 (RTE 의 |I_A1 ∩ I_LC|/r 직접 측정) 는 next iter 항목으로 명시 deferral 된 상태입니다. paper framing 의 hedging 강도와 statistical robustness 보강이 plan v1 대비 명확히 개선되어 PDF report 5 페이지의 source 로 진행 가능합니다.

## §A round 1 critique 5 항목의 해소 상태

| critique R1 항목 | plan_v2 반영 위치 | 상태 |
|---|---|---|
| §1 — A1 ≈ pure LC limit 의 충분조건 보완 (filter ratio f(λᵢ)/f(λⱼ) 가 cⱼ²/cᵢ² ratio 를 dominate, RTE λ_min=0 의 partial 미성립) | §1 (ii) 단락 전체 (lines 19–26) + §6 conclusion 1 (line 158) | **Resolved**. 식 cⱼ²/cᵢ² < f(λᵢ)/f(λⱼ) 명시, RTE 의 f(λ)=0 mode 가 *원리적으로* I_A1 ≠ I_LC 를 강제한다는 점 정확히 기술. 다만 |I_A1 ∩ I_LC|/r 의 r=30% 직접 측정은 §1 (ii) 끝 (line 26) 에서 "round 3 또는 next iter" 로 명시 deferral. plan v1 의 partial framing 이 critic 의 요구 수준까지 올라옴. |
| §2 — Spearman ρ=0.60 의 statistical robustness (n=7, p≈0.15, LOO sensitivity) | §3.1 (lines 79–91) 의 LOO 표 + Fisher CI band ±0.2 + §3.3 (line 105) 의 "single number 로 강조하지 않음" 정성 framing + §6 conclusion 3 (line 162) | **Resolved**. LOO 표는 critic 이 제시한 수치 (0.471 MNLI-out, 0.429 RTE-out) 와 1:1 일치. Fisher CI (−0.20, +0.92) 명시. §3.2 의 r=20% robustness check 는 critic 요구를 *넘어선* 추가 보강 — C 가 r 에 비의존이라는 분석이 신선합니다. paper framing 약화 의도가 §6.3 에 명시되어 final framing 통제됨. |
| §3 — prediction-side vs Shapley-side 분리 | §3.5 단락 전체 신설 (lines 107–124) + §4 mechanism 의 두 단계 분해 (lines 130–134) + §6 conclusion 2 (line 160) | **Resolved**. plan v1 에 없던 단락이 critic 의 비판을 본문에 정확히 흡수. 후보 A 의 ρ≈0 을 "negative result" 로 재해석, 후보 C 의 우월성을 Bordelon-Canatar task-model alignment 와 연결, framework 의 scope 를 "Shapley-side selection geometry" 로 한정한 진술 (line 124) 까지 들어갔습니다. critic 의 가장 중요한 비판이 가장 깊이 반영된 항목. |
| §4 — falsifier 2 개 (synthetic spectrum-label decoupling, intermediate Spearman dataset) | §7 lines 170–172 | **Resolved**. critic 이 제안한 falsifier 1 (aligned/anti-aligned synthetic Y) 과 falsifier 2 (CoLA 추가 또는 n-sweep) 가 next iter 의 *obligatory items* 로 격상되어 본문 그대로 인용. PR(c²) 가 보조로 강등된 것은 critic 의 §4 우선순위 (Spearman 가설 검증이 PR(c²) 보다 선행) 와 일치. |
| §5 — 출처 보정 ([B-BCP21] placeholder, lrfshap.pdf p.7 eq.(9) 명시) | §1 (iv) line 35 의 "lrfshap.pdf, p. 7, eq. (9)" 직접 인용 + §3.5 line 122 의 [B-BCP21] URL + §8 bibliography (lines 180–183) | **Resolved**. 세 인용 모두 page/eq/URL 단위로 정합. minor 항목까지 깔끔합니다. |

요약: 5 항목 모두 **Resolved**. 단 §1 의 |I_A1 ∩ I_LC|/r 직접 측정은 plan_v2 본문에서 측정값이 *직접 계산되어 들어간 것은 아니고* deferral 된 형태라 엄밀히 따지면 "본 iter 의 분석 범위 안에서는 Resolved, 측정 자체는 next iter" 의 hybrid 상태입니다. paper framing 으로는 "RTE 는 partial 예외" 의 hedging 이 본문에 충분히 깔려 있어 PDF source 로는 문제 없습니다.

## §B plan_v2 에서 새로 발견된 미세 문제

추가 발견 없음. §3.2 의 r=20% robustness 표는 critic 이 요청 안 한 보강이라 검토해봤는데, 후보 C 가 r 무관 +0.60 인 것은 정의상 (cᵢ² ranking 과 λᵢ ranking 자체의 Spearman 은 어떤 r 을 잡아도 변하지 않음) 자명하고 plan_v2 가 line 99 에서 그 점을 솔직히 적시한 게 정합적입니다. 후보 E (overlap) 의 r 안정성 (+0.43 → +0.45) 은 비자명한 결과로 추가 validation 가치가 있습니다.

## §C 최종 신뢰도 / 남은 risk

plan_v2 는 critique R1 의 5 항목을 모두 본문에 흡수했고 (4 Resolved + 1 본문 hedging + next iter deferral 의 hybrid), framework 의 scope 가 "Shapley-side selection geometry" 로 명시되어 over-claim 위험이 제거됐습니다. 정량 결과 (precomputed.json 의 수치 인용, LOO 표, r=20% robustness) 는 round 1 에서 이미 재계산 검증된 상태라 추가 의심 없습니다. 남은 risk 는 두 가지로 모두 *next iter 의 실험으로만 해결 가능* 한 항목입니다 — (1) Spearman(λ, c²) 가설의 falsifier (synthetic decoupling) 가 아직 안 돌아간 상태, (2) RTE 의 |I_A1 ∩ I_LC|/r 직접 측정 부재. 두 항목 모두 §7 에 obligatory items 로 명시 deferral 되어 있어, 본 iter PDF report 5 페이지의 source 로서는 **PDF 진행 가능**입니다. 한 라운드 더 돌릴 정당화는 없습니다.

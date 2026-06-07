# Iteration 01 critique (round 2, final sign-off) — planner v2 review

planner v2 (`state/iteration_01/plan_v2.md`) 를 round 1 의 6 항목과 한 줄씩
대조해 봤습니다. 결론부터 말씀드리면, 6 항목 모두 본문에서 명시적으로
처리됐고 미세한 open item 두 건만 남아 PDF reading copy 의 source 로
충분합니다.

## (A) Round 1 critique 6 항목 해소 상태

| # | Round 1 발견 | plan_v2 위치 | 상태 |
|---|------------|-------------|------|
| (a) | eq. (9) 페이지 인용 충돌 (p.7 vs p.12) | §2 첫 단락 (line 31–32): "본문 §5.5 (p.7, eq. (9)) — 그리고 같은 식이 appendix §A (p.12, eq. (8)–(9)) 에서 재기술" 로 두 위치 병기 | **Resolved** |
| (b) | Simon et al. venue 오기 ("Nature Comm 2023") | §4 Track C (line 244–250): "TMLR 2023" 로 정정, arXiv 2110.03922 + OpenReview ID 병기, v1 오기를 명시적으로 인정 | **Resolved** |
| (c) | "Liu et al." 저자명 재확인 | §4 Track B (line 229–238): Wang, Q., Zhang, K., Chen, Z., Wang, D., Jiang, G., Marsic, I. 로 정정, AAAI 2014 / Neurocomputing 2016 두 paper 모두 URL 포함 | **Resolved** |
| (d) | S = N 한정 가정 명시화 | §2 line 40–46 ("(a) fixed-basis selection only"), line 62–70 ("S = N 의 minimizer 가 \|S\| < n coalition 에 transfer 된다는 가정은 다음 iter 실험에서 verify"). 두 가정 모두 본문에 박힘 | **Resolved** |
| (e) | ρ → 0 null mode 안전장치 (ε floor) | §3 A1 step 3 (line 94: ε = 10·ρ default), A2 step 2 (line 137–138), A3 step 3 (line 170), 그리고 위험 단락 (line 108–114) 에서 ε ∈ {ρ, 10ρ, 100ρ} grid 명시 | **Resolved** |
| (f) | A2 novelty 약함 → ablation 강등 + falsifier #1 sharpness | §3 A2 도입부 (line 122–131) 에서 "ablation, novelty 제한" 명시, falsifier #1 (line 299–313) 이 A1 ↔ A2 를 정량적으로 분리, 표 (line 196–200) Novelty 열에 "**ablation**" 표기 | **Resolved** |

추가로, critique §3 의 부수 발견 두 가지도 plan_v2 에 흡수됐습니다 — Prop
4.2 statement 의 원문 인용은 "다음 iter reading 단계에서 한 줄 보강"
(§2 line 78–81), partial decomp 의 wall-clock 주장은 "flop 기준" 으로
한정 + wall-clock crossover 는 다음 iter 측정 (§3 A3 line 174–179, §5
line 282–286). Multi-class LC 정의 통일은 §5 line 270–280 에 sum-form
Frobenius extension 으로 명시.

## (B) plan_v2 에서 새로 발견된 미세 문제

두 건만, 둘 다 PDF 진행을 막을 정도는 아닙니다.

첫째, A3 의 partial decomp truncation error 분석 (§3 A3 line 181–186)
에서 "λ 가 power-law decay (lrfshap.pdf Remark E.3, p.21: λ_j ≤ C·j^{−α})
하면 i > μr 잔여가 (μr+1)^{−2α} 로 감소" 를 쓰고 있는데, supervised score
mass 의 잔여를 정량화하려면 (u_i⊤Y)² 의 *spectrum 위 분포* 도 같이
가정해야 합니다. 만약 (u_i⊤Y)² 가 i 에 대해 평탄하면 잔여 mass 가 단순히
Σ_{i>μr} (λ_i/(λ_i+ρ))² 로 떨어지지만, label 이 high-frequency mode 와
align 된 adversarial case 면 (μr+1)^{−2α} 의 prefactor 가 ‖Y‖² 와 비례해
커질 수 있습니다. 다음 iter 의 reading 단계에서 한 줄 보강 — "(u_i⊤Y)²
이 i 에 대해 polynomial 이상으로 decay 한다는 추가 가정 하에" — 정도면
충분.

둘째, §6 falsifier #2 (label noise) 의 검증 setup 에서 flip rate 를 5–20%
로만 적었는데 (line 328–330), SST-2 의 baseline test accuracy 가 ~91%
영역이라 5% flip 은 이미 baseline 자체를 흔들 수 있습니다. 다음 iter
실험에서 control 으로 flip rate = 0% 일 때 A1 vs top-r-by-λ 의 ER gap
을 함께 보여서, "noise 가 없을 때도 supervised 가 항상 이긴다" 가 얼마나
robust 한지를 같은 plot 에서 가늠할 수 있게 해주는 게 좋습니다. plan_v2
본문 수정은 불필요, 다음 iter executor 의 sweep 설계 시 반영하면 됩니다.

## (C) 최종 신뢰도 / 남은 risk

plan_v2 는 round 1 의 6 항목을 모두 본문에서 처리했고, 인용 정확성·가정
명시성·알고리즘 안전장치 모두 reading copy 에 들어갈 수준입니다. (B) 의
두 건은 본문 수정 없이 다음 iter reading/실험 단계에서 한 줄씩 보강하면
충분한 미세 사항이라, plan_v2 자체를 다시 돌릴 사유는 아닙니다. 남은
real risk 두 가지 — Prop 4.2 의 원문 norm (operator vs nuclear) 미확인,
S=N → S⊊n transfer 가정의 실험적 검증 — 은 모두 "다음 iter" 로 이미
명시 위임됐고, plan 단계에서 더 잡을 게 없습니다. **PDF 진행 가능**.

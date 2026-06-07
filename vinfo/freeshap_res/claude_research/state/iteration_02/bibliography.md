# Iteration 02 — Bibliography

iter 01 의 bibliography 위에 iter 02 에서 *새로 핵심으로 자리잡은* 두 reference 만 명시 추가. 나머지 4 트랙 (ridge leverage, KTA, eigenlearning 보조, data Shapley) 의 reference 는 iter 01 의 `bibliography.md` 그대로 유효.

## iter 02 에서 핵심으로 자리잡은 reference

[B-BCP21] **Bordelon, Canatar, Pehlevan (2021)** *Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks*. **Nature Communications** 12:2914. https://www.nature.com/articles/s41467-021-23103-1
    iter 02 의 가장 핵심 reference. §3.5 의 *prediction-side vs Shapley-side 분리* 와 §4 mechanism 의 task-model alignment 해석 모두 이 paper 의 framework 위에 서있음. Spearman(λᵢ, cᵢ²) 가 *best predictor* 인 이유 — kernel ridge generalization 의 *learnability* 가 (λ, c²) joint distribution 의 함수라는 정량 결과 — 가 [B-BCP21] 의 main theorem.

[B-Simon23] **Simon, Dickens, Karkada, DeWeese (2023)** *The Eigenlearning Framework: A Conservation Law Perspective on Kernel Ridge Regression*. **TMLR** 2023. arXiv:2110.03922. https://openreview.net/forum?id=FDbQGCAViI (blog 요약: https://jamiesimon.io/blog/eigenlearning/)
    learnability score 의 zero-sum (Σ over modes = n) 분해 — 본 iter 의 LC retention LC(I_LR), LC(I_A1) 와 dual 관계. *동일 양을 두 시각* (selection-side vs generalization-side) 에서 보는 것이 iter 02 의 핵심.

[B-LRFShap25] **LRFShap** (저자 미공개, ICML 2026 workshop submission), `references/lrfshap.pdf`.
    eq. (9) 의 정확한 인용은 p. 7 의 *in-sample predictor gap identity*. iter 02 의 miss_LC_by_LR, miss_FC_by_A1, eq9_gap 정의가 모두 이 식 위에 정의됨.

## iter 01 의 reference (재인용 없이 reference 만)

iter 01 의 4 트랙 — ridge leverage (Alaoui-Mahoney 2015, Bach 2013), kernel-target alignment (Cristianini 2002, Cortes-Mohri-Rostamizadeh 2012, Wang AAAI 2014 / Neurocomputing 2016), eigenlearning (Canatar 2021, Simon 2023), data Shapley + low-rank (FreeShap 2024, TRAK 2023, DataInf 2024) — 의 reference 들은 `state/iteration_01/bibliography.md` 그대로 유효. iter 02 의 식 / framework 는 이 4 트랙 위에 *Shapley-side geometry* layer 를 한 단계 더 얹은 것.

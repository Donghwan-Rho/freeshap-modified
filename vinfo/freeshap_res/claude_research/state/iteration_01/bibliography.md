# Iteration 01 — Bibliography

이번 iter 에서 인용된 모든 외부 자료. URL + 1줄 요약.

## Primary references (이미 references/ 에 있음)

[B1] **Wang et al. 2024a (FreeShap, ICML 2024)**, baseline 논문.
  `references/Freeshap.pdf`. eNTK 기반 retraining-free Shapley value
  approximation. LRFShap 의 직접 baseline.

[B2] **LRFShap (저자 미공개, ICML 2026 workshop submission)**, 우리 논문.
  `references/lrfshap.pdf`. FreeShap 의 ridge inverse 를 K(X_N) 의
  best rank-r eigendecomposition 으로 가속. 핵심 LC/FC 분석 §5.5 (p.7-8).

## Track A — kernel-side leverage scores (label-agnostic)

[B3] **Alaoui & Mahoney, NeurIPS 2015**, *Fast Randomized Kernel Methods
  with Statistical Guarantees*. https://arxiv.org/abs/1411.0306
  Ridge leverage score ℓ_i(ρ) = [K(K+ρI)⁻¹]_{ii} 를 Nyström sub-sampling
  의 sharp 가중으로 사용.

[B4] **Bach, JMLR 2013**, *Sharp analysis of low-rank kernel matrix
  approximations*. http://proceedings.mlr.press/v30/Bach13.pdf
  Effective dimension d_eff(ρ) = tr(K(K+ρI)⁻¹) = Σ λ_i/(λ_i+ρ) 를
  통한 rank-r approximation 의 sharp 분석.

## Track B — kernel-target alignment

[B5] **Cristianini, Shawe-Taylor, Elisseeff, Kandola, NeurIPS 2001**,
  *On Kernel-Target Alignment*. http://papers.neurips.cc/paper/1946
  KTA(K, yy⊤) = ⟨K, yy⊤⟩_F / (‖K‖_F · ‖yy⊤‖_F). 본 라인의 시조.

[B6] **Cortes, Mohri, Rostamizadeh, JMLR 2012**, *Algorithms for Learning
  Kernels Based on Centered Alignment*.
  https://www.jmlr.org/papers/v13/cortes12a.html
  Centered KTA 로 multiple kernel 의 가중을 학습.

[B7] **Wang, Q., Zhang, K., Chen, Z., Wang, D., Jiang, G., Marsic, I.,
  AAAI 2014**, *Improving Semi-Supervised Target Alignment via Label-Aware
  Base Kernels*. https://ojs.aaai.org/index.php/AAAI/article/view/8958
  Label 정보를 주입해 *새로운 base kernel* 의 eigenfunction 을
  extrapolation. v1 의 "Liu et al." 표기는 오기 (critique R1 §1 정정).

[B8] **Wang, Q. et al., Neurocomputing 2016**, *Enhancing semi-supervised
  learning through label-aware base kernels*. https://www.sciencedirect.
  com/science/article/abs/pii/S0925231215010796
  [B7] 의 후속.

## Track C — eigenlearning / spectrum-dependent generalization

[B9] **Canatar, Bordelon, Pehlevan, Nature Comm 2021**, *Spectral Bias
  and Task-Model Alignment Explain Generalization in Kernel Regression
  and Infinitely Wide Neural Networks*. https://www.nature.com/articles/
  s41467-021-23103-1 (arXiv: https://arxiv.org/abs/2006.13198)
  task-model alignment α_i := (u_i⊤Y)²/λ_i. lrfshap §5.5 가 이미 인용.
  본 plan 의 A2 가 α_i 를 selection criterion 으로 옮긴 것.

[B10] **Simon, Dickens, Karkada, DeWeese, TMLR 2023**, *The Eigenlearning
  Framework: A Conservation Law Perspective on Kernel Regression and Wide
  Neural Networks*. arXiv: https://arxiv.org/abs/2110.03922,
  OpenReview: https://openreview.net/forum?id=FDbQGCAViI
  KRR generalization 의 closed-form 이 정확히 mode-별 (λ_i/(λ_i+ρ))² ·
  (u_i⊤y)² 항의 합 — 우리 score s_i 와 본질적으로 같은 양.
  v1 의 "Nature Comm 2023" 표기는 오기 (critique R1 §2 정정).

## Track D — data Shapley + low-rank / influence-based attribution

[B11] **Park, Georgiev, Ilyas, Leclerc, Madry, ICML 2023 (TRAK)**,
  *Attributing Model Behavior at Scale*.
  https://arxiv.org/abs/2303.14186
  Random projection + ensembling 으로 attribution. kernel spectrum 직접
  사용 안 함.

[B12] **Kwon, Wu, Wang, Mohri, ICLR 2024 (DataInf)**, *Efficiently
  Estimating Data Influence Functions*.
  https://arxiv.org/abs/2310.00902
  LoRA Hessian 의 closed-form inverse. influence function 계열 (Shapley
  아님).

## Numerical / spectral methods

[B13] **Halko, Martinsson, Tropp, SIAM Review 2011**, *Finding Structure
  with Randomness: Probabilistic Algorithms for Constructing Approximate
  Matrix Decompositions*. https://arxiv.org/abs/0909.4061
  Randomized SVD 의 표준 reference. A3 의 partial decomp 에 사용.

[B14] **Cortes, Mohri, Talwalkar, AISTATS 2010**, *On the Impact of
  Kernel Approximation on Learning Accuracy*.
  https://proceedings.mlr.press/v9/cortes10a.html
  Kernel approximation 의 predictor gap 이 prediction accuracy 에 미치는
  영향. lrfshap Prop 4.2 의 원조 형태.

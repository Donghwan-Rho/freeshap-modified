# -*- coding: utf-8 -*-
"""Vision Shapley 검증 스모크: 상위 k% vs 랜덤 k% vs 하위 k% subset 정확도.

- 좋은 Shapley: 상위 > 랜덤 > 하위 (또는 최소 상위 > 하위)
- 반대이거나 랜덤과 비슷하면 문제

실행: cd /extdata1/donghwan/freeshap/vinfo && python vision/eval_shapley_smoke.py
"""
import os, sys, pickle
import numpy as np
import torch
from torchvision import datasets

_HERE  = os.path.dirname(os.path.abspath(__file__))
_VINFO = os.path.dirname(_HERE)
os.chdir(_VINFO)   # 상대경로들이 freeshap_res/... 를 찾도록

# ---------- 실험 스펙 (성공한 두 세팅) ----------
NTK_PATH = "./freeshap_res/ntk/cifar10/resnet_seed2024_num200_val100_signFalse.pkl"
CASES = [
    ("inv",   "./freeshap_res/shapley/cifar10/inv/resnet_seed2024_num200_val100_lam1e-03_signFalse_earlystopTrue_tmc50.pkl"),
    ("eigen", "./freeshap_res/shapley/cifar10/eigen/resnet_seed2024_num200_val100_eig10.0_eiglam1e-03_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc50.pkl"),
]
NUM_LABELS = 10
KRR_LAM    = 1e-3   # 평가용 ridge (NTK 스케일 대비 상대적으로 튜닝)
PCTS       = [10, 25, 50, 75, 100]

# ---------- 1) NTK / labels 로드 ----------
ntk_b = pickle.load(open(NTK_PATH, "rb"))
ntk = ntk_b["ntk"]                                # torch [1, 300, 200]
sampled_idx     = np.array(ntk_b["sampled_idx"])
sampled_val_idx = np.array(ntk_b["sampled_val_idx"])
n_train = ntk.shape[2]; n_all = ntk.shape[1]; n_val = n_all - n_train
print(f"[NTK] shape={tuple(ntk.shape)}  n_train={n_train}  n_val={n_val}")
print(f"[NTK] mean|ntk|={ntk.abs().mean():.2f}  max={ntk.max():.2f}")

# CIFAR-10 raw label (transform 없이)
cif_tr = datasets.CIFAR10("./datasets_vision", train=True,  download=False)
cif_va = datasets.CIFAR10("./datasets_vision", train=False, download=False)
y_train = np.array(cif_tr.targets)[sampled_idx]   # [200]
y_val   = np.array(cif_va.targets)[sampled_val_idx]  # [100]

# NTK block: rows [0:n_train] = train, [n_train:] = val
K_tr_full = ntk[0, :n_train, :].numpy().astype(np.float64)     # [200, 200]
K_va_full = ntk[0, n_train:, :].numpy().astype(np.float64)     # [100, 200]

# ---------- 2) 서브셋 KRR 평가 함수 ----------
def krr_eval(idx_subset, lam=KRR_LAM):
    """idx_subset: train 위치(0..n_train-1)의 배열. → val 정확도 반환."""
    K = K_tr_full[np.ix_(idx_subset, idx_subset)]                 # [k, k]
    Kv = K_va_full[:, idx_subset]                                 # [n_val, k]
    Y = np.zeros((len(idx_subset), NUM_LABELS))
    Y[np.arange(len(idx_subset)), y_train[idx_subset]] = 1.0      # one-hot
    # 스케일 무관한 정규화: lam × 대각 평균
    reg = lam * (K.diagonal().mean())
    alpha = np.linalg.solve(K + reg * np.eye(len(idx_subset)), Y) # [k, C]
    preds = Kv @ alpha                                            # [n_val, C]
    return (preds.argmax(axis=1) == y_val).mean()

# 참고 지표
acc_full = krr_eval(np.arange(n_train))
print(f"[baseline] full-train KRR acc = {acc_full:.3f}   (uniform random guess = 0.10)")

# ---------- 3) 각 세팅별 Shapley 순위로 subset acc ----------
rng = np.random.RandomState(2024)
print(f"\n{'mode':<7s} {'pct':>5s} | {'top':>7s} {'random':>7s} {'bottom':>7s}   Δ(top-bot)")
print("-"*60)

results = {}
for mode, sh_path in CASES:
    if not os.path.exists(sh_path):
        print(f"[skip] {mode}: {sh_path} 없음")
        continue
    sh = pickle.load(open(sh_path, "rb"))
    dv = sh["dv_result"]                              # (n_train, 2, n_val)
    # metric idx=1이 accuracy 기여. sample당 shapley = 그 합
    sv = np.array(dv)[:, 1, :].sum(axis=1)            # [n_train]
    print(f"\n[{mode}] shapley stats: mean={sv.mean():+.4f}  std={sv.std():.4f}  "
          f"min={sv.min():+.4f}  max={sv.max():+.4f}")

    order_desc = np.argsort(-sv)                       # 큰 값부터
    order_asc  = np.argsort( sv)                       # 작은 값부터

    results[mode] = []
    for pct in PCTS:
        k = max(1, int(round(n_train * pct / 100)))
        acc_top  = krr_eval(order_desc[:k])
        acc_bot  = krr_eval(order_asc [:k])
        # 3회 랜덤 평균 (안정성)
        acc_rands = [krr_eval(rng.permutation(n_train)[:k]) for _ in range(3)]
        acc_rand  = float(np.mean(acc_rands))
        results[mode].append((pct, acc_top, acc_rand, acc_bot))
        arrow = "✅" if acc_top > acc_rand > acc_bot else ("⚠️" if acc_top > acc_bot else "❌")
        print(f"{mode:<7s} {pct:>4d}% | {acc_top:>7.3f} {acc_rand:>7.3f} {acc_bot:>7.3f}   {acc_top-acc_bot:+.3f} {arrow}")

# ---------- 4) 두 mode의 shapley 값이 서로 얼마나 일치? ----------
if "inv" in results and "eigen" in results:
    sh_inv = pickle.load(open(CASES[0][1], "rb"))["dv_result"][:, 1, :].sum(axis=1)
    sh_eig = pickle.load(open(CASES[1][1], "rb"))["dv_result"][:, 1, :].sum(axis=1)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(sh_inv, sh_eig)
    pearson = np.corrcoef(sh_inv, sh_eig)[0, 1]
    print(f"\n[교차검증] inv vs eigen shapley  Pearson={pearson:+.3f}  Spearman={corr:+.3f}")
    print(f"  → 1에 가까우면 두 모드가 데이터 가치를 유사하게 판단 (건강한 신호)")

print("\n[해석]")
print("  ✅ top > random > bottom : shapley가 데이터 가치를 잘 반영")
print("  ⚠️ top > bottom 만       : 방향성은 맞으나 랜덤보다 그리 낫진 않음")
print("  ❌ top ≤ bottom          : shapley가 사실상 쓸모없거나 반대 방향")

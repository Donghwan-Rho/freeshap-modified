# -*- coding: utf-8 -*-
"""SV-landmark Nystrom 실험 리포트 — inv SV 대비 상관/오차를 rank 축으로 비교.

Nystrom 계열 (전부 pinv 구성, landmark 선택만 다름):
  inv SV 를 오름차순 정렬한 뒤 **분위 위치를 중심으로 연속한 d 개**를 landmark 로 쓰는
  결정적 5 단계 — 순서가 있는 축이다:
      q4(max) > q3(75%) > q2(median) > q1(25%) > q0(min)
  창 시작점 lo = clip(floor(q*n - d/2), 0, n-d). 양 끝은 clip 이 걸려 끝에 붙는다.
  d 가 커지면 인접 분위끼리 창이 겹친다 (n=2000, d=600 이면 q0-q1 이 400 점 공유)
  — 분위 대비가 가장 선명한 것은 낮은 rank 다.
  pinv    : 균등 랜덤 landmark (seed) — 실용 대조군
  lev     : ridge-leverage 비례 랜덤 landmark (seed) — 기본 제외, --with-lev 로 추가
※ 일곱 다 pseudoinverse(pinv) 구성이다 (NystromLevNTKRegression 도 NystromPinvNTKRegression 상속).
  차이는 오직 landmark 선택 규칙.

참조선:
  eigen   : rank-r eigen truncation (K_r = U_r Λ_r U_r^T). **landmark 개념이 없다.**
    Eckart-Young 로 rank-r 최적이라 ||K-K_r|| 가 근사를 유일하게 결정한다(level set 이 싱글턴).
    반대로 Nystrom 은 같은 ||K-L|| 을 갖는 landmark 집합이 무수히 많아 SV 오차가 그 위에서 흩어진다.
    이 대비가 리포트의 요지이므로 eigen 을 검은 점선 참조선으로 함께 그린다 (--no-eigen 로 제외).
    커널 상대오차는 Eckart-Young 으로 sqrt(sum_{j>r} λ_j^2)/||K||_F 로 계산한다.
    ※ landmark 가 없으므로 LM / nonLM 분리 페이지(p4, p5)에서는 제외된다.

landmark 에 들어간 점은 커널 행이 정확히 복원되므로 그 점의 SV 가 맞는 것은 자명하다.
그래서 **landmark 제외(nonLM)** 지표를 함께 그려 순환 논증이 아님을 보인다.

사용:
  python jitter_exp/build_landmark_report.py                       # bert/ag_news/seed2024
  python jitter_exp/build_landmark_report.py --dataset qqp --seeds 2024 2025
  python jitter_exp/build_landmark_report.py --with-lev     # leverage 계열도 함께
  python jitter_exp/build_landmark_report.py --no-eigen     # eigen 참조선 제외
"""
import argparse
import os
import pickle
import re

import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = (1, 5, 10, 15, 20, 25, 30)
VAL = {"sst2": 872, "mrpc": 408, "rte": 277}

# (라벨, out_root, method_dir, 파일명 태그)
#   eigen 은 out_root 가 본진(freeshap_res)이고 파일명 규약도 달라 _one_seed 에서 따로 분기한다.
SPECS = [("q4",    "nys_landmark_q4_res", "nystrom_q4", "nysq4"),
         ("q3",    "nys_landmark_q3_res", "nystrom_q3", "nysq3"),
         ("q2",    "nys_landmark_q2_res", "nystrom_q2", "nysq2"),
         ("q1",    "nys_landmark_q1_res", "nystrom_q1", "nysq1"),
         ("q0",    "nys_landmark_q0_res", "nystrom_q0", "nysq0"),
         ("pinv",   "nys_pinv_res",            "nystrom_pinv",   "nyspinv"),
         ("lev",    "nys_lev_res",             "nystrom_lev",    "nyslev"),
         ("eigen",  "",                        "eigen",          "eig")]

NYS_LABELS = ("q4", "q3", "q2", "q1", "q0", "pinv", "lev")  # landmark 가 있는 계열

# SV 오름차순 정렬 축에서 landmark 창의 중심 분위. entks/ntk_regression.py 의
# SV_LANDMARK_QUANTILES 와 **반드시 같은 값·같은 공식**이어야 한다 (nonLM 지표 재현용).
LM_Q = {"q0": 0.00, "q1": 0.25, "q2": 0.50, "q3": 0.75, "q4": 1.00}

# 8 계열을 색만으로 구분하는 것은 불가능하다(CVD all-pairs 를 통과할 수 없다).
# 그래서 **색 = SV 분포의 어느 쪽 / 선스타일 = 얼마나 극단** 으로 이중 인코딩한다:
#     q4     warm  실선      q3   warm  파선      <- SV 높은 쪽
#     q2     mid   실선                            <- 가운데
#     q1     cool  파선      q0   cool  실선      <- SV 낮은 쪽
#     pinv   ctrl  실선      lev  ctrl  일점쇄선   <- landmark 를 랜덤 추첨하는 실용 규칙 2 개
#     eigen  black 점선                            <- landmark 자체가 없는 참조선
# 이러면 색은 4 개(+검정)뿐이고, 인접한 분위끼리 같은 색을 공유해 "같은 쪽" 이라는
# 순서 구조가 눈에 먼저 들어온다 (부채꼴이 가운데에서 대칭으로 벌어지는 것이 요지).
# OKLCH 격자 탐색 + Machado CVD 시뮬레이션 검증:
#   4색+검정 all-pairs CVD(protan·deutan) ΔE 17.0 / 정상시야 ΔE 21.1 (목표 8 / 15 상회).
_WARM, _MID, _COOL, _CTRL, _REF = "#9D113B", "#40a000", "#00a0c0", "#0040c0", "#141414"
COLORS  = {"q4": _WARM, "q3": _WARM, "q2": _MID, "q1": _COOL, "q0": _COOL,
           "pinv": _CTRL, "lev": _CTRL, "eigen": _REF}
MARKERS = {"q4": "o", "q3": "^", "q2": "s", "q1": "v", "q0": "D",
           "pinv": "P", "lev": "X", "eigen": "*"}
LINESTYLES = {"q3": "--", "q1": "--", "lev": "-.", "eigen": ":"}   # 나머지는 실선
LABELS  = {"q4": "landmark = q4 SV  (max)",
           "q3": "landmark = q3 SV  (75%)",
           "q2": "landmark = q2 SV  (median)",
           "q1": "landmark = q1 SV  (25%)",
           "q0": "landmark = q0 SV  (min)",
           "pinv": "landmark = uniform (pinv)", "lev": "landmark = leverage (lev)",
           "eigen": "eigen rank-r  (no landmarks — reference)"}
ACC_K = 20          # selection 곡선 평균 구간 (1~ACC_K %)


# ------------------------------------------------------------------ 데이터
def sv_of(path):
    with open(path, "rb") as f:
        return np.array(pickle.load(f)["dv_result"])[:, 1, :].sum(1).astype(float)


def sel_mean(path, k=ACC_K):
    """selection predictions txt 의 inv-mode top 곡선 앞 k 개 평균 (val acc x 1e4).

    txt 는 append 모드라 마지막 블록이 현재 SV pkl 과 일치한다.
    """
    try:
        txt = open(path).read()
    except OSError:
        return np.nan
    ms = re.findall(r"inv mode lambda[^\n]*\ntop:\s*\n?\[([^\]]+)\]", txt)
    if not ms:
        return np.nan
    v = [int(x) for x in re.findall(r"\d+", ms[-1])][:k]
    return float(np.mean(v)) if v else np.nan


def kernel_norm(dataset, model, seed, n, v):
    """probe 가 로드할 때와 동일하게 mean 으로 정규화한 train-train 커널."""
    p = f"{ROOT}/freeshap_res/ntk/{dataset}/{model}_seed{seed}_num{n}_val{v}_signFalse.pkl"
    d = pickle.load(open(p, "rb"))
    full = np.asarray(d["ntk"])
    K = full[0, :n, :n].astype(np.float64)
    K = K / float(full.mean())
    return (K + K.T) / 2


_K_CACHE = {}
def kernel_cached(dataset, model, seed, n, v):
    key = (dataset, model, seed, n, v)
    if key not in _K_CACHE:
        _K_CACHE[key] = kernel_norm(dataset, model, seed, n, v)
    return _K_CACHE[key]


def kernel_relerr(K, S, jitter=1e-8):
    """||K - Phi Phi^T||_F / ||K||_F  — compute_kernel_pred_error.py 와 동일 정의.

    pinv 구성에서 Phi Phi^T = C W^+ C^T (W = K[S,S], 양의 스펙트럼만, cutoff=jitter).
    landmark 집합이 커널의 스펙트럼을 얼마나 덮는지의 직접 측정치.
    """
    W = K[np.ix_(S, S)]
    ev, Q = np.linalg.eigh((W + W.T) / 2)
    keep = ev > jitter
    if not keep.any():
        return float("nan")
    C = K[:, S]
    Phi = C @ (Q[:, keep] / np.sqrt(ev[keep]))
    return float(np.linalg.norm(K - Phi @ Phi.T) / np.linalg.norm(K))


_EV_CACHE = {}
def kernel_evals_desc(dataset, model, seed, n, v):
    """K 의 고유값 내림차순 (eigen 참조선의 Eckart-Young 오차용). seed 당 1회 eigh."""
    key = (dataset, model, seed, n, v)
    if key not in _EV_CACHE:
        K = kernel_cached(dataset, model, seed, n, v)
        _EV_CACHE[key] = np.linalg.eigvalsh((K + K.T) / 2)[::-1]
    return _EV_CACHE[key]


def eigen_relerr(ev_desc, K_fro, r):
    """||K - K_r||_F / ||K||_F = sqrt(sum_{j>r} lam_j^2) / ||K||_F  (Eckart-Young).

    Nystrom 의 kernel_relerr 와 **같은 norm·같은 정규화**라 직접 비교 가능하다.
    K_r 은 rank-r 최적 근사이므로 이 값이 그 rank 에서 달성 가능한 하한이다.
    """
    tail = ev_desc[r:]
    return float(np.sqrt((tail ** 2).sum()) / K_fro)


def landmarks(label, d, inv, n, seed, K=None, nys_lam=1e-2):
    """각 방식의 landmark 인덱스를 재구성 (nonLM 지표용). eigen 은 landmark 가 없어 None."""
    if label in LM_Q:
        # NystromSVLandmarkNTKRegression._select_landmarks 와 동일한 창 계산
        o = np.argsort(inv, kind="stable")
        lo = int(min(max(int(np.floor(LM_Q[label] * n - d / 2.0)), 0), n - d))
        return np.sort(o[lo:lo + d])
    if label == "pinv":
        return np.sort(np.random.RandomState(seed).choice(n, size=d, replace=False))
    if label == "lev":
        ev, Q = np.linalg.eigh(K)
        w = np.clip(ev / (ev + nys_lam * n), 0.0, None)
        lev = np.clip((Q ** 2) @ w, 1e-12, None)
        return np.sort(np.random.RandomState(seed).choice(n, size=d, replace=False,
                                                          p=lev / lev.sum()))
    return None


# ------------------------------------------------------------------ 예측(pacc)
#   "full-data 예측" = train n 개 **전부**로 학습한 예측기의 val accuracy.
#   SV 오차가 아니라 예측값 자체를 보는 지표라, landmark 선택이 실제 모델 성능을
#   얼마나 바꾸는지 직접 읽힌다. compute_kernel_pred_error.py 와 같은 정의:
#     inv    : beta = (K + 1e-6 I)^-1 Y                 -> logits = K_TN beta
#     Nystrom: Phi = C Q Sigma^-1/2 (cutoff 1e-8), W = (Phi^T Phi + 1e-2 I)^-1 Phi^T Y
#     eigen  : Phi = U_r sqrt(lam_r), Phi_te = K_TN U_r / sqrt(lam_r), 같은 ridge
#   모델은 로드하지 않는다 (캐시된 NTK + 데이터셋 라벨만).
#   검증: llama/sst2/n2000/seed2024 inv logits vs compute_kernel_pred_error.py 산출 npz
#         -> 상대차 4.5e-05, argmax 100% 일치, accuracy 동일(0.8784).
INVLAM_P, NYSLAM_P, EIGLAM_P, EPS_P = 1e-6, 1e-2, 1e-2, 1e-8
_PRED_CACHE, _EIGH_CACHE = {}, {}


def _labels_of(dataset, tr_idx, val_idx):
    """NTK 캐시의 인덱스로 train/val 라벨을 읽는다 (task_ntk.py 의 split 규약과 동일)."""
    if dataset == "cifar10":
        from torchvision import datasets as _tv
        root = f"{ROOT}/datasets_vision"
        tr = np.array(_tv.CIFAR10(root, train=True, download=False).targets)[tr_idx]
        va = np.array(_tv.CIFAR10(root, train=False, download=False).targets)[val_idx]
        return tr, va
    from datasets import load_dataset
    spec = {"sst2": ("sst2",), "mr": ("rotten_tomatoes",), "ag_news": ("ag_news",),
            "mnli": ("glue", "mnli"), "qqp": ("glue", "qqp"),
            "rte": ("glue", "rte"), "mrpc": ("glue", "mrpc")}[dataset]
    val_split = {"mnli": "validation_matched", "ag_news": "test"}.get(dataset, "validation")
    tr = np.array(load_dataset(*spec, split="train")["label"])[tr_idx]
    va = np.array(load_dataset(*spec, split=val_split)["label"])[val_idx]
    return tr, va


def pred_parts(dataset, model, seed, n, v):
    """(K_TN, y_train, y_val). K_TN 은 kernel_norm 과 **같은 스칼라**로 정규화한다."""
    key = (dataset, model, seed, n, v)
    if key not in _PRED_CACHE:
        p = f"{ROOT}/freeshap_res/ntk/{dataset}/{model}_seed{seed}_num{n}_val{v}_signFalse.pkl"
        d = pickle.load(open(p, "rb"))
        full = np.asarray(d["ntk"])
        K_te = full[0, n:, :n].astype(np.float64) / float(full.mean())
        y_tr, y_val = _labels_of(dataset, np.array(d["sampled_idx"]),
                                 np.array(d["sampled_val_idx"]))
        _PRED_CACHE[key] = (K_te, y_tr, y_val)
    return _PRED_CACHE[key]


def _onehot(y_tr, y_val):
    return np.eye(int(max(y_tr.max(), y_val.max())) + 1)[y_tr]


def acc_inv(K, K_te, y_tr, y_val):
    Y = _onehot(y_tr, y_val)
    beta = np.linalg.solve(K + INVLAM_P * np.eye(len(K)), Y)
    return float(((K_te @ beta).argmax(1) == y_val).mean())


def _acc_from_phi(Phi, Phi_te, Y, y_val, lam):
    W = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ Y)
    return float(((Phi_te @ W).argmax(1) == y_val).mean())


def acc_nystrom(K, K_te, y_tr, y_val, S):
    """landmark 집합 S 로 만든 pinv-Nystrom 예측기의 val accuracy."""
    W = K[np.ix_(S, S)]
    ev, Q = np.linalg.eigh((W + W.T) / 2)
    keep = ev > EPS_P
    if not keep.any():
        return np.nan
    T = Q[:, keep] / np.sqrt(ev[keep])
    return _acc_from_phi(K[:, S] @ T, K_te[:, S] @ T, _onehot(y_tr, y_val), y_val, NYSLAM_P)


def acc_eigen(dataset, model, seed, n, v, K, K_te, y_tr, y_val, r):
    """top-r 고유쌍으로 만든 eigen 예측기의 val accuracy (seed 당 eigh 1회 캐시)."""
    key = (dataset, model, seed, n, v)
    if key not in _EIGH_CACHE:
        ev, U = np.linalg.eigh((K + K.T) / 2)
        _EIGH_CACHE[key] = (ev[::-1], U[:, ::-1])          # 내림차순
    ev, U = _EIGH_CACHE[key]
    lam = np.clip(ev[:r], 0.0, None) + EPS_P
    Ur = U[:, :r]
    return _acc_from_phi(Ur * np.sqrt(lam), K_te @ (Ur / np.sqrt(lam)),
                         _onehot(y_tr, y_val), y_val, EIGLAM_P)


def topk_overlap(a, b, n, frac):
    k = max(1, int(n * frac))
    return len(set(np.argsort(a)[::-1][:k]) & set(np.argsort(b)[::-1][:k])) / k


def _one_seed(dataset, model, seed, n, specs):
    """seed 하나의 (method, rank) -> 지표 dict. 파일 없는 셀은 넣지 않는다."""
    v = VAL.get(dataset, 1000)
    ip = (f"{ROOT}/freeshap_res/shapley/{dataset}/inv/{model}_seed{seed}"
          f"_num{n}_val{v}_lam1e-06_signFalse_earlystopTrue_tmc500.pkl")
    if not os.path.exists(ip):
        return None, np.nan, np.nan, {}
    inv = sv_of(ip)
    inv_sel = sel_mean(f"{ROOT}/freeshap_res/data_selection/{dataset}/inv/predictions/"
                       f"{model}_seed{seed}_num{n}_val{v}_lam1e-06_signFalse_earlystopTrue_tmc500"
                       f"_predictions.txt")
    K = kernel_cached(dataset, model, seed, n, v)   # relerr + lev landmark 재현에 사용
    pp, inv_acc = None, np.nan                      # 예측(pacc) 재료 / inv 기준 정확도
    try:
        pp = pred_parts(dataset, model, seed, n, v)
        inv_acc = acc_inv(K, *pp)
    except Exception as ex:                         # 라벨/캐시가 없으면 pacc 만 건너뛴다
        print(f"  [warn] pacc 재료 없음 ({dataset}/{model}/seed{seed}): "
              f"{type(ex).__name__}: {ex}")
    want_eigen = any(l == "eigen" for l, *_ in specs)
    ev_desc = kernel_evals_desc(dataset, model, seed, n, v) if want_eigen else None
    K_fro = float(np.linalg.norm(K)) if want_eigen else np.nan
    out = {}
    for label, root, md, tag in specs:
        rows = {}
        for R in RANKS:
            if label == "eigen":
                # eigen 은 본진(freeshap_res)에 있고 파일명이 _eig{R}.0_eiglam.._eigeps.. 규약이다
                stem = (f"{model}_seed{seed}_num{n}_val{v}_eig{R}.0_eiglam1e-02_eigeps1e-8"
                        f"_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500")
                p = f"{ROOT}/freeshap_res/shapley/{dataset}/eigen/{stem}.pkl"
                sel_p = (f"{ROOT}/freeshap_res/data_selection/{dataset}/eigen"
                         f"/predictions/{stem}_predictions.txt")
            else:
                stem = (f"{model}_seed{seed}_num{n}_val{v}_{tag}{R}.0_nyslam1e-02_nyseps1e-8"
                        f"_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500")
                # bert/llama 의 pinv·lev 는 jitter_exp/{root} 에 격리돼 있지만,
                # resnet(vision) 은 본진 freeshap_res 아래에 있다. 둘 다 시도한다.
                p = sel_p = None
                for _base in (f"{ROOT}/jitter_exp/{root}", f"{ROOT}/freeshap_res"):
                    _cand = f"{_base}/shapley/{dataset}/{md}/{stem}.pkl"
                    if os.path.exists(_cand):
                        p = _cand
                        sel_p = (f"{_base}/data_selection/{dataset}/{md}"
                                 f"/predictions/{stem}_predictions.txt")
                        break
                if p is None:
                    continue
            if not os.path.exists(p):
                continue
            a = sv_of(p)
            d = int(n * R / 100)
            S = landmarks(label, d, inv, n, seed, K)
            m = np.ones(n, bool)
            if S is not None:
                m[S] = False
            e = a - inv
            _q = np.quantile(inv, np.linspace(0, 1, 11)); _q[-1] += 1e-9
            _dec = np.clip(np.digitize(inv, _q[1:-1]), 0, 9)
            _ord = np.argsort(inv, kind="stable")
            _t100, _b100 = _ord[::-1][:100], _ord[:100]     # inv SV 상위/하위 100 점
            rows[R] = dict(
                err=e, sv=a, inv_ref=inv,
                dec=np.array([e[_dec == i].mean() for i in range(10)]),
                # 커널 근사 상대오차 — Nystrom 은 ||K-CW^+C^T||, eigen 은 Eckart-Young 꼬리
                krel=(eigen_relerr(ev_desc, K_fro, d) if label == "eigen"
                      else (kernel_relerr(K, S) if S is not None else np.nan)),
                spread=float(np.std(a) / np.std(inv)),   # SV 분포 폭 비율 (>1 확대, <1 압축)
                # 최적 아핀 보정(inv ~ c1*approx + c0) 후 남는 오차 = sd(inv)*sqrt(1-r^2).
                # raw RMSE 와의 차이가 '폭/오프셋 왜곡' 이 만든 몫이다.
                cal_rmse=float(np.std(inv) * np.sqrt(max(0.0, 1 - pearsonr(inv, a)[0] ** 2))),
                t100_mae=np.abs(e[_t100]).mean(), t100_bias=e[_t100].mean(),
                b100_mae=np.abs(e[_b100]).mean(), b100_bias=e[_b100].mean(),
                err_lm=e[~m] if S is not None else np.array([]),
                err_nl=e[m],
                lm_mae=(np.abs(e[~m]).mean() if S is not None and (~m).any() else np.nan),
                lm_rmse=(np.sqrt((e[~m] ** 2).mean()) if S is not None and (~m).any() else np.nan),
                sp=spearmanr(inv, a).correlation, pe=pearsonr(inv, a)[0],
                mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                p99=np.percentile(np.abs(e), 99), mx=np.abs(e).max(),
                o1=topk_overlap(inv, a, n, .01), o5=topk_overlap(inv, a, n, .05),
                o20=topk_overlap(inv, a, n, .20),
                sgn=(np.sign(a) == np.sign(inv)).mean(),
                nl_sp=spearmanr(inv[m], a[m]).correlation,
                nl_mae=np.abs(e[m]).mean(),
                nl_rmse=np.sqrt((e[m] ** 2).mean()),
                sel=sel_mean(sel_p),
                # 예측 정확도: 학습셋 전체로 학습한 예측기의 val accuracy
                pacc=(np.nan if pp is None else
                      (acc_eigen(dataset, model, seed, n, v, K, *pp, d) if label == "eigen"
                       else (acc_nystrom(K, *pp, S) if S is not None else np.nan))),
            )
        out[label] = rows
    return inv, inv_sel, inv_acc, out


SCALAR_KEYS = ("sp", "pe", "mae", "rmse", "p99", "mx", "o1", "o5", "o20",
               "sgn", "nl_sp", "nl_mae", "nl_rmse", "lm_mae", "lm_rmse",
               "krel", "spread", "cal_rmse", "t100_mae", "t100_bias", "b100_mae", "b100_bias",
               "sel", "pacc")


def collect(dataset, model, seeds, n, specs):
    """seed 별로 지표를 구한 뒤 **seed 평균**을 낸다 (다른 리포트들과 동일 규약).

    반환: (inv 통계, exact selection 평균, data, 사용된 seed 목록)
      data[label][rank] = {키: 평균, 키+'_sd': 표준편차, 'ns': seed 수,
                           'err': seed pooling 한 점별 오차}
    자료가 없는 (seed, method, rank) 셀은 그 seed 만 빠진다.
    """
    per, invs, isels, iaccs, used = {}, [], [], [], []
    for s in seeds:
        inv, isel, iacc, d = _one_seed(dataset, model, s, n, specs)
        if inv is None or not any(d.get(l) for l, *_ in specs):
            continue
        used.append(s); invs.append(inv)
        if not np.isnan(isel):
            isels.append(isel)
        if not np.isnan(iacc):
            iaccs.append(iacc)
        per[s] = d
    data = {}
    for label, *_ in specs:
        rows = {}
        for R in RANKS:
            vals = [per[s][label][R] for s in used
                    if R in per[s].get(label, {})]
            if not vals:
                continue
            agg = {"ns": len(vals),
                   "err": np.concatenate([x["err"] for x in vals]),
                   "err_lm": np.concatenate([x["err_lm"] for x in vals]),
                   "err_nl": np.concatenate([x["err_nl"] for x in vals]),
                   "inv_ref": np.concatenate([x["inv_ref"] for x in vals]),
                   "sv": np.concatenate([x["sv"] for x in vals]),
                   "dec_mu": np.mean([x["dec"] for x in vals], axis=0),
                   "dec_sd": (np.std([x["dec"] for x in vals], axis=0, ddof=1)
                              if len(vals) > 1 else np.zeros(10))}
            for k in SCALAR_KEYS:
                arr = np.array([x[k] for x in vals], dtype=float)
                arr = arr[np.isfinite(arr)]
                agg[k] = float(arr.mean()) if arr.size else np.nan
                agg[k + "_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            rows[R] = agg
        data[label] = rows
    inv_all = np.concatenate(invs) if invs else np.array([0.0])
    return (inv_all, (float(np.mean(isels)) if isels else np.nan),
            (float(np.mean(iaccs)) if iaccs else np.nan), data, used)


# ------------------------------------------------------------------ 그림
def panel(ax, data, key, title, ylabel, better, specs, hline=None):
    for label, *_ in specs:
        rs = sorted(data[label])
        if not rs:
            continue
        mu = np.array([data[label][r][key] for r in rs], dtype=float)
        sd = np.array([data[label][r].get(key + "_sd", 0.0) for r in rs], dtype=float)
        if np.any(sd > 0):      # seed 간 표준편차 음영
            ax.fill_between(rs, mu - sd, mu + sd, color=COLORS[label], alpha=.13, lw=0)
        ax.plot(rs, mu, color=COLORS[label], ls=LINESTYLES.get(label, "-"),
                marker=MARKERS[label], markersize=(9 if label == "eigen" else 6),
                lw=2.0, label=LABELS[label],
                markeredgecolor="white", markeredgewidth=.8)
    if hline is not None:
        ax.axhline(hline, color="0.45", ls="--", lw=1.3, zorder=1)
    ax.set_title(f"{title}  ({better})", fontsize=12)
    ax.set_xlabel("Nystrom rank (% of n)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(list(RANKS))
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def build(dataset, model, seeds, n, specs, out_pdf):
    inv, inv_sel, inv_acc, data, used = collect(dataset, model, seeds, n, specs)
    if not used:
        print(f"[skip] {dataset}: 자료 없음"); return None
    sd_txt = "seeds " + ",".join(map(str, used)) + f" (n={len(used)}, mean over seeds)"
    has_eigen = any(l == "eigen" and data.get(l) for l, *_ in specs)
    hdr = (f"{model} / {dataset} / {sd_txt} / N={n}  —  "
           f"Nystrom landmark choice (pinv construction throughout)"
           + ("   |   dotted black = eigen rank-r reference (no landmarks)" if has_eigen else ""))

    with PdfPages(out_pdf) as pdf:
        # ---------- p1: 상관 · 오차 · top-k 겹침 ----------
        fig, axes = plt.subplots(2, 5, figsize=(27, 10))
        panel(axes[0][0], data, "sp",  "Spearman vs inv SV",  "Spearman ρ", "higher better", specs)
        panel(axes[0][1], data, "pe",  "Pearson vs inv SV",   "Pearson r",  "higher better", specs)
        panel(axes[0][2], data, "o1",  "Top-1% set overlap",     "overlap",    "higher better", specs)
        panel(axes[0][3], data, "o20", "Top-20% set overlap",    "overlap",    "higher better", specs)
        panel(axes[0][4], data, "sgn", "Sign agreement",      "agreement",  "higher better", specs)
        # 커널 근사 오차 = landmark 가 커널 스펙트럼을 얼마나 덮는가 (SV 오차의 상류 원인)
        panel(axes[1][0], data, "krel", "Kernel approx. rel. error\n||K-PhiPhi^T||_F / ||K||_F",
              "relative error", "lower better", specs)
        panel(axes[1][1], data, "mae", "MAE = mean|e|",       "MAE",        "lower better", specs)
        panel(axes[1][2], data, "rmse","RMSE = sqrt(mean e^2)", "RMSE",     "lower better", specs)
        panel(axes[1][3], data, "p99", "99th pct |e|",        "p99 |e|",    "lower better", specs)
        panel(axes[1][4], data, "spread", "SV spread ratio  sd(approx)/sd(inv)",
              "spread ratio", ">1 dilated, <1 compressed", specs, hline=1.0)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(4, len(specs)), fontsize=12,
                   bbox_to_anchor=(0.5, -0.03), frameon=False)
        fig.suptitle(hdr, fontsize=15, fontweight="bold")
        # 오차 정의를 그림 안에 명시 (p2 의 landmark-excluded 판과 구분되도록)
        fig.text(0.5, 0.945,
                 f"error of point i:  e_i = sv_approx(i) - sv_inv(i)      "
                 f"MAE = (1/N) sum |e_i|      RMSE = sqrt( (1/N) sum e_i^2 )   "
                 f"[RMSE >= MAE; gap grows when a few points dominate]\n"
                 f"ALL {n} training points on this page (landmarks included)   |   "
                 f"lines = mean over {len(used)} seed(s), shaded band = +/-1 sd across seeds"
                 + ("\nkernel rel. error: Nystrom = ||K - C W^+ C^T||_F / ||K||_F ; "
                    "eigen = ||K - K_r||_F / ||K||_F = sqrt(sum_{j>r} lam_j^2)/||K||_F "
                    "(Eckart-Young -- the optimum at that rank)" if has_eigen else ""),
                 ha="center", va="top", fontsize=10.5, color="0.25")
        fig.tight_layout(rect=(0, 0.04, 1, 0.925))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---------- p2: 부호 있는 오차의 구조 — inv SV 십분위별 편향 + selection ----------
        #   landmark 집합에서 보이는 "top 은 음수 / bottom 은 양수" 패턴이
        #   landmark 라서인지, 그 점들의 SV 가 극단이라서인지 가르는 진단.
        #   방식과 무관하게 우하향이면 저rank 근사의 일반적 수축(shrinkage) 이다.
        fig, axes = plt.subplots(1, 5, figsize=(30, 5.6))
        for ax, R in zip(axes[:2], (RANKS[0], RANKS[-1])):
            for label, *_ in specs:
                if R not in data[label]:
                    continue
                mu, sd = data[label][R]["dec_mu"], data[label][R]["dec_sd"]
                x10 = np.arange(1, 11)
                if np.any(sd > 0):     # seed 별로 구한 뒤 평균 -> 음영은 seed 간 sd
                    ax.fill_between(x10, mu - sd, mu + sd, color=COLORS[label], alpha=.13, lw=0)
                ax.plot(x10, mu,
                        color=COLORS[label], marker=MARKERS[label], markersize=6, lw=2.0,
                        label=LABELS[label], markeredgecolor="white", markeredgewidth=.8)
            ax.axhline(0, color="black", ls=":", lw=1.1)
            ax.set_title(f"Signed error by inv-SV decile — rank {R}%", fontsize=12)
            ax.set_xlabel("inv SV decile  (1 = lowest SV, 10 = highest)", fontsize=10)
            ax.set_ylabel("mean (approx - inv)", fontsize=10)
            ax.set_xticks(range(1, 11)); ax.grid(alpha=.18); ax.tick_params(labelsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        panel(axes[2], data, "cal_rmse", "RMSE after affine calibration",
              "calibrated RMSE", "removes spread/offset", specs)
        panel(axes[3], data, "sel", f"Selection acc, mean over top 1-{ACC_K}%",
              "val acc (x1e4)", "higher better", specs,
              hline=None if np.isnan(inv_sel) else inv_sel)
        if not np.isnan(inv_sel):
            axes[3].annotate(f"exact (inv) = {inv_sel:.0f}", xy=(RANKS[-1], inv_sel),
                             xytext=(-6, 6), textcoords="offset points",
                             ha="right", fontsize=10, color="0.35")
        # 예측 정확도: SV 오차가 아니라 "그 근사로 학습한 모델이 얼마나 맞히나"
        panel(axes[4], data, "pacc", "Val accuracy of full-data predictor",
              "val accuracy", "higher better", specs,
              hline=None if np.isnan(inv_acc) else inv_acc)
        if not np.isnan(inv_acc):
            axes[4].annotate(f"exact (inv) = {inv_acc:.4f}", xy=(RANKS[-1], inv_acc),
                             xytext=(-6, 6), textcoords="offset points",
                             ha="right", fontsize=10, color="0.35")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(4, len(specs)), fontsize=12,
                   bbox_to_anchor=(0.5, -0.03), frameon=False)
        fig.suptitle("Where the error sits: signed bias across the inv-SV spectrum",
                     fontsize=14, fontweight="bold")
        fig.text(0.5, 0.905,
                 "A downward slope = shrinkage (high-SV points under-estimated, low-SV over-estimated), "
                 "the generic low-rank effect.\n"
                 "Deviations from the uniform(pinv) curve are what landmark CHOICE adds on top of it.   "
                 "Calibrated RMSE = sd(inv)*sqrt(1-r^2): what is left once the spread/offset distortion "
                 "is removed.\n"
                 "Val accuracy = the predictor trained on ALL n points with that approximation "
                 "(not an SV error): inv = (K+1e-6 I)^-1 Y, Nystrom = C W^+ C^T features, eigen = top-r "
                 "pairs; ridge 1e-2.",
                 ha="center", va="top", fontsize=10, color="0.25")
        fig.tight_layout(rect=(0, 0.06, 1, 0.88))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---------- p3~p5: 점별 SV 오차 분포 (all / landmark / non-landmark) ----------
        #   d = sv_approx - sv_inv. inv 는 오차 0 이므로 x=0 검정 점선.
        #   세 판 모두 **같은 XLIM** (전체 오차의 99 분위) 을 써서 직접 비교 가능하게 한다.
        alld = np.concatenate([data[l][r]["err"] for l, *_ in specs for r in data[l]])
        XLIM = float(np.percentile(np.abs(alld), 99)) or 1.0
        bins = np.linspace(-XLIM, XLIM, 61)

        def errpage(key, what, note, specs=specs):
            fig, axes = plt.subplots(2, 4, figsize=(20, 8.4))
            for k, R in enumerate(RANKS):
                ax = axes[k // 4][k % 4]
                ax.text(.02, .97, f"inv sv: mu={inv.mean():.3g} sd={inv.std():.3g} "
                                  f"[{inv.min():.3g}, {inv.max():.3g}]",
                        transform=ax.transAxes, va="top", fontsize=6.5, color="black")
                y, dy, clipped, tot = .91, .062, 0, 0
                for label, *_ in specs:
                    if R not in data[label]:
                        continue
                    e = data[label][R][key]
                    if e.size == 0:
                        continue
                    clipped += int((np.abs(e) > XLIM).sum()); tot += e.size
                    ax.hist(np.clip(e, -XLIM, XLIM), bins=bins, histtype="step",
                            color=COLORS[label], lw=1.7, density=True)
                    ax.text(.02, y, f"{label}: n={e.size} | MAE={np.abs(e).mean():.3g} "
                                    f"RMSE={np.sqrt((e**2).mean()):.3g} "
                                    f"p99={np.percentile(np.abs(e),99):.3g} "
                                    f"max={np.abs(e).max():.3g}",
                            transform=ax.transAxes, va="top", fontsize=6.5, color=COLORS[label])
                    y -= dy
                ax.axvline(0, color="black", ls=":", lw=1.1)
                ax.set_title(f"rank {R}%  (d={int(n*R/100)})", fontsize=11)
                ax.set_xlabel("SV error  (approx - inv)", fontsize=9)
                ax.set_ylabel("density", fontsize=9)
                ax.set_xlim(-XLIM, XLIM); ax.tick_params(labelsize=8)
                ax.grid(alpha=.18)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if tot:
                    ax.text(.98, .02, f"clipped {clipped}/{tot}", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=6, color="0.45")
            axes[1][3].axis("off")
            hh = [plt.Line2D([], [], color=COLORS[l], lw=2.2, ls=LINESTYLES.get(l, "-"),
                             marker=MARKERS[l], markersize=7, markeredgecolor="white",
                             label=LABELS[l]) for l, *_ in specs]
            axes[1][3].legend(handles=hh, loc="center", fontsize=10, frameon=False)
            fig.suptitle(f"Per-point Shapley-value error — {what}   "
                         f"(shared x-limit = 99th pct of ALL errors = {XLIM:.3g})",
                         fontsize=14, fontweight="bold")
            fig.text(0.5, 0.945, note, ha="center", va="top", fontsize=10, color="0.25")
            fig.tight_layout(rect=(0, 0, 1, 0.915))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        errpage("err", "ALL points",
                f"All {n} training points, seeds pooled.")
        # eigen 은 landmark 가 없으므로 아래 두 페이지에서 제외한다 (분할 자체가 정의되지 않는다).
        nys_specs = [s for s in specs if s[0] in NYS_LABELS]
        errpage("err_lm", "LANDMARK points only",
                "The d points used as Nystrom landmarks. Their kernel ROWS are reconstructed exactly, "
                "but their Shapley VALUES are not:\n"
                "a point's SV depends on the whole kernel (every coalition involves other points). "
                "This page shows how much of an advantage, if any, being a landmark actually buys.  "
                "[eigen is omitted: it has no landmark set.]",
                specs=nys_specs)
        errpage("err_nl", "NON-LANDMARK points only",
                f"The {n} - d points NOT used as landmarks -- the complement of the previous page.  "
                "[eigen is omitted: it has no landmark set.]",
                specs=nys_specs)

        # ---------- p6: inv SV 상·하위 100 점의 오차 (rank 축) ----------
        #   극단 SV 점들이 rank 를 늘릴 때 좋아지는지/나빠지는지 직접 본다.
        #   ※ top 방식은 rank>=5% 부터 상위 100 점이 통째로 landmark 에 포함되고,
        #     bottom 방식은 하위 100 점이 포함된다 (d = rank% x n, rank 5% -> d=100).
        fig, axes = plt.subplots(1, 4, figsize=(24, 5.6))
        panel(axes[0], data, "t100_mae",  "MAE on TOP-100 by inv SV",    "MAE",  "lower better", specs)
        panel(axes[1], data, "b100_mae",  "MAE on BOTTOM-100 by inv SV", "MAE",  "lower better", specs)
        panel(axes[2], data, "t100_bias", "Signed bias on TOP-100",    "mean (approx - inv)", "0 is unbiased", specs)
        panel(axes[3], data, "b100_bias", "Signed bias on BOTTOM-100", "mean (approx - inv)", "0 is unbiased", specs)
        for ax in axes[2:]:
            ax.axhline(0, color="black", ls=":", lw=1.1)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(4, len(specs)), fontsize=12,
                   bbox_to_anchor=(0.5, -0.03), frameon=False)
        fig.suptitle("Extreme points: SV error on the 100 highest / 100 lowest inv-SV points",
                     fontsize=14, fontweight="bold")
        fig.text(0.5, 0.905,
                 "100 points each (not 100%). Lines = mean over seeds, band = +/-1 sd across seeds.\n"
                 "Note: for 'q4' the TOP-100 are inside the landmark set once rank >= 5% (d = 100); "
                 "same for 'q0' and the BOTTOM-100.",
                 ha="center", va="top", fontsize=10, color="0.25")
        fig.tight_layout(rect=(0, 0.06, 1, 0.88))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---------- p7: 수치 표 ----------
        cols = ["method", "rank", "d", "#seed", "Spearman", "Pearson", "MAE", "RMSE",
                "p99|e|", "top1%", "top5%", "top20%", "sign", "nonLM ρ", "nonLM MAE",
                "nonLM RMSE", "LM MAE", "LM RMSE",
                "kernel relerr", "spread", "calRMSE", "top100 MAE", "top100 bias", "bot100 MAE", "bot100 bias", "sel", "pred acc"]
        rows = []
        for label, *_ in specs:
            for R in sorted(data[label]):
                x = data[label][R]
                def _f(k, nd=4):     # 평균 (seed 2개 이상이면 ±sd 병기)
                    if np.isnan(x[k]):
                        return "--"
                    return (f"{x[k]:.{nd}f}" if x["ns"] < 2
                            else f"{x[k]:.{nd}f}±{x[k+'_sd']:.{nd}f}")
                rows.append([label, f"{R}%", int(n * R / 100), x["ns"],
                             _f("sp"), _f("pe"), _f("mae"), _f("rmse"), _f("p99"),
                             _f("o1", 3), _f("o5", 3), _f("o20", 3), _f("sgn", 3),
                             _f("nl_sp"), _f("nl_mae"), _f("nl_rmse"),
                             _f("lm_mae"), _f("lm_rmse"),
                             _f("krel"), _f("spread", 3), _f("cal_rmse"), _f("t100_mae", 3), _f("t100_bias", 3),
                             _f("b100_mae", 3), _f("b100_bias", 3),
                             "--" if np.isnan(x["sel"]) else f"{x['sel']:.0f}",
                             _f("pacc", 4)])
        fig, ax = plt.subplots(figsize=(22, 1.1 + 0.30 * len(rows)))
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.35)
        for j in range(len(cols)):
            t[0, j].set_facecolor("#eeeeee"); t[0, j].set_text_props(weight="bold")
        for i, r in enumerate(rows, start=1):
            t[i, 0].set_text_props(color=COLORS[r[0]], weight="bold")
        ax.set_title(hdr + f"\ninv SV: min {inv.min():.3f} / median {np.median(inv):.3f} / "
                           f"max {inv.max():.3f} / std {inv.std():.3f}"
                           + ("" if np.isnan(inv_sel) else f"   |   exact selection = {inv_sel:.0f}"),
                     fontsize=13, fontweight="bold", pad=16)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"[saved] {out_pdf}")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ag_news")
    ap.add_argument("--model", default="bert")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026],
                    help="이 seed 들의 지표를 평균한다 (자료 없는 seed 는 자동 제외)")
    ap.add_argument("--num_train_dp", type=int, default=2000)
    ap.add_argument("--with-lev", action="store_true",
                    help="leverage 계열도 함께 표시. 기본은 제외 — 이 실험의 대조군은\n"
                         "'SV 정보를 쓴 landmark(top/bottom)' vs '균등 랜덤(pinv)' 이다.")
    ap.add_argument("--no-eigen", action="store_true",
                    help="eigen rank-r 참조선을 빼고 Nystrom 계열만 그린다 (기본은 포함).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    specs = [s for s in SPECS
             if (a.with_lev or s[0] != "lev") and not (a.no_eigen and s[0] == "eigen")]
    out = a.out or (f"{ROOT}/jitter_exp/nys_landmark_q4_res/"
                    f"report_landmark_{a.model}_{a.dataset}_n{a.num_train_dp}.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build(a.dataset, a.model, a.seeds, a.num_train_dp, specs, out)

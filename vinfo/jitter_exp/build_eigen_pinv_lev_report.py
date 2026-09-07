# -*- coding: utf-8 -*-
"""3-way rank 비교 리포트: Eigen(파랑) vs Nystrom-pinv(빨강) vs Nystrom-lev(초록).

(λ=1e-2, ε=1e-8) 고정, rank ∈ {1,5,10,15,20,25,30}% 스윕. eps×λ grid 아님.

  p1: rank 히트맵 (x=rank, 1×7 스트립) — 지표 6행 × method 3열
      [SV Spearman / SV Pearson / Selection AUC(1-100%) / Selection AUC(1-K%) /
       Top-removal AUC(낮을수록 좋음) / Bottom-removal AUC(높을수록 좋음)]
  p2: Fidelity/AUC vs rank 선그래프 3×2 (+ inv 기준선)
  p3: 상대오차 rank 히트맵 — [Kernel approx / Kernel val-예측 / SV / SV 정규화] × method
  p4: 상대오차 vs rank 선그래프 2×2 (y=log) — rel_error 리포트의 rank 정보 통합판
      (커널/예측 오차 npz 는 없으면 자동 계산; lev 포함)
  p5: per-point max SV 오차 vs 이론 상한 (lrfshap Prop 0.2 / Cor 0.1) — 같은 단위 직접 비교.
      점선 = Eigen bound(Eckart–Young), uniform-Nyström bound(δ=0.1). ρ=λ_fix 해석.
  p6: rank별 SV 점별 오차 분포 (2×4) — d=sv_approx−sv_inv 히스토그램, inv=검정 점선(x=0),
      seed pooling, 클리핑 한계는 |오차| 99% 분위수 자동.

데이터 탐색: 각 method 파일을 모든 root(nys_lev_res / nys_pinv_res / jitter res /
freeshap_res)에서 fallback 탐색 → eigen rank-sweep(freeshap_res)도 자동으로 찾음.
없는 셀은 빈칸/곡선 결손 (실험 쌓이면 재실행만 하면 채워짐).

사용:
  python jitter_exp/build_eigen_pinv_lev_report.py --dataset qqp
  -> ./jitter_exp/report_eigen_pinv_lev_qqp.pdf
"""
import os, sys, glob, re, copy, argparse, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pinv_selection_report import (  # noqa: E402  (폰트 설정도 이 import 에서 적용됨)
    draw_heat, _fmt_eps, _lam_tag, _mean_std_n, load_sv, parse_pred_inv, align, _rank,
)
from build_pinv_removal_report import parse_removal  # noqa: E402

M3 = {
    "eigen":        dict(mdir="eigen",        tag="eig",     lam="eiglam", eps="nyseps_NA",
                         kor="Eigen",        col="#1f77b4"),
    "nystrom_pinv": dict(mdir="nystrom_pinv", tag="nyspinv", lam="nyslam", eps="nyseps",
                         kor="Nyström-pinv", col="#d62728"),
    "nystrom_lev":  dict(mdir="nystrom_lev",  tag="nyslev",  lam="nyslam", eps="nyseps",
                         kor="Nyström-lev",  col="#2ca02c"),
}
M3["eigen"]["eps"] = "eigeps"
ROOTS_ALL = ["./jitter_exp/nys_lev_res", "./jitter_exp/nys_pinv_res",
             "./jitter_exp/res", "./freeshap_res"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--methods", type=str, nargs="+",
                   default=["eigen", "nystrom_lev", "nystrom_pinv"])
    p.add_argument("--model", type=str, default="bert")
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--val", type=int, default=None, help="없으면 NTK 파일명에서 자동 감지")
    p.add_argument("--tmc", type=int, default=500)
    p.add_argument("--invlam", type=str, default="1e-06")
    p.add_argument("--ranks", type=float, nargs="+", default=[1, 5, 10, 15, 20, 25, 30])
    p.add_argument("--lam_fix", type=float, default=1e-2)
    p.add_argument("--eps_fix", type=float, default=1e-8)
    p.add_argument("--sel_k", type=int, default=20, help="selection 저-k AUC 상한 (%%)")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


# ============ 경로 (모든 root fallback 탐색) ============
def _stem(a, m, r, seed):
    d = M3[m]
    return (f"{a.model}_seed{seed}_num{a.num_train}_val{a.val}"
            f"_{d['tag']}{float(r)}_{d['lam']}{_lam_tag(a.lam_fix)}_{d['eps']}{_fmt_eps(a.eps_fix)}"
            f"_invlam{a.invlam}_cholesky_float32_signFalse_earlystopTrue_tmc{a.tmc}")

def find3(a, m, r, seed, kind):
    d = M3[m]; st = _stem(a, m, r, seed)
    sub = {"sv":  f"shapley/{a.dataset}/{d['mdir']}/{st}.pkl",
           "sel": f"data_selection/{a.dataset}/{d['mdir']}/predictions/{st}_predictions.txt",
           "rem": f"data_removing/{a.dataset}/{d['mdir']}/predictions/{st}_predictions.txt"}[kind]
    for root in ROOTS_ALL:
        p = os.path.join(root, sub)
        if os.path.exists(p):
            return p
    return os.path.join(ROOTS_ALL[0], sub)

def _inv_stem(a, seed):
    return (f"{a.model}_seed{seed}_num{a.num_train}_val{a.val}"
            f"_lam{a.invlam}_signFalse_earlystopTrue_tmc{a.tmc}")

def inv_path(a, seed, kind):
    st = _inv_stem(a, seed)
    sub = {"sv":  f"shapley/{a.dataset}/inv/{st}.pkl",
           "sel": f"data_selection/{a.dataset}/inv/predictions/{st}_predictions.txt",
           "rem": f"data_removing/{a.dataset}/inv/predictions/{st}_predictions.txt"}[kind]
    return os.path.join("./freeshap_res", sub)


# ============ 지표 (seed 집계: mean,std,n) ============
def fid_seeds(a, m, r, INV, which):
    vals = []
    for s in a.seeds:
        asv, asi = load_sv(find3(a, m, r, s, "sv"))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        if len(asv) == len(isv) and np.array_equal(asi, isi):
            av, bv = asv, isv
        else:
            av, bv = align(asv, asi, isv, isi)
        if len(av) < 10: continue
        if which == "sp":
            vals.append(float(np.corrcoef(_rank(av), _rank(bv))[0, 1]))
        else:
            vals.append(float(np.corrcoef(av, bv)[0, 1]))
    return _mean_std_n(vals)

def sel_auc_seeds(a, m, r, cap):
    vals = []
    for s in a.seeds:
        top, _ = parse_pred_inv(find3(a, m, r, s, "sel"))
        if top is None or not len(top): continue
        vals.append(float(np.mean(top[:cap])))
    return _mean_std_n(vals)

def rem_auc_seeds(a, m, r, strat):
    vals = []
    for s in a.seeds:
        d = parse_removal(find3(a, m, r, s, "rem"))
        if strat in d and len(d[strat]):
            vals.append(float(np.mean(d[strat])))
    return _mean_std_n(vals)

# ---- 기준선 (inv / random; rank 무관) ----
def base_sel(a, cap, which):
    vals = []
    for s in a.seeds:
        top, rnd = parse_pred_inv(inv_path(a, s, "sel"))
        arr = top if which == "inv" else rnd
        if arr is None or not len(arr): continue
        vals.append(float(np.mean(arr[:cap])))
    return _mean_std_n(vals)

def base_rem(a, strat):
    vals = []
    for s in a.seeds:
        d = parse_removal(inv_path(a, s, "rem"))
        if strat in d and len(d[strat]):
            vals.append(float(np.mean(d[strat])))
    return _mean_std_n(vals)


# ============ 상대오차 (rel_error 리포트 rank 정보 통합) ============
M3ROOT = {"eigen": "./jitter_exp/res", "nystrom_pinv": "./jitter_exp/nys_pinv_res",
          "nystrom_lev": "./jitter_exp/nys_lev_res"}

def kp_npz(a, m, r, seed):
    """kernel_prediction npz 경로 (compute_kernel_pred_error.py 생성물)."""
    if m == "inv":
        return (f"./freeshap_res/kernel_prediction/{a.dataset}/inv/predictions/"
                f"{_inv_stem(a, seed)}_predictions.npz")
    return (f"{M3ROOT[m]}/kernel_prediction/{a.dataset}/{M3[m]['mdir']}/predictions/"
            f"{_stem(a, m, r, seed)}_predictions.npz")

def ensure_npz(a):
    """npz 없으면 compute(예측+kernel_relerr), 필드만 없으면 backfill (lev 포함)."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_kernel_pred_error.py")
    common = ["--dataset", a.dataset, "--model", a.model,
              "--ranks", *[f"{r:g}" for r in a.ranks],
              "--lams", str(a.lam_fix), "--epss", str(a.eps_fix),
              "--methods", *a.methods]
    def _paths(s):
        yield kp_npz(a, "inv", None, s)
        for m in a.methods:
            for r in a.ranks:
                yield kp_npz(a, m, r, s)
    missing = [s for s in a.seeds if any(not os.path.exists(p) for p in _paths(s))]
    if missing and a.model == "llama":   # llama 모델 로드(대용량) 유발 방지 — llama 서버에서 수동 실행
        print(f"[auto] kernel_prediction npz 미비 seed {missing} (llama 는 자동 계산 안 함 — "
              "llama 서버에서 compute_kernel_pred_error.py --model llama 실행)")
        missing = []
    if missing:
        print(f"[auto] kernel_prediction npz 미비 seed {missing} -> compute 실행...")
        rr = subprocess.run([sys.executable, script, *common, "--seeds", *map(str, missing)])
        if rr.returncode != 0:
            print("[auto] compute 실패 - 있는 데이터만으로 진행")
    need_bf = [s for s in a.seeds
               if any(os.path.exists(p) and "kernel_relerr" not in np.load(p).files
                      for p in _paths(s))]
    if need_bf:
        print(f"[auto] kernel_relerr 필드 미비 seed {need_bf} -> backfill (NTK 만 필요)...")
        rr = subprocess.run([sys.executable, script, *common, "--seeds", *map(str, need_bf),
                             "--backfill_kernel"])
        if rr.returncode != 0:
            print("[auto] backfill 실패 - 있는 데이터만으로 진행")

_NPZ_CACHE = {}
def _load_npz(p):
    if p in _NPZ_CACHE: return _NPZ_CACHE[p]
    out = None
    if os.path.exists(p):
        try: out = dict(np.load(p))
        except Exception: out = None
    _NPZ_CACHE[p] = out; return out

def kernel_err_seeds(a, m, r):
    vals = []
    for s in a.seeds:
        d = _load_npz(kp_npz(a, m, r, s))
        if d is None or "kernel_relerr" not in d: continue
        vals.append(float(d["kernel_relerr"]))
    return _mean_std_n(vals)

def kpred_err_seeds(a, m, r):
    vals = []
    for s in a.seeds:
        di = _load_npz(kp_npz(a, "inv", None, s)); da = _load_npz(kp_npz(a, m, r, s))
        if di is None or da is None: continue
        Li, La = di.get("logits"), da.get("logits")
        if Li is None or La is None or Li.shape != La.shape: continue
        den = float(np.linalg.norm(Li))
        if den == 0: continue
        vals.append(float(np.linalg.norm(La - Li) / den))
    return _mean_std_n(vals)

def sv_err_seeds(a, m, r, INV, normed=False):
    vals = []
    for s in a.seeds:
        asv, asi = load_sv(find3(a, m, r, s, "sv"))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        if len(asv) == len(isv) and np.array_equal(asi, isi):
            av, bv = asv, isv
        else:
            av, bv = align(asv, asi, isv, isi)
        if len(av) < 10: continue
        if normed:
            na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
            if na == 0 or nb == 0: continue
            vals.append(float(np.linalg.norm(av / na - bv / nb)))
        else:
            den = float(np.linalg.norm(bv))
            if den == 0: continue
            vals.append(float(np.linalg.norm(av - bv) / den))
    return _mean_std_n(vals)

def _err_metrics(a):
    return [
        ("Kernel approx 상대오차 ||K-PhiPhi^T||_F/||K||_F (train kernel)",
         lambda m, r, INV: kernel_err_seeds(a, m, r), "RdYlGn_r", None, None),
        ("Kernel val-예측 상대오차 ||K(D)-K_a(D)||_F/||K(D)||_F (D=val)",
         lambda m, r, INV: kpred_err_seeds(a, m, r), "RdYlGn_r", None, None),
        ("SV 상대오차 ||sv_a-sv_inv||_2/||sv_inv||_2",
         lambda m, r, INV: sv_err_seeds(a, m, r, INV, False), "RdYlGn_r", None, None),
        ("SV 정규화 상대오차 (스케일 제거; =sqrt(2(1-cos)))",
         lambda m, r, INV: sv_err_seeds(a, m, r, INV, True), "RdYlGn_r", None, None),
    ]

def page_err_curves(pdf, a, INV, ERRM):
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"[{a.dataset}] 상대오차 vs rank ({fixed} 고정, y=log scale, mean±std band, "
                 f"seeds={a.seeds}) — 낮을수록 inv 에 가까움", fontsize=12.5, fontweight="bold")
    ranks = np.array(a.ranks, dtype=float)
    for ax, (ttl, fmaker, _c, _v, _b) in zip(axes.flat, ERRM):
        for m in a.methods:
            mm, ss = rank_row(a, (lambda r, INV, _m=m: fmaker(_m, r, INV)), INV)
            ok = np.isfinite(mm)
            if not ok.any(): continue
            col = M3[m]["col"]
            ax.plot(ranks[ok], mm[ok], "o-", color=col, lw=1.9, ms=5, label=M3[m]["kor"])
            lo = np.clip(mm[ok] - ss[ok], 1e-12, None)
            ax.fill_between(ranks[ok], lo, mm[ok] + ss[ok], color=col, alpha=.15)
        ax.set_yscale("log")
        ax.set_title(ttl.split(" ||")[0], fontsize=11)
        ax.set_xlabel("rank/d (%)", fontsize=10); ax.set_ylabel("relative error (log)", fontsize=10)
        ax.set_xticks(ranks); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, .93])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 이론 상한 (lrfshap.pdf Prop 0.2 / Cor 0.1) ============
_SPEC_CACHE = {}
def _spectrum(a, seed):
    """정규화된 train kernel 의 (고유값 내림차순, κ²=max K_ii, N). NTK 캐시에서 계산."""
    key = (a.dataset, seed)
    if key in _SPEC_CACHE: return _SPEC_CACHE[key]
    import pickle, torch
    p = f"./freeshap_res/ntk/{a.dataset}/{a.model}_seed{seed}_num{a.num_train}_val{a.val}_signFalse.pkl"
    out = None
    if os.path.exists(p):
        try:
            ntk = pickle.load(open(p, "rb"))["ntk"]
            if abs(float(ntk.mean())) > 1000:   # get_cached_ntk 와 동일 정규화
                ntk = ntk / 10000
            K0 = (ntk[0] if ntk.ndim == 3 else ntk).detach().to(torch.float64).cpu().numpy()
            n = K0.shape[1]
            K = 0.5 * (K0[:n, :] + K0[:n, :].T)
            evals = np.linalg.eigh(K)[0][::-1]          # 내림차순
            kappa2 = float(np.max(np.diag(K)))
            out = (np.clip(evals, 0.0, None), kappa2, n)
        except Exception as ex:
            print(f"[bound] spectrum 계산 실패 (seed{seed}): {ex}")
    _SPEC_CACHE[key] = out
    return out

def theory_bounds(a, r, delta=0.1):
    """rank r(%) 에서 (eigen bound, uniform-Nyström bound) — seed 평균.
    Prop 0.2: |Δφ_i| ≤ 4κ²ρ⁻²(1+κ²ρ⁻¹)·(1+log N)/N · ε_N,  ε_N(eigen)=λ_{r+1}.
    Cor 0.1 (uniform):  ... × [λ_{m+1}/N + (κ²/√m)(2+log δ⁻¹)],  m=landmark 수.
    ρ = a.lam_fix (구현의 K+λI 를 ρ=λ 로 해석)."""
    rho = float(a.lam_fix)
    eb, nb = [], []
    for s in a.seeds:
        sp = _spectrum(a, s)
        if sp is None: continue
        evals, k2, n = sp
        pref = 4.0 * k2 * rho**-2 * (1.0 + k2 / rho) * (1.0 + np.log(n))
        m = max(1, int(n * r / 100))
        lam_next = float(evals[m]) if m < len(evals) else 0.0
        eb.append(pref * lam_next / n)
        nb.append(pref * (lam_next / n + (k2 / np.sqrt(m)) * (2.0 + np.log(1.0 / delta))))
    return (float(np.mean(eb)) if eb else np.nan,
            float(np.mean(nb)) if nb else np.nan)

def sv_maxabs_seeds(a, m, r, INV):
    """실측 per-point max SV 오차 max_i|φ_a,i − φ_inv,i| — bound 와 같은 단위."""
    vals = []
    for s in a.seeds:
        asv, asi = load_sv(find3(a, m, r, s, "sv"))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        if len(asv) == len(isv) and np.array_equal(asi, isi):
            av, bv = asv, isv
        else:
            av, bv = align(asv, asi, isv, isi)
        if len(av) < 10: continue
        vals.append(float(np.max(np.abs(av - bv))))
    return _mean_std_n(vals)

def page_bound(pdf, a, INV):
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    ranks = np.array(a.ranks, dtype=float)
    # 실측: 3 method 의 per-point max 오차
    for m in a.methods:
        mm, ss = rank_row(a, (lambda r, INV, _m=m: sv_maxabs_seeds(a, _m, r, INV)), INV)
        ok = np.isfinite(mm)
        if not ok.any(): continue
        col = M3[m]["col"]
        ax.plot(ranks[ok], mm[ok], "o-", color=col, lw=1.9, ms=5,
                label=f"{M3[m]['kor']} (실측 max|Δφ|)")
        lo = np.clip(mm[ok] - ss[ok], 1e-12, None)
        ax.fill_between(ranks[ok], lo, mm[ok] + ss[ok], color=col, alpha=.15)
    # 이론 상한 점선 (eigen / uniform-Nyström)
    eb = []; nb = []
    for r in a.ranks:
        e, nn = theory_bounds(a, r)
        eb.append(e); nb.append(nn)
    eb = np.array(eb); nb = np.array(nb)
    if np.isfinite(eb).any():
        ax.plot(ranks, eb, ":", color=M3["eigen"]["col"], lw=2.4,
                label="Eigen upper bound (Prop 0.2 + Eckart–Young)")
    if np.isfinite(nb).any():
        ax.plot(ranks, nb, ":", color=M3["nystrom_pinv"]["col"], lw=2.4,
                label="Nyström upper bound (Cor 0.1, uniform, δ=0.1)")
    ax.set_yscale("log")
    ax.set_title(f"[{a.dataset}] per-point SV 오차 max|φ_approx − φ_inv| vs 이론 상한 "
                 f"({fixed} 고정, ρ=λ={a.lam_fix:g} 해석, κ²=max K_ii, mean±std, seeds={a.seeds})",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("rank/d (%)", fontsize=11); ax.set_ylabel("per-point SV error (log)", fontsize=11)
    ax.set_xticks(ranks); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9.5)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 페이지 ============
def _metrics(a):
    return [
        ("SV Spearman ρ (vs inv)",            lambda m, r, INV: fid_seeds(a, m, r, INV, "sp"), "viridis", (0, 1), None),
        ("SV Pearson r (vs inv)",             lambda m, r, INV: fid_seeds(a, m, r, INV, "pe"), "viridis", (0, 1), None),
        (f"Selection AUC k1–100%",            lambda m, r, INV: sel_auc_seeds(a, m, r, 100),   "viridis", None,
         dict(inv=base_sel(a, 100, "inv"), rnd=base_sel(a, 100, "rnd"), better="높을수록")),
        (f"Selection AUC k1–{a.sel_k}%",      lambda m, r, INV: sel_auc_seeds(a, m, r, a.sel_k), "viridis", None,
         dict(inv=base_sel(a, a.sel_k, "inv"), rnd=base_sel(a, a.sel_k, "rnd"), better="높을수록")),
        ("Top-removal AUC 0–99% (낮을수록 좋음)",  lambda m, r, INV: rem_auc_seeds(a, m, r, "top_removal"), "viridis_r", None,
         dict(inv=base_rem(a, "top_removal"), rnd=base_rem(a, "random"), better="낮을수록")),
        ("Bottom-removal AUC 0–99% (높을수록 좋음)", lambda m, r, INV: rem_auc_seeds(a, m, r, "bottom_removal"), "viridis", None,
         dict(inv=base_rem(a, "bottom_removal"), rnd=base_rem(a, "random"), better="높을수록")),
    ]

def rank_row(a, fn, INV):
    mm = np.full(len(a.ranks), np.nan); ss = np.full(len(a.ranks), np.nan)
    for j, r in enumerate(a.ranks):
        v, sd, _ = fn_call(fn, r, INV)
        mm[j] = v; ss[j] = sd
    return mm, ss

def fn_call(fn_m, r, INV):
    return fn_m(r, INV)

def page_heat(pdf, a, INV, METRICS):
    xl = [f"{r:g}" for r in a.ranks]
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    nM = len(a.methods); nR = len(METRICS)
    fig, axes = plt.subplots(nR, nM, figsize=(7.2 * nM, 3.3 * nR), squeeze=False)
    fig.suptitle(f"[{a.dataset}] Eigen vs Nyström-pinv vs Nyström-lev — rank sweep 히트맵 "
                 f"({fixed} 고정, x=rank/d %, mean±std)\n"
                 f"n={a.num_train}, tmc={a.tmc}, seeds={a.seeds}",
                 fontsize=13, fontweight="bold")
    for ri, (ttl, fmaker, cmap, vrange, _base) in enumerate(METRICS):
        # 지표별 공통 색범위 (method 간 비교 가능하게)
        allv = []
        rows = {}
        for m in a.methods:
            mm, ss = rank_row(a, (lambda r, INV, _m=m: fmaker(_m, r, INV)), INV)
            rows[m] = (mm, ss); allv += [v for v in mm if np.isfinite(v)]
        if vrange is None:
            lo, hi = (min(allv), max(allv)) if allv else (0.0, 1.0)
            if hi - lo < 1e-6: hi = lo + 1e-6
        else:
            lo, hi = vrange
        for ci, m in enumerate(a.methods):
            ax = axes[ri][ci]
            mm, ss = rows[m]
            draw_heat(ax, mm[None, :], ss[None, :], xl, [f"{M3[m]['kor']}"], ttl,
                      cmap, lo, hi, None, xlab="rank/d (%)", ylab="",
                      fmt="{:.3f}", sfmt="{:.3f}", fs_scale=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def page_curves(pdf, a, INV, METRICS):
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    fig, axes = plt.subplots(3, 2, figsize=(15, 16.2))
    fig.suptitle(f"[{a.dataset}] Fidelity/AUC vs rank ({fixed} 고정, mean±std band, seeds={a.seeds})\n"
                 f"파랑=Eigen, 초록=Nyström-lev, 빨강=Nyström-pinv / 검정 파선=inv(exact)",
                 fontsize=12.5, fontweight="bold")
    ranks = np.array(a.ranks, dtype=float)
    # 1행: Spearman/Pearson (base=None) -> 2~3행: downstream AUC 4개
    ordered = [M for M in METRICS if M[4] is None] + [M for M in METRICS if M[4] is not None]
    for ax, (ttl, fmaker, _c, _v, base) in zip(axes.flat, ordered):
        y_lo = []
        for m in a.methods:
            mm, ss = rank_row(a, (lambda r, INV, _m=m: fmaker(_m, r, INV)), INV)
            ok = np.isfinite(mm)
            if not ok.any(): continue
            col = M3[m]["col"]
            ax.plot(ranks[ok], mm[ok], "o-", color=col, lw=1.9, ms=5, label=M3[m]["kor"])
            ax.fill_between(ranks[ok], mm[ok] - ss[ok], mm[ok] + ss[ok], color=col, alpha=.15)
            band_lo = mm[ok] - np.where(np.isfinite(ss[ok]), ss[ok], 0)
            y_lo.append(float(np.min(band_lo)))
        if base is None:
            # fidelity 상관: 1.0 참조선은 유지하되 아래쪽은 데이터에 맞게 (여백 제거)
            ax.axhline(1.0, color="black", ls="--", lw=1.2, alpha=.6, label="inv (=1)")
            lo = min(y_lo) if y_lo else 0.0
            pad = max(0.02, (1.0 - lo) * 0.10)
            ax.set_ylim(max(0.0, lo - pad), 1.02)
            ax.set_ylabel("correlation (vs inv)", fontsize=10)
        else:
            im, isd, _ = base["inv"]
            if np.isfinite(im):
                ax.axhline(im, color="black", ls="--", lw=1.8, label="inv (exact)")
                if np.isfinite(isd) and isd > 0:
                    ax.axhspan(im - isd, im + isd, color="black", alpha=.06)
            ax.set_ylabel("AUC (val acc 평균)", fontsize=10)
        ax.set_title(f"{ttl}", fontsize=11)
        ax.set_xlabel("rank/d (%)", fontsize=10)
        ax.set_xticks(ranks); ax.grid(True, alpha=.3); ax.legend(fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, .94])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_errdist(pdf, a, INV):
    """p6: rank별 SV 점별 오차 분포 (2×4) — d = sv_approx − sv_inv 히스토그램.
    inv 는 오차 0 이므로 검정 점선(x=0)으로 표시. seed 는 존재하는 것 전부 pooling.
    클리핑 한계 XLIM 은 전체 오차의 99% 분위수로 자동 결정 (칸마다 동일 축)."""
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    # ---- 1) 오차 + 방법별 SV 값 수집: {rank: {method: (pooled diffs, pooled sv)}} ----
    diffs, used_seeds = {}, set()
    for r in a.ranks:
        diffs[r] = {}
        for m in a.methods:
            pool_d, pool_sv = [], []
            for s in a.seeds:
                asv, asi = load_sv(find3(a, m, r, s, "sv"))
                isv, isi = INV.get(s, (None, None))
                if asv is None or isv is None:
                    continue
                if len(asv) == len(isv) and np.array_equal(asi, isi):
                    av, bv = np.asarray(asv, float), np.asarray(isv, float)
                else:
                    av, bv = align(asv, asi, isv, isi)
                    av, bv = np.asarray(av, float), np.asarray(bv, float)
                if len(av) < 10:
                    continue
                pool_d.append(av - bv); pool_sv.append(av); used_seeds.add(s)
            if pool_d:
                diffs[r][m] = (np.concatenate(pool_d), np.concatenate(pool_sv))
    all_d = np.concatenate([d for r in a.ranks for d, _ in diffs[r].values()]) \
        if any(diffs[r] for r in a.ranks) else np.array([0.0])
    XLIM = float(np.percentile(np.abs(all_d), 99)) or 1.0
    bins = np.linspace(-XLIM, XLIM, 61)

    # inv SV 자체의 스케일 (seed pooling) — 오차 크기 해석 기준
    inv_pool = [np.asarray(INV[s][0], float) for s in a.seeds
                if INV.get(s, (None,))[0] is not None]
    inv_stats = None
    if inv_pool:
        iv = np.concatenate(inv_pool)
        inv_stats = (iv.mean(), iv.std(), iv.min(), iv.max())

    # ---- 2) 2×4 그리드 (7 rank + 범례 칸) ----
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    for k, r in enumerate(a.ranks):
        ax = axes[k // 4][k % 4]
        res = diffs[r]
        if not res:
            ax.text(.5, .5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        clipped = tot = 0
        # ---- 좌상단 주석: inv SV 스케일 + 방법별 [SV 값 통계 / |err| max·p99] (방법 색) ----
        y_txt = .97; dy = .058
        if inv_stats is not None:
            mu, sd, mn, mx = inv_stats
            ax.text(.02, y_txt, f"inv sv: μ={mu:.3g} σ={sd:.3g} [{mn:.3g}, {mx:.3g}]",
                    transform=ax.transAxes, va="top", fontsize=6.0, color="black")
            y_txt -= dy
        for m in a.methods:
            if m not in res:
                continue
            d, sv = res[m]; clipped += int((np.abs(d) > XLIM).sum()); tot += d.size
            ax.hist(np.clip(d, -XLIM, XLIM), bins=bins, histtype="step",
                    color=M3[m]["col"], lw=1.6, density=True)
            ad = np.abs(d)
            ax.text(.02, y_txt,
                    f"{M3[m]['kor']} sv: μ={sv.mean():.3g} σ={sv.std():.3g} "
                    f"[{sv.min():.3g}, {sv.max():.3g}]",
                    transform=ax.transAxes, va="top", fontsize=6.0, color=M3[m]["col"])
            y_txt -= dy
            ax.text(.02, y_txt,
                    f"  |err|: max={ad.max():.3g} p99={np.percentile(ad, 99):.3g}",
                    transform=ax.transAxes, va="top", fontsize=6.0, color=M3[m]["col"])
            y_txt -= dy
        ax.axvline(0, color="black", ls="--", lw=1.3)   # inv (오차 0 기준선)
        ax.set_title(f"rank {r:g}%", fontsize=11)
        if tot:
            ax.text(.02, y_txt, f"{100*clipped/tot:.1f}% |err|>{XLIM:.2g}",
                    transform=ax.transAxes, va="top", fontsize=6.0, color="dimgray")
        ax.set_xlabel("error (approx − inv)", fontsize=9)
        if k % 4 == 0:
            ax.set_ylabel("density", fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(True, alpha=.25)
    # 8번째 칸: 범례
    axL = axes[1][3]; axL.axis("off")
    hands = [plt.Line2D([0], [0], color="black", ls="--", lw=1.3)] + \
            [plt.Line2D([0], [0], color=M3[m]["col"], lw=1.6) for m in a.methods]
    labs = ["inv (오차 0)"] + [M3[m]["kor"] for m in a.methods]
    axL.legend(hands, labs, loc="center", fontsize=11, frameon=False)
    axL.text(.5, .08, f"seeds pooled: {sorted(used_seeds)}", ha="center",
             transform=axL.transAxes, fontsize=8.5, color="dimgray")
    fig.suptitle(f"[{a.dataset}] SV 점별 오차 분포 — d = sv_approx − sv_inv "
                 f"({fixed} 고정, clip ±{XLIM:.2g}, 61 bins, density)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _spike_npzs(a):
    return sorted(glob.glob(f"./jitter_exp/coalition_spike/{a.model}_{a.dataset}_seed*.npz"))


def ensure_spike(a):
    """coalition_spike npz 없으면 compute 시도 (모델 로드 필요 — 실패 시 빈 페이지)."""
    if _spike_npzs(a):
        return
    if a.model == "llama":   # llama 모델 로드(대용량) 유발 방지 — llama 서버에서 수동 실행
        print("[auto] coalition_spike npz 없음 (llama 는 자동 계산 안 함 — "
              "llama 서버에서 compute_coalition_spike.py 실행)")
        return
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_coalition_spike.py")
    print("[auto] coalition_spike npz 없음 -> compute 시도 (seed 2024)...")
    rr = subprocess.run([sys.executable, script, "--dataset", a.dataset, "--model", a.model,
                         "--seeds", "2024", "--ranks", *[f"{r:g}" for r in a.ranks],
                         "--lam", str(a.lam_fix), "--eps", str(a.eps_fix)])
    if rr.returncode != 0:
        print("[auto] coalition_spike compute 실패 — p7 은 빈 페이지로 진행")


def page_spike(pdf, a):
    """p7: coalition-스파이크 (2×4) — 부분집합 크기 n 별 |acc_approx − acc_inv| (%p).
    저랭크 근사의 보간 임계(|S|≈d)에서 유틸리티 오염이 피크 — TMC 적분을 통해
    SV 오차 정체/상승의 원인이 되는 구간을 rank 별로 시각화. npz 는 compute_coalition_spike.py."""
    files = _spike_npzs(a)
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    if not files:
        for row in axes:
            for ax in row: ax.axis("off")
        axes[0][0].text(.5, .5, "(no data — compute_coalition_spike.py 실행 필요;\n"
                        "llama 는 llama 있는 서버에서)", ha="center", va="center", fontsize=11)
        fig.suptitle(f"[{a.dataset}] coalition-스파이크 (데이터 없음)", fontsize=12.5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); return
    zs = [dict(np.load(f, allow_pickle=True)) for f in files]
    seeds_used = [re.search(r"_seed(\d+)\.npz", f).group(1) for f in files]
    for k, r in enumerate(a.ranks):
        ax = axes[k // 4][k % 4]
        d_act = int(a.num_train * r / 100)
        for m in a.methods:
            key_s, key_v = f"{m}_r{r:g}_sizes", f"{m}_r{r:g}_vals"
            zs_ok = [z for z in zs if key_s in z]
            if not zs_ok:
                continue
            sizes = zs_ok[0][key_s]
            vals = np.mean([z[key_v] for z in zs_ok], axis=0)
            ax.plot(sizes, vals, marker="o", ms=3.5, lw=1.5,
                    color=M3[m]["col"], label=M3[m]["kor"])
        ax.axvline(d_act, color="black", ls=":", lw=1.2)
        ax.text(d_act, ax.get_ylim()[1] * .97, f" n=d={d_act}", fontsize=7,
                va="top", color="black")
        ax.set_title(f"rank {r:g}% (d={d_act})", fontsize=11)
        ax.set_xlabel("coalition size |S|", fontsize=9)
        if k % 4 == 0:
            ax.set_ylabel("|acc_approx − acc_inv| (%p)", fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(True, alpha=.25)
    axL = axes[1][3]; axL.axis("off")
    hands = [plt.Line2D([0], [0], color=M3[m]["col"], marker="o", ms=4, lw=1.5) for m in a.methods] \
        + [plt.Line2D([0], [0], color="black", ls=":", lw=1.2)]
    labs = [M3[m]["kor"] for m in a.methods] + ["보간 임계 n=d"]
    axL.legend(hands, labs, loc="center", fontsize=11, frameon=False)
    axL.text(.5, .1, f"seeds: {seeds_used} / 랜덤 coalition 평균", ha="center",
             transform=axL.transAxes, fontsize=8.5, color="dimgray")
    fig.suptitle(f"[{a.dataset}] coalition-스파이크 — 부분집합 크기별 유틸리티 오염 "
                 f"|acc_approx(S) − acc_inv(S)| ({fixed} 고정; |S|≈d 에서 피크 = SV 오차 정체/상승 원인)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    a = parse_args()
    if a.val is None:
        for s in a.seeds:
            g = glob.glob(f"./freeshap_res/ntk/{a.dataset}/{a.model}_seed{s}_num{a.num_train}_val*_signFalse.pkl")
            if g:
                a.val = int(re.search(r"_val(\d+)_", os.path.basename(g[0])).group(1)); break
    if a.val is None:
        print("[error] val 자동 감지 실패 — --val 지정 필요"); return
    if a.out is None:
        a.out = f"./jitter_exp/report_eigen_pinv_lev_{a.dataset}.pdf"
    print(f"[cfg] dataset={a.dataset} seeds={a.seeds} n={a.num_train} val={a.val} tmc={a.tmc} "
          f"ranks={[f'{r:g}' for r in a.ranks]} lam_fix={a.lam_fix:g} eps_fix={a.eps_fix:g} "
          f"methods={a.methods}")

    INV = {s: load_sv(inv_path(a, s, "sv")) for s in a.seeds}
    print(f"[inv] SV 존재 seed: {[s for s in a.seeds if INV[s][0] is not None]}")

    ensure_npz(a)   # 커널/예측 오차용 npz 자동 준비 (lev 포함)

    METRICS = _metrics(a)
    ERRM = _err_metrics(a)
    with PdfPages(a.out) as pdf:
        page_heat(pdf, a, INV, METRICS)      # p1: fidelity+AUC 히트맵 (6지표 × method)
        page_curves(pdf, a, INV, METRICS)    # p2: fidelity+AUC 곡선 3×2
        page_heat(pdf, a, INV, ERRM)         # p3: 상대오차 히트맵 (4지표 × method)
        page_err_curves(pdf, a, INV, ERRM)   # p4: 상대오차 곡선 2×2 (log y)
        page_bound(pdf, a, INV)              # p5: per-point max 오차 vs 이론 상한 (점선)
        page_errdist(pdf, a, INV)            # p6: rank별 SV 점별 오차 분포 (2×4)
        # p7 coalition-스파이크는 제외 (데이터셋 간 경향성 불명확 — 필요 시 아래 두 줄 복원)
        # ensure_spike(a)
        # page_spike(pdf, a)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()

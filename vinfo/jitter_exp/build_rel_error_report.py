# -*- coding: utf-8 -*-
"""상대오차 리포트 — 3종 지표 (모두 '낮을수록 inv 에 가까움'):

  1. Kernel approx 상대오차      ||K - Phi Phi^T||_F / ||K||_F   (train kernel 2000x2000; λ 무관)
  2. Kernel val-예측 상대오차    ||K(D) - K_a(D)||_F / ||K(D)||_F (D=val set, inv λ=1e-6 기준)
  3. Shapley value 상대오차      ||sv_a - sv_inv||_2 / ||sv_inv||_2 (i번째 포인트끼리 직접)

두 가지 sweep 모드:
  --sweep grid (기본): rank 고정(20%), 4x4 (λ×ε) 히트맵 1페이지.
  --sweep rank       : (λ,ε) 고정, rank ∈ {1,5,10,15,20,25,30}% 스윕.
                       p1 = 1×7 히트맵 (지표×method), p2 = 지표별 선그래프 3개
                       (x=rank, y=상대오차 log 스케일; 빨강=nys_pinv, 파랑=eigen).

데이터: compute_kernel_pred_error.py 의 npz (logits+kernel_relerr) + shapley pkl.
npz 없거나 필드 없으면 자동 계산/backfill (커널·예측 오차는 SV pkl 없이도 계산됨 →
rank sweep 의 그래프 1·2 는 지금 데이터로도 채워지고, 그래프 3(SV)만 rank 별 SV pkl 필요).

사용:
  python jitter_exp/build_rel_error_report.py --dataset rte                       # grid 모드
  python jitter_exp/build_rel_error_report.py --dataset rte --sweep rank          # rank 모드
      (기본 --ranks 1 5 10 15 20 25 30 --lam_fix 1e-2 --eps_fix 1e-8)
"""
import os, sys, copy, argparse, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pinv_selection_report as R
from build_pinv_selection_report import (  # noqa: E402
    METH, GRID_LAMS, GRID_EPS, draw_heat, _fmt_eps, _mean_std_n,
    load_sv, sv_path, inv_sv_path, kp_npz_path, kpred_err_seeds, sv_relerr_seeds,
)

MCOL = {"nystrom_pinv": "#d62728", "eigen": "#1f77b4", "nystrom": "#7f7f7f"}  # 빨강=pinv, 파랑=eigen


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--methods", type=str, nargs="+", default=["nystrom_pinv", "eigen"])
    p.add_argument("--model", type=str, default="bert")
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--val", type=int, default=None, help="없으면 파일명에서 자동 감지")
    p.add_argument("--tmc", type=int, default=500)
    p.add_argument("--rank", type=float, default=20, help="grid 모드에서의 고정 rank/d %%")
    p.add_argument("--invlam", type=str, default="1e-06")
    p.add_argument("--sweep", choices=["grid", "rank"], default="grid",
                   help="grid=λ×ε 4x4 (rank 고정) / rank=rank 스윕 ((λ,ε) 고정)")
    p.add_argument("--ranks", type=float, nargs="+", default=[1, 5, 10, 15, 20, 25, 30],
                   help="rank 모드에서의 rank/d %% 목록")
    p.add_argument("--lam_fix", type=float, default=1e-2, help="rank 모드에서 고정할 λ")
    p.add_argument("--eps_fix", type=float, default=1e-8, help="rank 모드에서 고정할 ε")
    p.add_argument("--res_root", type=str, default="./jitter_exp/res")
    p.add_argument("--pinv_root", type=str, default="./jitter_exp/nys_pinv_res")
    p.add_argument("--inv_root", type=str, default="./freeshap_res")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def a_rank(a, r):
    ar = copy.copy(a); ar.rank = r; return ar


# ---------- npz 자동 준비 ----------
def ensure_npz(a, ranks, lams, epss):
    """조합별 npz 없으면 compute, kernel_relerr 필드 없으면 backfill."""
    kp_methods = [m for m in a.methods if m in ("eigen", "nystrom_pinv")]
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_kernel_pred_error.py")
    common = ["--dataset", a.dataset, "--ranks", *[f"{r:g}" for r in ranks],
              "--lams", *map(str, lams), "--epss", *map(str, epss),
              "--res_root", a.res_root, "--pinv_root", a.pinv_root, "--inv_root", a.inv_root]

    def _paths(s):
        yield kp_npz_path(a, "inv", None, None, s)
        for r in ranks:
            ar = a_rank(a, r)
            for m in kp_methods:
                for l in lams:
                    for e in epss:
                        yield kp_npz_path(ar, m, l, e, s)

    missing = [s for s in a.seeds if any(not os.path.exists(p) for p in _paths(s))]
    if missing:
        print(f"[auto] npz 미비 seed {missing} -> compute_kernel_pred_error 실행...")
        r = subprocess.run([sys.executable, script, *common, "--seeds", *map(str, missing)])
        if r.returncode != 0:
            print("[auto] compute 실패 - 있는 데이터만으로 진행")
    need_bf = [s for s in a.seeds
               if any(os.path.exists(p) and "kernel_relerr" not in np.load(p).files
                      for p in _paths(s))]
    if need_bf:
        print(f"[auto] kernel_relerr 필드 미비 seed {need_bf} -> backfill (NTK 만 필요)...")
        r = subprocess.run([sys.executable, script, *common, "--seeds", *map(str, need_bf),
                            "--backfill_kernel"])
        if r.returncode != 0:
            print("[auto] backfill 실패 - 있는 데이터만으로 진행")


# ---------- SV fallback 탐색 (--res_root 안 바꿔도 freeshap_res 등에서 찾음) ----------
SV_FALLBACK_ROOTS = ["./jitter_exp/res", "./freeshap_res", "./jitter_exp/nys_pinv_res"]

def sv_path_any(a, method, lam, eps, seed):
    """1차: 정규 root(sv_path). 없으면 같은 파일명을 fallback root 들에서 탐색."""
    p = sv_path(a, method, lam, eps, seed)
    if os.path.exists(p): return p
    fn = os.path.basename(p)
    for root in SV_FALLBACK_ROOTS:
        q = os.path.join(root, "shapley", a.dataset, method, fn)
        if os.path.exists(q): return q
    return p

def sv_relerr_seeds_any(a, method, lam, eps, INV):
    """R.sv_relerr_seeds 와 동일하되 SV 를 fallback 탐색으로 로드."""
    vals = []
    for s in a.seeds:
        asv, asi = load_sv(sv_path_any(a, method, lam, eps, s))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        if len(asv) == len(isv) and np.array_equal(asi, isi):
            av, bv = asv, isv
        else:
            av, bv = R.align(asv, asi, isv, isi)
        if len(av) < 10: continue
        den = float(np.linalg.norm(bv))
        if den == 0: continue
        vals.append(float(np.linalg.norm(av - bv) / den))
    return _mean_std_n(vals)


def sv_normed_relerr_seeds_any(a, method, lam, eps, INV):
    """스케일 제거 SV 오차: ||sv_a/||sv_a||_2 - sv_inv/||sv_inv||_2||_2  (= sqrt(2(1-cos)), 범위 0~2)."""
    vals = []
    for s in a.seeds:
        asv, asi = load_sv(sv_path_any(a, method, lam, eps, s))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        if len(asv) == len(isv) and np.array_equal(asi, isi):
            av, bv = asv, isv
        else:
            av, bv = R.align(asv, asi, isv, isi)
        if len(av) < 10: continue
        na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
        if na == 0 or nb == 0: continue
        vals.append(float(np.linalg.norm(av / na - bv / nb)))
    return _mean_std_n(vals)


def sv_corr_seeds_any(a, method, lam, eps, INV, which):
    """SV vs inv 상관: which='sp'(Spearman)/'pe'(Pearson). fallback 로드 공유."""
    vals = []
    for s in a.seeds:
        asv, asi = load_sv(sv_path_any(a, method, lam, eps, s))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        if len(asv) == len(isv) and np.array_equal(asi, isi):
            av, bv = asv, isv
        else:
            av, bv = R.align(asv, asi, isv, isi)
        if len(av) < 10: continue
        if which == "sp":
            vals.append(float(np.corrcoef(R._rank(av), R._rank(bv))[0, 1]))
        else:
            vals.append(float(np.corrcoef(av, bv)[0, 1]))
    return _mean_std_n(vals)


# ---------- 지표별 seed 집계 ----------
def kernel_err_seeds(a, method, lam, eps):
    vals = []
    for s in a.seeds:
        p = kp_npz_path(a, method, lam, eps, s)
        if not os.path.exists(p): continue
        d = np.load(p)
        if "kernel_relerr" not in d.files: continue
        vals.append(float(d["kernel_relerr"]))
    return _mean_std_n(vals)


METRICS = [
    ("kernel", "Kernel approx 상대오차  ||K - PhiPhi^T||_F / ||K||_F   (train kernel; λ 무관·ε 만 관여)",
     lambda a, m, l, e, INV: kernel_err_seeds(a, m, l, e)),
    ("kpred", "Kernel val-예측 상대오차  ||K(D) - K_a(D)||_F / ||K(D)||_F   (D=val set, 기준 inv λ=1e-6)",
     lambda a, m, l, e, INV: kpred_err_seeds(a, m, l, e)),
    ("sv", "Shapley value 상대오차  ||sv_a - sv_inv||_2 / ||sv_inv||_2   (i번째 포인트끼리 직접)",
     lambda a, m, l, e, INV: sv_relerr_seeds_any(a, m, l, e, INV)),
    ("svn", "SV 정규화 상대오차  ||sv_a/||sv_a|| - sv_inv/||sv_inv||||_2   (스케일 제거 = 방향 차이; =sqrt(2(1-cos)), 0~2)",
     lambda a, m, l, e, INV: sv_normed_relerr_seeds_any(a, m, l, e, INV)),
]


CURVE_METRICS = METRICS + [
    ("sp", "SV Spearman ρ (vs inv)   — 높을수록 좋음", 
     lambda a, m, l, e, INV: sv_corr_seeds_any(a, m, l, e, INV, "sp")),
    ("pe", "SV Pearson r (vs inv)   — 높을수록 좋음",
     lambda a, m, l, e, INV: sv_corr_seeds_any(a, m, l, e, INV, "pe")),
]


# ---------- grid 모드 (rank 고정, λ×ε 4x4) ----------
def page_grid(pdf, a, INV):
    xl = [_fmt_eps(e) for e in GRID_EPS]; yl = [_fmt_eps(l) for l in GRID_LAMS]
    nM = len(a.methods)
    fig, axes = plt.subplots(len(METRICS), nM, figsize=(7.4 * nM, 5.4 * len(METRICS)), squeeze=False)
    fig.suptitle(f"[{a.dataset}] 상대오차 리포트 — 4x4 고정 grid (λ×ε), mean±std, "
                 f"n={a.num_train}, tmc={a.tmc}, rank/d={a.rank:g}%, seeds={a.seeds}"
                 f"\n(모든 지표: 낮을수록 inv 에 가까움)", fontsize=12.5, fontweight="bold")
    for ri, (key, rttl, fn) in enumerate(METRICS):
        for ci, method in enumerate(a.methods):
            ax = axes[ri][ci]; m = METH[method]
            Mm = np.full((len(GRID_LAMS), len(GRID_EPS)), np.nan); Ms = np.full_like(Mm, np.nan)
            for i, l in enumerate(GRID_LAMS):
                for j, e in enumerate(GRID_EPS):
                    mm, ss, _ = fn(a, method, l, e, INV); Mm[i, j] = mm; Ms[i, j] = ss
            fin = Mm[np.isfinite(Mm)]
            vmx = min(1.0, float(fin.max())) if fin.size else 1.0
            draw_heat(ax, Mm, Ms, xl, yl,
                      f"{m['kor']} — {rttl.split('  ')[0]}  (색 상한 {vmx:g}; 숫자=실제값)",
                      "RdYlGn_r", 0.0, vmx, None,
                      xlab=f"{m['eps']} (ε)", ylab=f"{m['lam']} (λ)", fmt="{:.3f}", sfmt="{:.3f}")
        axes[ri][0].annotate(rttl, xy=(0, 1.16), xycoords="axes fraction",
                             fontsize=9.5, fontweight="bold", color="#0b5394")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------- rank 모드 (λ,ε 고정, rank 스윕) ----------
def rank_matrix(a, method, fn, INV):
    """(mean[7], std[7]) — rank 별 지표."""
    mm = np.full(len(a.ranks), np.nan); ss = np.full(len(a.ranks), np.nan)
    for j, r in enumerate(a.ranks):
        m, s, _ = fn(a_rank(a, r), method, a.lam_fix, a.eps_fix, INV)
        mm[j] = m; ss[j] = s
    return mm, ss


def page_rank_heat(pdf, a, INV):
    xl = [f"{r:g}" for r in a.ranks]
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    nM = len(a.methods)
    fig, axes = plt.subplots(len(METRICS), nM, figsize=(7.4 * nM, 3.5 * len(METRICS)), squeeze=False)
    fig.suptitle(f"[{a.dataset}] 상대오차 — rank sweep ({fixed} 고정), mean±std, "
                 f"n={a.num_train}, tmc={a.tmc}, seeds={a.seeds}"
                 f"\n(모든 지표: 낮을수록 inv 에 가까움)", fontsize=12.5, fontweight="bold")
    for ri, (key, rttl, fn) in enumerate(METRICS):
        for ci, method in enumerate(a.methods):
            ax = axes[ri][ci]; m = METH[method]
            mm, ss = rank_matrix(a, method, fn, INV)
            Mm = mm[None, :]; Ms = ss[None, :]
            fin = Mm[np.isfinite(Mm)]
            vmx = min(1.0, float(fin.max())) if fin.size else 1.0
            draw_heat(ax, Mm, Ms, xl, [fixed], f"{m['kor']} — {rttl.split('  ')[0]}",
                      "RdYlGn_r", 0.0, vmx, None,
                      xlab="rank/d (%)", ylab="", fmt="{:.3f}", sfmt="{:.3f}")
        axes[ri][0].annotate(rttl, xy=(0, 1.28), xycoords="axes fraction",
                             fontsize=9.5, fontweight="bold", color="#0b5394")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_rank_curves(pdf, a, INV):
    fixed = f"λ={_fmt_eps(a.lam_fix)}, ε={_fmt_eps(a.eps_fix)}"
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.0))
    fig.suptitle(f"[{a.dataset}] 상대오차/상관 vs rank ({fixed} 고정, 오차=log·상관=linear, "
                 f"mean±std band, seeds={a.seeds})   빨강=Nyström-pinv, 파랑=Eigen",
                 fontsize=12.5, fontweight="bold")
    ranks = np.array(a.ranks, dtype=float)
    for ax, (key, rttl, fn) in zip(axes.flat, CURVE_METRICS):
        is_corr = key in ("sp", "pe")
        y_lo = []
        for method in a.methods:
            mm, ss = rank_matrix(a, method, fn, INV)
            ok = np.isfinite(mm)
            if not ok.any(): continue
            col = MCOL.get(method, None)
            ax.plot(ranks[ok], mm[ok], "o-", color=col, lw=1.8, ms=5,
                    label=METH[method]["kor"])
            lo = mm[ok] - ss[ok] if is_corr else np.clip(mm[ok] - ss[ok], 1e-12, None)
            ax.fill_between(ranks[ok], lo, mm[ok] + ss[ok], color=col, alpha=.15)
            y_lo.append(float(np.min(lo)))
        if is_corr:
            # adaptive: 데이터 하한에 맞춰 아래 여백 제거 (상단 1.02 는 1.0 기준 가시화)
            lo_all = min(y_lo) if y_lo else 0.0
            pad = max(0.02, (1.0 - lo_all) * 0.10)
            ax.set_ylim(max(0.0, lo_all - pad), 1.02)
            ax.set_ylabel("correlation (vs inv)", fontsize=10)
        else:
            ax.set_yscale("log"); ax.set_ylabel("relative error (log)", fontsize=10)
        ax.set_title(rttl.split("  ")[0], fontsize=11)
        ax.set_xlabel("rank/d (%)", fontsize=10)
        ax.set_xticks(ranks); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .94])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    a = parse_args()
    if a.val is None:
        for m in a.methods:
            v = R.autodetect_val(a, m)
            if v is not None: a.val = v; break
    if a.val is None:
        print("[error] val 자동 감지 실패 — --val 지정 필요"); return
    if a.out is None:
        a.out = (f"./jitter_exp/report_rel_error_{a.dataset}.pdf" if a.sweep == "grid"
                 else f"./jitter_exp/report_rel_error_{a.dataset}_rank.pdf")
    print(f"[cfg] dataset={a.dataset} sweep={a.sweep} seeds={a.seeds} n={a.num_train} "
          f"val={a.val} tmc={a.tmc} methods={a.methods}"
          + (f" ranks={[f'{r:g}' for r in a.ranks]} lam_fix={a.lam_fix:g} eps_fix={a.eps_fix:g}"
             if a.sweep == "rank" else f" rank={a.rank:g}"))

    if a.sweep == "grid":
        ensure_npz(a, ranks=[a.rank], lams=GRID_LAMS, epss=GRID_EPS)
    else:
        ensure_npz(a, ranks=a.ranks, lams=[a.lam_fix], epss=[a.eps_fix])

    INV = {s: load_sv(inv_sv_path(a, s)) for s in a.seeds}
    print(f"[inv] SV 존재 seed: {[s for s in a.seeds if INV[s][0] is not None]}")

    with PdfPages(a.out) as pdf:
        if a.sweep == "grid":
            page_grid(pdf, a, INV)
        else:
            page_rank_heat(pdf, a, INV)     # p1: 1×7 히트맵 (지표×method)
            page_rank_curves(pdf, a, INV)   # p2: 선그래프 3개 (log y)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Data-removal 리포트 — build_selection_report.py(selection)와 동일한 그리드 히트맵 구조.
  * SV/fidelity 는 selection 과 동일한 clean SV 재사용 (Spearman/Pearson vs inv).
  * downstream = removal 곡선 요약(AUC = 제거% 곡선 평균), inv-mode(approx SV 랭킹+exact 커널).
      - Top-removal AUC : 고가치 제거 곡선 평균. **낮을수록** valuation 충실(빨리 하강).
      - Bottom-removal AUC: 저가치 제거 곡선 평균. **높을수록** 좋음.
  * method(eigen/nystrom)별 1페이지: 상단 Spearman/Pearson/TopRemAUC/BotRemAUC 히트맵(전체 격자) + arm.
  * 마지막: anchor 곡선 비교(inv vs eigen vs nystrom).

사용:
  python jitter_exp/build_removal_report.py --dataset qqp --seeds 2024 2025 2026 \
      --num_train 2000 --tmc 500 --remove_pcts 0 5 10 20 30 40 50 60 70 80 90
"""
import os, sys, argparse, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_selection_report as R
from build_selection_report import (draw_heat, draw_strip, cross_matrices, fid_seeds,
                          load_sv, inv_sv_path, sv_path, scan_sweeps, _fmt_eps, METH)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--methods", type=str, nargs="+", default=["nystrom", "eigen"])
    p.add_argument("--model", type=str, default="bert")
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--val", type=int, default=None)
    p.add_argument("--tmc", type=int, default=500)
    p.add_argument("--rank", type=float, default=20)
    p.add_argument("--invlam", type=str, default="1e-06")
    p.add_argument("--eps_anchor", type=float, default=None)
    p.add_argument("--lam_anchor", type=float, default=None)
    p.add_argument("--cross_only", action="store_true",
                   help="anchor 십자(arm)만 표시. 기본은 폴더의 모든 (lam,eps) 조합을 격자로 (없는 셀은 빈칸).")
    p.add_argument("--remove_pcts", type=int, nargs="+",
                   default=list(range(0, 100)),
                   help="run_removal.sh 의 REM 과 동일해야 함 (기본 0~99 전체). 곡선 x축용.")
    p.add_argument("--auc_k", type=int, default=30,
                   help="best-config 비교 페이지 우측 패널의 저-k 범위 (제거 0~K%%). 보통 20~30.")
    p.add_argument("--res_root", type=str, default="./jitter_exp/res")
    p.add_argument("--inv_root", type=str, default="./freeshap_res")
    # build_selection_report 헬퍼 호환용 (fid_seeds 등에서 참조)
    p.add_argument("--overlap_pct", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def build_grid(a, method):
    """selection SV 파일명에서 모든 (lam,eps) 조합 → 전체 격자(또는 cross)."""
    m = METH[method]
    eps0 = a.eps_anchor if a.eps_anchor is not None else m["eps_anchor_default"]
    lam0 = a.lam_anchor if a.lam_anchor is not None else m["lam_anchor_default"]
    combos = scan_sweeps(a, method)
    if not combos: return None
    def close(x, y): return abs(x - y) < abs(y) * 1e-6 + 1e-30
    if getattr(a, "cross_only", False):
        eps_at = sorted({e for (l, e) in combos if close(l, lam0)})
        lam_at = sorted({l for (l, e) in combos if close(e, eps0)})
        eps_vals = sorted(set(eps_at) | {eps0}); lam_vals = sorted(set(lam_at) | {lam0})
        cells = {(l, e) for (l, e) in combos if close(l, lam0) or close(e, eps0)}
    else:
        # 기본: 전체 격자 (없는 셀은 빈칸)
        eps_vals = sorted({e for (l, e) in combos}); lam_vals = sorted({l for (l, e) in combos})
        cells = set(combos)
    return dict(eps_vals=eps_vals, lam_vals=lam_vals, eps0=eps0, lam0=lam0, cells=cells, close=close)


def removal_path(a, method, lam, eps, seed):
    base = os.path.basename(sv_path(a, method, lam, eps, seed)).replace(".pkl", "")
    return os.path.join(a.res_root, "data_removing", a.dataset, method, "predictions", base + "_predictions.txt")

def inv_removal_path(a, seed):
    base = os.path.basename(inv_sv_path(a, seed)).replace(".pkl", "")
    return os.path.join(a.inv_root, "data_removing", a.dataset, "inv", "predictions", base + "_predictions.txt")


_RC = {}
def parse_removal(path):
    if path in _RC: return _RC[path]
    if not os.path.exists(path): _RC[path] = {}; return {}
    lines = open(path).read().splitlines()
    out = {}
    for l in lines:
        if l.strip().lower().startswith("remove_pct_list:"):
            try: out["_pcts"] = ast.literal_eval(l.split(":", 1)[1].strip())
            except Exception: pass
            break
    i = next((k for k, l in enumerate(lines) if l.strip().lower().startswith("inv mode")), None)
    if i is not None:
        cur = None
        for j in range(i + 1, len(lines)):
            s = lines[j].strip(); low = s.lower()
            if low.startswith(("eigen mode", "nystrom mode")): break
            if low.startswith("top_removal:"): cur = "top_removal"; continue
            if low.startswith("bottom_removal:"): cur = "bottom_removal"; continue
            if low.startswith("random:"): cur = "random"; continue
            if s.startswith("[") and cur is not None:
                try: out[cur] = np.array(ast.literal_eval(s), float) / 10000.0
                except Exception: pass
                cur = None
    _RC[path] = out; return out


def rem_summary_seeds(a, method, lam, eps, strat):
    vals = []
    for s in a.seeds:
        d = parse_removal(removal_path(a, method, lam, eps, s))
        if strat in d and len(d[strat]): vals.append(float(np.mean(d[strat])))
    if not vals: return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def _rng(M):
    f = np.asarray(M)[np.isfinite(M)]
    if not f.size: return 0.0, 1.0
    lo, hi = float(f.min()), float(f.max())
    return (lo, hi + 1e-6) if hi - lo < 1e-6 else (lo, hi)


def page_grid(pdf, a, method, INV):
    m = METH[method]; cross = build_grid(a, method)
    if cross is None: return
    SPm, SPs, ev, lv = cross_matrices(a, method, cross, lambda l, e: fid_seeds(a, method, l, e, INV, "sp"))
    PEm, PEs, _, _   = cross_matrices(a, method, cross, lambda l, e: fid_seeds(a, method, l, e, INV, "pe"))
    TPm, TPs, _, _   = cross_matrices(a, method, cross, lambda l, e: rem_summary_seeds(a, method, l, e, "top_removal"))
    BTm, BTs, _, _   = cross_matrices(a, method, cross, lambda l, e: rem_summary_seeds(a, method, l, e, "bottom_removal"))
    xl = [_fmt_eps(e) for e in ev]; yl = [_fmt_eps(l) for l in lv]
    ai = lv.index(cross["lam0"]) if cross["lam0"] in lv else None
    aj = ev.index(cross["eps0"]) if cross["eps0"] in ev else None
    anchor = (ai, aj)
    tlo, thi = _rng(TPm); blo, bhi = _rng(BTm)

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(3, 8, height_ratios=[3, 1, 1], hspace=.6, wspace=.85)
    fig.suptitle(f"[{a.dataset}] {m['kor']} — Fidelity + Data-removal AUC (전체 격자)  "
                 f"| rank/d={a.rank:g}%, tmc={a.tmc}, seeds={a.seeds}", fontsize=13, fontweight="bold")
    xe, ye = f"{m['eps']} (ε)", f"{m['lam']} (λ)"
    ax_sp = fig.add_subplot(gs[0, 0:2]); ax_pe = fig.add_subplot(gs[0, 2:4])
    ax_tp = fig.add_subplot(gs[0, 4:6]); ax_bt = fig.add_subplot(gs[0, 6:8])
    draw_heat(ax_sp, SPm, SPs, xl, yl, "Spearman ρ (mean±std)", "viridis", 0.0, 1.0, anchor, xlab=xe, ylab=ye)
    draw_heat(ax_pe, PEm, PEs, xl, yl, "Pearson r (mean±std)", "viridis", 0.0, 1.0, anchor, xlab=xe, ylab=ye)
    draw_heat(ax_tp, TPm, TPs, xl, yl, "Top-removal AUC (낮을수록 좋음)", "viridis", tlo, thi, anchor,
              xlab=xe, ylab=ye, fmt="{:.3f}", sfmt="{:.3f}")
    draw_heat(ax_bt, BTm, BTs, xl, yl, "Bottom-removal AUC (높을수록 좋음)", "viridis", blo, bhi, anchor,
              xlab=xe, ylab=ye, fmt="{:.3f}", sfmt="{:.3f}")
    hi = lv.index(cross["lam0"]) if cross["lam0"] in lv else None
    vj = ev.index(cross["eps0"]) if cross["eps0"] in ev else None
    arms = []
    if hi is not None:
        arms.append(("Spearman — ε (λ 고정)", SPm[hi, :], SPs[hi, :], xl, (0, 1), "{:.2f}"))
        arms.append(("Top-removal AUC — ε (λ 고정)", TPm[hi, :], TPs[hi, :], xl, (tlo, thi), "{:.3f}"))
    if vj is not None:
        arms.append(("Spearman — λ (ε 고정)", SPm[:, vj], SPs[:, vj], yl, (0, 1), "{:.2f}"))
        arms.append(("Top-removal AUC — λ (ε 고정)", TPm[:, vj], TPs[:, vj], yl, (tlo, thi), "{:.3f}"))
    for idx, (ttl, mm, ss, labs, (vlo, vhi), f) in enumerate(arms[:4]):
        ax = fig.add_subplot(gs[1 + idx // 2, (0 if idx % 2 == 0 else 4):(4 if idx % 2 == 0 else 8)])
        draw_strip(ax, list(mm), list(ss), labs, ttl, "viridis", vlo, vhi, fmt=f)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def rem_curve_seeds(a, method, lam, eps, strat):
    """seed 평균 removal 곡선 + 그 파일의 pct 목록."""
    arrs, pcts = [], None
    for s in a.seeds:
        d = parse_removal(removal_path(a, method, lam, eps, s))
        if strat in d:
            arrs.append(d[strat])
            if pcts is None: pcts = d.get("_pcts")
    if not arrs: return None, None
    L = min(len(x) for x in arrs)
    return np.vstack([x[:L] for x in arrs]).mean(0), (list(pcts)[:L] if pcts else list(range(L)))

def inv_rem_curve(a, strat):
    arrs, pcts = [], None
    for s in a.seeds:
        d = parse_removal(inv_removal_path(a, s))
        if strat in d:
            arrs.append(d[strat])
            if pcts is None: pcts = d.get("_pcts")
    if not arrs: return None, None
    L = min(len(x) for x in arrs)
    return np.vstack([x[:L] for x in arrs]).mean(0), (list(pcts)[:L] if pcts else list(range(L)))

# ============ 페이지: removal 곡선 (모든 sweep 오버레이, wld/selection 방식) ============
def page_removal_curves(pdf, a, strat, axis):
    label = {"top_removal": "top 제거(고가치)", "bottom_removal": "bottom 제거(저가치)"}[strat]
    axsym = "ε" if axis == "eps" else "λ"
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"wspace": .6})
    fig.suptitle(f"[{a.dataset}] Data removal ({label}) — acc vs 제거%, 모든 {axsym} sweep 오버레이 "
                 f"(seed 평균)", fontsize=13, fontweight="bold")
    invc, invp = inv_rem_curve(a, strat)
    rndc, rndp = inv_rem_curve(a, "random")   # random 은 inv 파일에서 baseline 으로
    for ax, method in zip(axes, a.methods):
        m = METH[method]; cross = build_grid(a, method)
        if cross is None:
            ax.set_title(m["kor"]); ax.text(.5, .5, "데이터 없음", ha="center", va="center",
                                            transform=ax.transAxes); continue
        if axis == "eps":
            counts = {}
            for (l, e) in cross["cells"]: counts[l] = counts.get(l, 0) + 1
            fixL = cross["lam0"] if any(cross["close"](l, cross["lam0"]) for l in counts) \
                else (max(counts, key=counts.get) if counts else cross["lam0"])
            vals = sorted({e for (l, e) in cross["cells"] if cross["close"](l, fixL)})
            pairs = [(v, fixL, v) for v in vals]; fixed = f"λ={_fmt_eps(fixL)}"; tagname = f"{m['eps']} (ε)"
        else:
            counts = {}
            for (l, e) in cross["cells"]: counts[e] = counts.get(e, 0) + 1
            fixE = cross["eps0"] if any(cross["close"](e, cross["eps0"]) for e in counts) \
                else (max(counts, key=counts.get) if counts else cross["eps0"])
            vals = sorted({l for (l, e) in cross["cells"] if cross["close"](e, fixE)})
            pairs = [(v, v, fixE) for v in vals]; fixed = f"ε={_fmt_eps(fixE)}"; tagname = "lam (λ)"
        ax.set_title(f"{m['kor']} — {axsym} sweep ({fixed} 고정)", fontsize=11)
        ax.set_xlabel("제거 %", fontsize=10); ax.set_ylabel("val acc (seed 평균)", fontsize=10)
        cmap = plt.get_cmap("tab20")
        for idx, (lab, lam, eps) in enumerate(pairs):
            mc, mp = rem_curve_seeds(a, method, lam, eps, strat)
            if mc is not None:
                ax.plot(mp, mc, "-", color=cmap(idx % 20), lw=1.5, label=f"{axsym}={_fmt_eps(lab)}")
        if rndc is not None:
            ax.plot(rndp, rndc, ":", color="gray", lw=1.3, alpha=.85, label="random (inv)", zorder=1)
        if invc is not None:
            ax.plot(invp, invc, "--", color="black", lw=1.9, label="inv (exact)", zorder=4)
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, .5),
                  title=tagname, title_fontsize=8, ncol=1, handlelength=1.6)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _auc(curve, pcts, cap):
    if cap is None: return float(np.mean(curve))
    idx = [i for i, p in enumerate(pcts) if p <= cap]
    return float(np.mean([curve[i] for i in idx])) if idx else float("nan")

def _trunc(curve, pcts, cap):
    if cap is None: return list(pcts), list(curve)
    idx = [i for i, p in enumerate(pcts) if p <= cap]
    return [pcts[i] for i in idx], [curve[i] for i in idx]

def _best_cell(a, method, strat, cap, better):
    """히트맵 모든 셀 중 AUC(cap 기준) best 인 (lam,eps,curve,pcts,auc)."""
    cross = build_grid(a, method)
    if cross is None: return None
    best, bauc = None, None
    for (l, e) in cross["cells"]:
        curve, pcts = rem_curve_seeds(a, method, l, e, strat)
        if curve is None: continue
        auc = _auc(curve, pcts, cap)
        if np.isnan(auc): continue
        if bauc is None or (better == "min" and auc < bauc) or (better == "max" and auc > bauc):
            bauc, best = auc, (l, e, curve, pcts, auc)
    return best

def page_best_compare(pdf, a, strat, better):
    """좌: 전체(0~99%) AUC best / 우: 0~K% AUC best.  각 4곡선: random, inv, eigen-best, nys-best."""
    label = {"top_removal": "top 제거(고가치)", "bottom_removal": "bottom 제거(저가치)"}[strat]
    arrow = "낮을수록" if better == "min" else "높을수록"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"wspace": .28})
    fig.suptitle(f"[{a.dataset}] Removal best-config 비교 ({label}) — random/inv vs 각 method의 최적 sweep "
                 f"(AUC {arrow} 좋음), seeds={a.seeds}", fontsize=12.5, fontweight="bold")
    invc, invp = inv_rem_curve(a, strat)
    rndc, rndp = inv_rem_curve(a, "random")
    mcol = {"eigen": "#1f77b4", "nystrom": "#d62728"}
    for ax, cap, ttl in [(axes[0], None, "AUC 전체 (제거 0~99%)"),
                         (axes[1], a.auc_k, f"AUC 저-k (제거 0~{a.auc_k}%)")]:
        if rndc is not None:
            xp, yp = _trunc(rndc, rndp, cap); ax.plot(xp, yp, ":", color="gray", lw=1.4, alpha=.85, label="random (inv)")
        if invc is not None:
            xp, yp = _trunc(invc, invp, cap); ax.plot(xp, yp, "--", color="black", lw=2.0, label="inv (exact)")
        for method in a.methods:
            b = _best_cell(a, method, strat, cap, better)
            if b is None: continue
            l, e, curve, pcts, auc = b
            xp, yp = _trunc(curve, pcts, cap)
            ax.plot(xp, yp, "-", color=mcol.get(method, None), lw=1.9,
                    label=f"{METH[method]['kor']} best (λ={_fmt_eps(l)}, ε={_fmt_eps(e)}) AUC={auc:.3f}")
        ax.set_title(ttl, fontsize=11); ax.set_xlabel("제거 %", fontsize=10)
        ax.set_ylabel("val acc (seed 평균)", fontsize=10); ax.grid(True, alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, .95]); pdf.savefig(fig); plt.close(fig)


def main():
    a = parse_args()
    if a.val is None:
        a.val = R.autodetect_val(a, a.methods[0]) or R.autodetect_val(a, a.methods[-1])
    if a.val is None:
        print("[err] val 자동감지 실패 — --val 지정"); return
    print(f"[cfg] {a.dataset} seeds={a.seeds} n={a.num_train} val={a.val} tmc={a.tmc}")
    INV = {s: load_sv(inv_sv_path(a, s)) for s in a.seeds}
    out = a.out or f"./jitter_exp/report_removal_{a.dataset}_n{a.num_train}_tmc{a.tmc}.pdf"
    with PdfPages(out) as pdf:
        for method in a.methods:
            page_grid(pdf, a, method, INV)
        for strat in ["top_removal", "bottom_removal"]:
            page_removal_curves(pdf, a, strat, "eps")
            page_removal_curves(pdf, a, strat, "lam")
        page_best_compare(pdf, a, "top_removal", better="min")
        page_best_compare(pdf, a, "bottom_removal", better="max")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()

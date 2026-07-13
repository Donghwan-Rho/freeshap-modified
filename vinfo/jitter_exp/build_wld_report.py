# -*- coding: utf-8 -*-
"""
Wrong-Label-Detection(WLD) 리포트 PDF — build_report.py 의 십자(cross) 구조를 그대로 쓰되
downstream 을 "selection acc" → "poison detection" 으로 교체한 버전. multi-seed(mean±std).

핵심 차이 (build_report 대비):
  * SV 는 poisoned 데이터에서 계산됨 (파일명에 _poison{pct}_ps{seed}).
  * fidelity = corr(approx poisoned-SV, inv poisoned-SV)  (측정 방식은 동일).
  * downstream = detection.txt 의 "detection rates"(최하위 SV 부터 poison recall, k=1~100%).
      - Detection AUC(1~100%) 와 Detection AUC(1~30%) 두 요약을 2D 히트맵으로.
  * sweep 은 십자: nystrom(nyseps×λ) / eigen(eigeps×λ) — method 별 페이지 분리.

페이지 (데이터셋 1개):
  [Fidelity+Detection]  method마다 1p: 상단 Spearman/Pearson/DetAUC100/DetAUC30 십자 4개 + 하단 arm strip
  [Detection 곡선] eps/λ 각 1p: recall vs k%(1~100) 오버레이 (+ inv, random 기준선)
  [Fidelity–Detection coupling] eps/λ 각 1p: x=Spearman, y=DetAUC (k100/k30), method별

사용:
  cd /extdata1/donghwan/freeshap/vinfo
  python jitter_exp/build_wld_report.py --dataset qqp --seeds 2024 2025 2026 \
      --num_train 2000 --tmc 500 --poison 10
"""
import os, sys, re, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_report import (  # noqa: E402  (generic 헬퍼 재사용)
    draw_heat, draw_strip, cross_matrices, align, _rank, _mean_std_n,
    load_sv, _fmt_eps, _lam_tag,
)

# WLD 파일명 태그: rank=nys/eig, lam='lam'(공통), eps=nyseps/eigeps
METHW = {
    "nystrom": dict(rank="nys", eps="nyseps", eps_anchor_default=1e1,  lam_anchor_default=1e-2, kor="Nyström"),
    "eigen":   dict(rank="eig", eps="eigeps", eps_anchor_default=1e-8, lam_anchor_default=1e-2, kor="Eigen"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--methods", type=str, nargs="+", default=["nystrom", "eigen"])
    p.add_argument("--model", type=str, default="bert")
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--val", type=int, default=None, help="없으면 파일명에서 자동 감지")
    p.add_argument("--tmc", type=int, default=500)
    p.add_argument("--rank", type=float, default=20)
    p.add_argument("--invlam", type=str, default="1e-06")
    p.add_argument("--poison", type=int, default=10, help="poison 퍼센트")
    p.add_argument("--eps_anchor", type=float, default=None)
    p.add_argument("--lam_anchor", type=float, default=None)
    p.add_argument("--cross_only", action="store_true",
                   help="설정 시 anchor 십자(arm)만. 기본은 폴더의 모든 (lam,eps) 조합을 격자로 표시.")
    p.add_argument("--det_k30", type=int, default=30, help="Detection 저-k 요약 상한 (k≤N%%)")
    p.add_argument("--tail_pct", type=int, default=10,
                   help="bottom-overlap 기준 최하위 %% (detection 꼬리 fidelity). 기본 poison%%와 맞춤=10.")
    p.add_argument("--res_root", type=str, default="./jitter_exp/res")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


# ============ 파일명 (WLD 스킴) ============
def _poison_tag(a, seed):
    # task_wrong_label_detection: poison_seed 기본=seed(≠2023) → _ps{seed} 포함
    return f"_poison{a.poison}_ps{seed}"

def _rank_g(v):
    return f"{float(v):g}"          # 20.0 → '20' (WLD 는 nys20/eig20)

def _approx_stem(a, method, lam, eps, seed):
    m = METHW[method]
    return (f"{a.model}_seed{seed}_num{a.num_train}_val{a.val}"
            f"_{m['rank']}{_rank_g(a.rank)}_lam{_lam_tag(lam)}_{m['eps']}{_fmt_eps(eps)}"
            f"_cholesky_float32_signFalse_earlystopTrue_tmc{a.tmc}{_poison_tag(a, seed)}")

def sv_path(a, method, lam, eps, seed):
    return os.path.join(a.res_root, "shapley", a.dataset, method, _approx_stem(a, method, lam, eps, seed) + ".pkl")

def det_path(a, method, lam, eps, seed):
    return os.path.join(a.res_root, "wrong_label_detection", a.dataset, method,
                        "predictions", _approx_stem(a, method, lam, eps, seed) + "_detection.txt")

def _inv_stem(a, seed):
    return (f"{a.model}_seed{seed}_num{a.num_train}_val{a.val}"
            f"_lam{a.invlam}_signFalse_earlystopTrue_tmc{a.tmc}{_poison_tag(a, seed)}")

def inv_sv_path(a, seed):
    return os.path.join(a.res_root, "shapley", a.dataset, "inv", _inv_stem(a, seed) + ".pkl")

def inv_det_path(a, seed):
    return os.path.join(a.res_root, "wrong_label_detection", a.dataset, "inv",
                        "predictions", _inv_stem(a, seed) + "_detection.txt")


def autodetect_val(a, method):
    for s in a.seeds:
        for p in glob.glob(os.path.join(a.res_root, "shapley", a.dataset, method, f"*_seed{s}_*poison{a.poison}_*.pkl")):
            m = re.search(r"_val(\d+)_", os.path.basename(p))
            if m:
                return int(m.group(1))
    return None

def scan_sweeps(a, method):
    """(lam, eps) 조합 — poison pkl 파일명에서 자동 감지."""
    m = METHW[method]; combos = set()
    for p in glob.glob(os.path.join(a.res_root, "shapley", a.dataset, method, f"*poison{a.poison}_*.pkl")):
        fn = os.path.basename(p)
        mr = re.search(rf"_{m['rank']}([0-9.]+)_", fn)
        ml = re.search(r"_lam([0-9.e+-]+)_", fn)
        me = re.search(rf"_{m['eps']}([0-9.e+-]+)_", fn)
        ms = re.search(r"_seed(\d+)_", fn)
        if not (mr and ml and me and ms): continue
        if int(ms.group(1)) not in a.seeds: continue
        if abs(float(mr.group(1)) - float(a.rank)) > 1e-9: continue
        combos.add((float(ml.group(1)), float(me.group(1))))
    return combos

def build_cross(a, method):
    m = METHW[method]
    eps0 = a.eps_anchor if a.eps_anchor is not None else m["eps_anchor_default"]
    lam0 = a.lam_anchor if a.lam_anchor is not None else m["lam_anchor_default"]
    combos = scan_sweeps(a, method)
    if not combos: return None
    def close(x, y): return abs(x - y) < abs(y) * 1e-6 + 1e-30
    if getattr(a, "cross_only", False):
        eps_at = sorted({e for (l, e) in combos if close(l, lam0)})
        lam_at = sorted({l for (l, e) in combos if close(e, eps0)})
        eps_vals = sorted(set(eps_at) | {eps0})
        lam_vals = sorted(set(lam_at) | {lam0})
        cells = {(l, e) for (l, e) in combos if close(l, lam0) or close(e, eps0)}
    else:
        # 전체 격자: 폴더에 존재하는 모든 조합 (십자/1자/사각형 자동 대응)
        eps_vals = sorted({e for (l, e) in combos})
        lam_vals = sorted({l for (l, e) in combos})
        cells = set(combos)
    return dict(eps_vals=eps_vals, lam_vals=lam_vals, eps0=eps0, lam0=lam0, cells=cells, close=close)


# ============ detection 파서 ============
_DET_CACHE = {}
def parse_detection(path):
    """detection.txt → (det[100], rnd[100]) recall 배열 (0~1). 없으면 (None,None)."""
    if path in _DET_CACHE: return _DET_CACHE[path]
    if not os.path.exists(path):
        _DET_CACHE[path] = (None, None); return None, None
    import ast
    lines = open(path).read().splitlines()
    det = rnd = None; want = None
    for ln in lines:
        low = ln.strip().lower()
        if low.startswith("detection rates"): want = "det"; continue
        if low.startswith("random baseline"): want = "rnd"; continue
        s = ln.strip()
        if s.startswith("[") and want is not None:
            try: arr = np.array(ast.literal_eval(s), dtype=float) / 10000.0
            except Exception: arr = None
            if want == "det": det = arr
            elif want == "rnd": rnd = arr
            want = None
    _DET_CACHE[path] = (det, rnd); return det, rnd


# ============ INV(poisoned) SV 로드 ============
def load_inv(a):
    return {s: load_sv(inv_sv_path(a, s)) for s in a.seeds}


# ============ seed 집계 ============
def fid_seeds(a, method, lam, eps, INV, which):
    """poisoned SV fidelity: 'sp'(Spearman)/'pe'(Pearson) → (mean,std,n)."""
    out = []
    for s in a.seeds:
        asv, asi = load_sv(sv_path(a, method, lam, eps, s))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        av, bv = align(asv, asi, isv, isi)
        if len(av) < 10: continue
        if which == "sp": out.append(float(np.corrcoef(_rank(av), _rank(bv))[0, 1]))
        elif which == "pe": out.append(float(np.corrcoef(av, bv)[0, 1]))
        elif which == "bov": out.append(_bottom_overlap(av, bv, a.tail_pct))
    return _mean_std_n(out)

def _bottom_overlap(av, bv, tail_pct):
    """최하위 tail_pct% SV 집합의 겹침 비율 (detection 꼬리 fidelity)."""
    k = max(1, int(len(av) * tail_pct / 100.0))
    return len(set(np.argsort(av)[:k].tolist()) & set(np.argsort(bv)[:k].tolist())) / k

def _fid_one(a, method, lam, eps, seed, INV, which):
    asv, asi = load_sv(sv_path(a, method, lam, eps, seed))
    isv, isi = INV.get(seed, (None, None))
    if asv is None or isv is None: return None
    av, bv = align(asv, asi, isv, isi)
    if len(av) < 10: return None
    if which == "sp": return float(np.corrcoef(_rank(av), _rank(bv))[0, 1])
    if which == "pe": return float(np.corrcoef(av, bv)[0, 1])
    if which == "bov": return _bottom_overlap(av, bv, a.tail_pct)
    return None

def det_summary_seeds(a, method, lam, eps, k):
    """detection recall 곡선의 k%까지 평균(≈AUC) → (mean,std,n).  k=100/30."""
    out = []
    for s in a.seeds:
        det, _ = parse_detection(det_path(a, method, lam, eps, s))
        if det is not None and len(det) >= 1:
            out.append(float(np.mean(det[:k])))
    return _mean_std_n(out)

def _det_one(a, method, lam, eps, seed, k):
    det, _ = parse_detection(det_path(a, method, lam, eps, seed))
    if det is None: return None
    return float(np.mean(det[:k]))

def det_curve_seeds(a, method, lam, eps):
    """seed별 detection 곡선 → (mean[100], std[100], rnd_mean, n)."""
    ds, rs = [], []
    for s in a.seeds:
        det, rnd = parse_detection(det_path(a, method, lam, eps, s))
        if det is not None: ds.append(det[:100])
        if rnd is not None: rs.append(rnd[:100])
    if not ds: return None, None, None, 0
    L = min(len(d) for d in ds); D = np.vstack([d[:L] for d in ds])
    R = np.vstack([r[:L] for r in rs]) if rs else None
    return D.mean(0), D.std(0), (R.mean(0) if R is not None else None), len(ds)

def inv_det_curve_seeds(a):
    ds = []
    for s in a.seeds:
        det, _ = parse_detection(inv_det_path(a, s))
        if det is not None: ds.append(det[:100])
    if not ds: return None, 0
    L = min(len(d) for d in ds); return np.vstack([d[:L] for d in ds]).mean(0), len(ds)

def inv_det_summary(a, k):
    out = []
    for s in a.seeds:
        det, _ = parse_detection(inv_det_path(a, s))
        if det is not None: out.append(float(np.mean(det[:k])))
    return _mean_std_n(out)


def _arm_cells(cross, axis):
    if axis == "eps":
        return [(cross["lam0"], e) for e in
                sorted({e for (l, e) in cross["cells"] if cross["close"](l, cross["lam0"])})]
    return [(l, cross["eps0"]) for l in
            sorted({l for (l, e) in cross["cells"] if cross["close"](e, cross["eps0"])})]


# ============ 페이지: fidelity + detection 십자 ============
def page_fidelity(pdf, a, method, INV):
    m = METHW[method]; cross = build_cross(a, method)
    if cross is None: return
    SPm, SPs, ev, lv = cross_matrices(a, method, cross, lambda l, e: fid_seeds(a, method, l, e, INV, "sp"))
    PEm, PEs, _, _   = cross_matrices(a, method, cross, lambda l, e: fid_seeds(a, method, l, e, INV, "pe"))
    BOm, BOs, _, _   = cross_matrices(a, method, cross, lambda l, e: fid_seeds(a, method, l, e, INV, "bov"))
    D1m, D1s, _, _   = cross_matrices(a, method, cross, lambda l, e: det_summary_seeds(a, method, l, e, 100))
    D3m, D3s, _, _   = cross_matrices(a, method, cross, lambda l, e: det_summary_seeds(a, method, l, e, a.det_k30))
    xl = [_fmt_eps(e) for e in ev]; yl = [_fmt_eps(l) for l in lv]
    ai = lv.index(cross["lam0"]) if cross["lam0"] in lv else None
    aj = ev.index(cross["eps0"]) if cross["eps0"] in ev else None
    anchor = (ai, aj)
    def _rng(M):
        f = np.asarray(M)[np.isfinite(M)]
        if not f.size: return 0.0, 1.0
        lo, hi = float(f.min()), float(f.max())
        return (lo, hi + 1e-6) if hi - lo < 1e-6 else (lo, hi)
    bolo, bohi = _rng(BOm); d1lo, d1hi = _rng(D1m); d3lo, d3hi = _rng(D3m)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 6, height_ratios=[3, 3, 1, 1], hspace=.55, wspace=.7)
    fig.suptitle(f"[{a.dataset}] {m['kor']} — Fidelity(poisoned SV) + Poison Detection AUC "
                 f"| rank/d={a.rank:g}%, poison={a.poison}%, tail={a.tail_pct}%, tmc={a.tmc}, seeds={a.seeds}",
                 fontsize=13, fontweight="bold")
    xe, ye = f"{m['eps']} (ε)", "lam (λ)"
    # 행0: fidelity 3종
    ax_sp = fig.add_subplot(gs[0, 0:2]); ax_pe = fig.add_subplot(gs[0, 2:4]); ax_bo = fig.add_subplot(gs[0, 4:6])
    draw_heat(ax_sp, SPm, SPs, xl, yl, "Spearman ρ (mean±std)", "viridis", 0.0, 1.0, anchor, xlab=xe, ylab=ye)
    draw_heat(ax_pe, PEm, PEs, xl, yl, "Pearson r (mean±std)", "viridis", 0.0, 1.0, anchor, xlab=xe, ylab=ye)
    draw_heat(ax_bo, BOm, BOs, xl, yl, f"Bottom-{a.tail_pct}% overlap (꼬리 fidelity)", "viridis", bolo, bohi, anchor,
              xlab=xe, ylab=ye, fmt="{:.2f}")
    # 행1: detection 2종
    ax_d1 = fig.add_subplot(gs[1, 0:3]); ax_d3 = fig.add_subplot(gs[1, 3:6])
    draw_heat(ax_d1, D1m, D1s, xl, yl, "Detection AUC k1–100% (mean±std)", "viridis", d1lo, d1hi, anchor,
              xlab=xe, ylab=ye, fmt="{:.3f}", sfmt="{:.3f}")
    draw_heat(ax_d3, D3m, D3s, xl, yl, f"Detection AUC k1–{a.det_k30}% (mean±std)", "viridis", d3lo, d3hi, anchor,
              xlab=xe, ylab=ye, fmt="{:.3f}", sfmt="{:.3f}")
    # 행2-3: arm strip (꼬리 fidelity vs detection 를 나란히)
    hi = lv.index(cross["lam0"]) if cross["lam0"] in lv else None
    vj = ev.index(cross["eps0"]) if cross["eps0"] in ev else None
    arms = []
    if hi is not None:
        arms.append((f"Bottom-{a.tail_pct}%ov — ε sweep", BOm[hi, :], BOs[hi, :], xl, "viridis", (bolo, bohi), "{:.2f}"))
        arms.append(("Detection AUC(100%) — ε sweep", D1m[hi, :], D1s[hi, :], xl, "viridis", (d1lo, d1hi), "{:.3f}"))
    if vj is not None:
        arms.append((f"Bottom-{a.tail_pct}%ov — λ sweep", BOm[:, vj], BOs[:, vj], yl, "viridis", (bolo, bohi), "{:.2f}"))
        arms.append(("Detection AUC(100%) — λ sweep", D1m[:, vj], D1s[:, vj], yl, "viridis", (d1lo, d1hi), "{:.3f}"))
    for idx, (ttl, mm, ss, labs, cmap, (vlo, vhi), f) in enumerate(arms[:4]):
        ax = fig.add_subplot(gs[2 + idx // 2, (0 if idx % 2 == 0 else 3):(3 if idx % 2 == 0 else 6)])
        draw_strip(ax, list(mm), list(ss), labs, ttl, cmap, vlo, vhi, fmt=f)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 페이지: detection 곡선 (recall vs k%) ============
def page_curves(pdf, a, axis="eps"):
    ks = list(range(1, 101))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"wspace": .6})
    axsym = "ε" if axis == "eps" else "λ"
    fig.suptitle(f"[{a.dataset}] Poison Detection 곡선 — recall vs k%(최하위 SV부터), 모든 {axsym} sweep "
                 f"(poison={a.poison}%, seed 평균)", fontsize=13, fontweight="bold")
    invc, _ = inv_det_curve_seeds(a)
    for ax, method in zip(axes, a.methods):
        m = METHW[method]; cross = build_cross(a, method)
        if cross is None:
            ax.set_title(m["kor"]); ax.text(.5, .5, "데이터 없음", ha="center", va="center",
                                            transform=ax.transAxes); continue
        if axis == "eps":
            counts = {}
            for (l, e) in cross["cells"]: counts[l] = counts.get(l, 0) + 1
            fixL = cross["lam0"] if any(cross["close"](l, cross["lam0"]) for l in counts) \
                else (max(counts, key=counts.get) if counts else cross["lam0"])
            vals = sorted({e for (l, e) in cross["cells"] if cross["close"](l, fixL)})
            pairs = [(v, fixL, v) for v in vals]; fixed = f"λ={_fmt_eps(fixL)}"
            tagname = f"{m['eps']} (ε)"
        else:
            counts = {}
            for (l, e) in cross["cells"]: counts[e] = counts.get(e, 0) + 1
            fixE = cross["eps0"] if any(cross["close"](e, cross["eps0"]) for e in counts) \
                else (max(counts, key=counts.get) if counts else cross["eps0"])
            vals = sorted({l for (l, e) in cross["cells"] if cross["close"](e, fixE)})
            pairs = [(v, v, fixE) for v in vals]; fixed = f"ε={_fmt_eps(fixE)}"
            tagname = "lam (λ)"
        ax.set_title(f"{m['kor']} — {axsym} sweep ({fixed} 고정)", fontsize=11)
        ax.set_xlabel("k% (최하위 SV부터 검사)", fontsize=10); ax.set_ylabel("poison recall (seed 평균)", fontsize=10)
        cmap = plt.get_cmap("tab20"); ref_drawn = False
        for idx, (lab, lam, eps) in enumerate(pairs):
            md, sd, mr, n = det_curve_seeds(a, method, lam, eps)
            if md is not None:
                ax.plot(ks[:len(md)], md, "-", color=cmap(idx % 20), lw=1.6, label=f"{axsym}={_fmt_eps(lab)}")
            if not ref_drawn and mr is not None:
                ax.plot(ks[:len(mr)], mr, ":", color="gray", lw=1.2, alpha=.8, label="random", zorder=1)
                ref_drawn = True
        if invc is not None:
            ax.plot(ks[:len(invc)], invc, "--", color="black", lw=1.8, label="inv (exact)", zorder=4)
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, .5),
                  title=tagname, title_fontsize=8.5, ncol=1, handlelength=1.6)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 페이지: fidelity–detection coupling ============
def page_coupling(pdf, a, INV, axis="eps"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axsym = "ε" if axis == "eps" else "λ"
    fig.suptitle(f"[{a.dataset}] Fidelity–Detection coupling — y=Detection AUC(k1–100%) "
                 f"({axsym} sweep, seed 평균).  위=global Spearman, 아래=꼬리 Bottom-{a.tail_pct}%overlap",
                 fontsize=13, fontweight="bold")
    fidrows = [("sp", "Spearman ρ (global)"), ("bov", f"Bottom-{a.tail_pct}% overlap (꼬리)")]
    for ci, method in enumerate(a.methods):
        m = METHW[method]; cross = build_cross(a, method)
        for ri, (fk, fname) in enumerate(fidrows):
            ax = axes[ri, ci]; k = 100
            ax.set_title(f"{m['kor']} — x={fname}", fontsize=11)
            ax.set_xlabel(fname + " (poisoned SV vs inv)", fontsize=9)
            ax.set_ylabel(f"Detection AUC(k1–{k}%)", fontsize=9)
            if cross is None:
                ax.text(.5, .5, "데이터 없음", ha="center", va="center", transform=ax.transAxes); continue
            cells = _arm_cells(cross, axis) if a.cross_only else sorted(cross["cells"])
            xs, ys, labs, cvals = [], [], [], []
            for (l, e) in cells:
                fm, _, _ = fid_seeds(a, method, l, e, INV, fk)
                dm, _, _ = det_summary_seeds(a, method, l, e, k)
                if np.isnan(fm) or np.isnan(dm): continue
                xs.append(fm); ys.append(dm)
                labs.append(_fmt_eps(e if axis == "eps" else l))
                cvals.append(np.log10(e if axis == "eps" else l))
            if len(xs) >= 2:
                xs, ys = np.array(xs), np.array(ys)
                if a.cross_only:  # arm 은 순서가 있으니 연결선, 격자는 산점만
                    ax.plot(xs, ys, "-", color="0.7", lw=.8, zorder=1)
                sc = ax.scatter(xs, ys, s=60, c=cvals, cmap="plasma", zorder=3, edgecolors="k", linewidths=.6)
                cb = plt.colorbar(sc, ax=ax, fraction=.046, pad=.02)
                cb.set_label(f"log10({'ε' if axis == 'eps' else 'λ'})", fontsize=7); cb.ax.tick_params(labelsize=6)
                for x, y, t in zip(xs, ys, labs):
                    ax.annotate(t, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points", color="0.3")
                if len(xs) >= 3 and xs.std() > 1e-9:
                    r = float(np.corrcoef(xs, ys)[0, 1])
                    ax.text(.03, .97, f"r={r:+.2f} (n={len(xs)})", transform=ax.transAxes, va="top",
                            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=.9))
            # inv 기준선
            im, _, _ = inv_det_summary(a, k)
            if not np.isnan(im):
                ax.axhline(im, ls="--", color="black", lw=1.2, alpha=.7)
                ax.text(.99, im, " inv", va="bottom", ha="right", fontsize=7, color="black", transform=ax.get_yaxis_transform())
            ax.grid(True, alpha=.3)
    fig.tight_layout(rect=[0, 0, 1, .96]); pdf.savefig(fig); plt.close(fig)


def main():
    a = parse_args()
    if a.val is None:
        for meth in a.methods:
            v = autodetect_val(a, meth)
            if v: a.val = v; break
    if a.val is None:
        print("[err] val 자동 감지 실패 — --val 지정 필요"); return
    print(f"[cfg] dataset={a.dataset} seeds={a.seeds} n={a.num_train} val={a.val} "
          f"tmc={a.tmc} poison={a.poison}% rank={a.rank:g}")
    INV = load_inv(a)
    print(f"[inv] poisoned inv SV 존재 seed: {[s for s in a.seeds if INV.get(s,(None,))[0] is not None]}")
    out = a.out or f"./jitter_exp/report_wld_{a.dataset}_n{a.num_train}_tmc{a.tmc}_poison{a.poison}.pdf"
    with PdfPages(out) as pdf:
        for meth in a.methods:
            page_fidelity(pdf, a, meth, INV)
        page_curves(pdf, a, "eps")
        page_curves(pdf, a, "lam")
        page_coupling(pdf, a, INV, "eps")
        page_coupling(pdf, a, INV, "lam")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()

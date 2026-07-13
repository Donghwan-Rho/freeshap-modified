# -*- coding: utf-8 -*-
"""
jitter_exp canonical 결과(res/)로 리포트 PDF 생성 — multi-seed (mean±std) 지원.

구조 (데이터셋 1개 기준):
  [Fidelity]  method마다 1페이지:
      상단  = Spearman 십자(좌) + Pearson 십자(우)  (셀=seed 평균, 작게 ±std)
      하단  = 위 두 십자의 4개 arm 을 각각 가로 1D 히트맵
  [Selection 곡선 (eps sweep)]  1페이지: 좌 nys / 우 eigen — 모든 eps, seed 평균 선 (band 없음)
  [Selection 곡선 (λ sweep)]    1페이지: 위와 동일하되 λ sweep
  [Selection sweep]  1페이지: acc vs eps (k=20/50/70), mean ± std band
  [Fidelity–Acc coupling]  1페이지: x=Spearman, y=acc(AUC), eps 파라미터 + error bar
  [Selection 십자]  method마다 1페이지: acc@30% / acc@70% (셀=평균±std)

특징:
  * dataset/num_train/tmc/seeds/model/rank/anchor 전부 argparse
  * eps/lam sweep 값은 res/ 파일명에서 자동 감지
  * seed 는 있는 것만 평균 (부분만 있어도 OK), 없는 셀/곡선은 빈칸/생략
  * 지표: predictions 의 "inv mode / top:" (근사 SV 랭킹을 INV 커널로 평가)
  * metric 은 seed별로 계산 후 평균 (raw SV 는 seed 간 안 섞음)

사용 예:
  cd /extdata1/donghwan/freeshap/vinfo
  python jitter_exp/build_report.py --dataset qqp --seeds 2024 2025 2026 --num_train 2000 --tmc 500
"""
import os, re, glob, pickle, argparse, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm

_FP = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
if os.path.exists(_FP):
    fm.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FP).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"   # 수식($...$)은 Computer Modern 로 렌더

METH = {
    "nystrom": dict(rank="nys", lam="nyslam", eps="nyseps",
                    eps_anchor_default=1e1,  lam_anchor_default=1e-2, kor="Nyström"),
    "eigen":   dict(rank="eig", lam="eiglam", eps="eigeps",
                    eps_anchor_default=1e-8, lam_anchor_default=1e-2, kor="Eigen"),
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
    p.add_argument("--eps_anchor", type=float, default=None)
    p.add_argument("--lam_anchor", type=float, default=None)
    p.add_argument("--overlap_pct", type=int, default=20, help="top-shap overlap 기준 상위 %%")
    p.add_argument("--acc_k30", type=int, default=30, help="acc 저-k 요약 상한 (k≤N%%)")
    p.add_argument("--sel_ks", type=int, nargs="+", default=[30, 70])
    p.add_argument("--sweep_ks", type=int, nargs="+", default=[20, 50, 70])
    p.add_argument("--res_root", type=str, default="./jitter_exp/res")
    p.add_argument("--inv_root", type=str, default="./freeshap_res")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


# ============ 파일명 유틸 ============
def _fmt_eps(v):
    return f"{v:.0e}".replace("e+0", "e+").replace("e-0", "e-")
def _lam_tag(v):
    return f"{v:.0e}"
def _rank_tag(v):
    return f"{float(v)}"

def sv_path(a, method, lam, eps, seed):
    m = METH[method]
    fn = (f"{a.model}_seed{seed}_num{a.num_train}_val{a.val}"
          f"_{m['rank']}{_rank_tag(a.rank)}_{m['lam']}{_lam_tag(lam)}_{m['eps']}{_fmt_eps(eps)}"
          f"_invlam{a.invlam}_cholesky_float32_signFalse_earlystopTrue_tmc{a.tmc}.pkl")
    return os.path.join(a.res_root, "shapley", a.dataset, method, fn)

def pred_path(a, method, lam, eps, seed):
    m = METH[method]
    fn = (f"{a.model}_seed{seed}_num{a.num_train}_val{a.val}"
          f"_{m['rank']}{_rank_tag(a.rank)}_{m['lam']}{_lam_tag(lam)}_{m['eps']}{_fmt_eps(eps)}"
          f"_invlam{a.invlam}_cholesky_float32_signFalse_earlystopTrue_tmc{a.tmc}_predictions.txt")
    return os.path.join(a.res_root, "data_selection", a.dataset, method, "predictions", fn)

def inv_sv_path(a, seed):
    fn = (f"{a.model}_seed{seed}_num{a.num_train}_val{a.val}"
          f"_lam{a.invlam}_signFalse_earlystopTrue_tmc{a.tmc}.pkl")
    return os.path.join(a.inv_root, "shapley", a.dataset, "inv", fn)


def autodetect_val(a, method):
    for s in a.seeds:
        for p in glob.glob(os.path.join(a.res_root, "shapley", a.dataset, method, f"*_seed{s}_*.pkl")):
            m = re.search(r"_val(\d+)_", os.path.basename(p))
            if m:
                return int(m.group(1))
    return None

def scan_sweeps(a, method):
    """(lam, eps) 조합 — 여러 seed 중 하나라도 있으면 포함."""
    m = METH[method]; combos = set()
    for p in glob.glob(os.path.join(a.res_root, "shapley", a.dataset, method, "*.pkl")):
        fn = os.path.basename(p)
        mr = re.search(rf"_{m['rank']}([0-9.]+)_", fn)
        ml = re.search(rf"_{m['lam']}([0-9.e+-]+)_", fn)
        me = re.search(rf"_{m['eps']}([0-9.e+-]+)_", fn)
        ms = re.search(r"_seed(\d+)_", fn)
        if not (mr and ml and me and ms): continue
        if int(ms.group(1)) not in a.seeds: continue
        if abs(float(mr.group(1)) - float(a.rank)) > 1e-9: continue
        combos.add((float(ml.group(1)), float(me.group(1))))
    return combos


# ============ 데이터 로드 ============
_SV_CACHE = {}
def load_sv(path):
    if path in _SV_CACHE: return _SV_CACHE[path]
    if not os.path.exists(path):
        _SV_CACHE[path] = (None, None); return None, None
    d = pickle.load(open(path, "rb"))
    dv = np.array(d["dv_result"]); si = np.array(d["sampled_idx"])
    sv = dv[:, 1, :].sum(axis=1) if dv.ndim == 3 else dv[:, 1]
    _SV_CACHE[path] = (sv, si); return sv, si

def align(a_sv, a_si, b_sv, b_si):
    da = {int(i): v for i, v in zip(a_si.tolist(), a_sv)}
    db = {int(i): v for i, v in zip(b_si.tolist(), b_sv)}
    common = sorted(set(a_si.tolist()) & set(b_si.tolist()))
    return (np.array([da[i] for i in common]), np.array([db[i] for i in common]))

def _rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r

_PRED_CACHE = {}
def parse_pred_inv(path):
    if path in _PRED_CACHE: return _PRED_CACHE[path]
    if not os.path.exists(path):
        _PRED_CACHE[path] = (None, None); return None, None
    lines = open(path).read().splitlines()
    i = next((idx for idx, ln in enumerate(lines)
              if ln.strip().lower().startswith("inv mode")), None)
    if i is None:
        _PRED_CACHE[path] = (None, None); return None, None
    top = rnd = None; cur = None
    for j in range(i + 1, len(lines)):
        low = lines[j].strip().lower()
        if low.startswith(("eigen mode", "nystrom mode")): break
        if low == "top:": cur = "top"; continue
        if low == "random:": cur = "random"; continue
        s = lines[j].strip()
        if s.startswith("["):
            try: arr = np.array(ast.literal_eval(s), dtype=float) / 10000.0
            except Exception: arr = None
            if cur == "top": top = arr
            elif cur == "random": rnd = arr
    _PRED_CACHE[path] = (top, rnd); return top, rnd


# ============ seed 집계 ============
def _mean_std_n(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals: return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)

def fid_seeds(a, method, lam, eps, INV, which, ov_pct=20):
    """seed별 Spearman('sp')/Pearson('pe')/top-overlap('ov') → (mean, std, n)."""
    out = []
    for s in a.seeds:
        asv, asi = load_sv(sv_path(a, method, lam, eps, s))
        isv, isi = INV.get(s, (None, None))
        if asv is None or isv is None: continue
        av, bv = align(asv, asi, isv, isi)
        if len(av) < 10: continue
        if which == "sp":
            out.append(float(np.corrcoef(_rank(av), _rank(bv))[0, 1]))
        elif which == "pe":
            out.append(float(np.corrcoef(av, bv)[0, 1]))
        elif which == "ov":
            k = max(1, int(len(av) * ov_pct / 100.0))
            out.append(len(set(np.argsort(av)[-k:].tolist()) &
                           set(np.argsort(bv)[-k:].tolist())) / k)
    return _mean_std_n(out)

def acc_summary_seeds(a, method, lam, eps, mode, k30=30):
    """seed별 acc 요약 → (mean, std, n).  mode='auc'(전 k 평균) / 'k30'(k≤N% 평균)."""
    return _mean_std_n([_acc_one(a, method, lam, eps, s, mode, k30) for s in a.seeds])

def _fid_one(a, method, lam, eps, seed, INV, which, ov_pct):
    """단일 seed fidelity 값 (없으면 None)."""
    asv, asi = load_sv(sv_path(a, method, lam, eps, seed))
    isv, isi = INV.get(seed, (None, None))
    if asv is None or isv is None: return None
    av, bv = align(asv, asi, isv, isi)
    if len(av) < 10: return None
    if which == "sp": return float(np.corrcoef(_rank(av), _rank(bv))[0, 1])
    if which == "pe": return float(np.corrcoef(av, bv)[0, 1])
    if which == "ov":
        k = max(1, int(len(av) * ov_pct / 100.0))
        return len(set(np.argsort(av)[-k:].tolist()) & set(np.argsort(bv)[-k:].tolist())) / k
    return None

def _acc_one(a, method, lam, eps, seed, mode, k30):
    """단일 seed acc 요약 값 (없으면 None)."""
    top, _ = parse_pred_inv(pred_path(a, method, lam, eps, seed))
    if top is None: return None
    return float(np.mean(top)) if mode == "auc" else float(np.mean(top[:k30]))

def _arm_cells(cross, axis):
    """axis='eps' → ε sweep 셀 [(lam0,e)...] / axis='lam' → λ sweep 셀 [(l,eps0)...]."""
    if axis == "eps":
        return [(cross["lam0"], e) for e in
                sorted({e for (l, e) in cross["cells"] if cross["close"](l, cross["lam0"])})]
    return [(l, cross["eps0"]) for l in
            sorted({l for (l, e) in cross["cells"] if cross["close"](e, cross["eps0"])})]

def regret_seeds(a, method, cross, fk, ak, INV, ov_pct, k30, axis):
    """regret = max acc − acc(argmax fidelity),  해당 arm(axis) 셀에서.  seed별 → (mean,std,n)."""
    cells = _arm_cells(cross, axis)
    regs = []
    for s in a.seeds:
        fs, ac = [], []
        for (l, e) in cells:
            f = _fid_one(a, method, l, e, s, INV, fk, ov_pct)
            c = _acc_one(a, method, l, e, s, ak, k30)
            if f is None or c is None: continue
            fs.append(f); ac.append(c)
        if len(fs) < 3: continue
        fs = np.array(fs); ac = np.array(ac)
        regs.append(float(ac.max() - ac[int(np.argmax(fs))]))
    return _mean_std_n(regs)

def sel_seeds(a, method, lam, eps):
    """seed별 inv-mode top acc(100개) → (mean[100], std[100], n).  random 도 mean."""
    tops, rnds = [], []
    for s in a.seeds:
        top, rnd = parse_pred_inv(pred_path(a, method, lam, eps, s))
        if top is not None: tops.append(top[:100])
        if rnd is not None: rnds.append(rnd[:100])
    if not tops: return None, None, None, 0
    L = min(len(t) for t in tops)
    A = np.vstack([t[:L] for t in tops])
    R = np.vstack([r[:L] for r in rnds]) if rnds else None
    return A.mean(0), A.std(0), (R.mean(0) if R is not None else None), len(tops)

def sel_at_k_seeds(a, method, lam, eps, k):
    """seed별 acc@k% → (mean, std, n)."""
    out = []
    for s in a.seeds:
        top, _ = parse_pred_inv(pred_path(a, method, lam, eps, s))
        if top is not None and len(top) >= k: out.append(float(top[k - 1]))
    return _mean_std_n(out)


# ============ 십자 grid ============
def build_cross(a, method):
    m = METH[method]
    eps0 = a.eps_anchor if a.eps_anchor is not None else m["eps_anchor_default"]
    lam0 = a.lam_anchor if a.lam_anchor is not None else m["lam_anchor_default"]
    combos = scan_sweeps(a, method)
    if not combos: return None
    def close(x, y): return abs(x - y) < abs(y) * 1e-6 + 1e-30
    eps_at_lam0 = sorted({e for (l, e) in combos if close(l, lam0)})
    lam_at_eps0 = sorted({l for (l, e) in combos if close(e, eps0)})
    eps_vals = sorted(set(eps_at_lam0) | {eps0})
    lam_vals = sorted(set(lam_at_eps0) | {lam0})
    cells = {(l, e) for (l, e) in combos if close(l, lam0) or close(e, eps0)}
    return dict(eps_vals=eps_vals, lam_vals=lam_vals, eps0=eps0, lam0=lam0,
                cells=cells, close=close)

def cross_matrices(a, method, cross, val_fn):
    """val_fn(lam,eps)->(mean,std,n). 십자 셀만 채움."""
    ev, lv = cross["eps_vals"], cross["lam_vals"]
    Mm = np.full((len(lv), len(ev)), np.nan); Ms = np.full_like(Mm, np.nan)
    for i, l in enumerate(lv):
        for j, e in enumerate(ev):
            if any(cross["close"](l, cl) and cross["close"](e, ce) for (cl, ce) in cross["cells"]):
                mm, ss, _ = val_fn(l, e); Mm[i, j] = mm; Ms[i, j] = ss
    return Mm, Ms, ev, lv


# ============ 플롯 헬퍼 ============
def draw_heat(ax, Mm, Ms, xlabels, ylabels, title, cmap, vmin, vmax,
              anchor_ij=None, xlab="jitter ε", ylab="ridge λ", fmt="{:.2f}", sfmt="{:.2f}"):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax); cm = plt.get_cmap(cmap)
    nr, nc = Mm.shape
    for i in range(nr):
        for j in range(nc):
            v = Mm[i, j]
            fc = "#dddddd" if np.isnan(v) else cm(norm(v))
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=fc, edgecolor="white", lw=.4))
            if not np.isnan(v):
                tc = "white" if norm(v) < .45 else "black"
                ax.text(j, i + .14, fmt.format(v), ha="center", va="center", fontsize=6.5, color=tc)
                if Ms is not None and not np.isnan(Ms[i, j]):
                    ax.text(j, i - .24, "±" + sfmt.format(Ms[i, j]), ha="center", va="center", fontsize=4.5, color=tc)
    ax.set_xlim(-.5, nc - .5); ax.set_ylim(-.5, nr - .5)
    ax.set_xticks(range(nc)); ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_yticks(range(nr)); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6); ax.set_aspect("auto")
    if anchor_ij is not None and anchor_ij[0] is not None and anchor_ij[1] is not None:
        ai, aj = anchor_ij
        ax.add_patch(plt.Rectangle((aj - .5, ai - .5), 1, 1, fill=False, edgecolor="red", lw=2, zorder=5))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=.046, pad=.04)

def draw_strip(ax, mm, ss, labels, title, cmap, vmin, vmax, fmt="{:.2f}"):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax); cm = plt.get_cmap(cmap)
    n = len(mm)
    for j in range(n):
        v = mm[j]
        fc = "#dddddd" if (v is None or np.isnan(v)) else cm(norm(v))
        ax.add_patch(plt.Rectangle((j - .5, -.5), 1, 1, facecolor=fc, edgecolor="white", lw=.4))
        if v is not None and not np.isnan(v):
            tc = "white" if norm(v) < .45 else "black"
            ax.text(j, .1, fmt.format(v), ha="center", va="center", fontsize=6.5, color=tc)
            if ss is not None and not np.isnan(ss[j]):
                ax.text(j, -.26, f"±{ss[j]:.2f}", ha="center", va="center", fontsize=4.5, color=tc)
    ax.set_xlim(-.5, n - .5); ax.set_ylim(-.5, .5)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7); ax.set_yticks([])
    ax.set_title(title, fontsize=9, loc="left", pad=3)


# ============ 페이지: fidelity ============
def page_fidelity(pdf, a, method, INV):
    m = METH[method]; cross = build_cross(a, method)
    if cross is None: return
    SPm, SPs, ev, lv = cross_matrices(a, method, cross, lambda l, e: fid_seeds(a, method, l, e, INV, "sp"))
    PEm, PEs, _, _   = cross_matrices(a, method, cross, lambda l, e: fid_seeds(a, method, l, e, INV, "pe"))
    AUm, AUs, _, _   = cross_matrices(a, method, cross, lambda l, e: acc_summary_seeds(a, method, l, e, "auc"))
    xl = [_fmt_eps(e) for e in ev]; yl = [_fmt_eps(l) for l in lv]
    ai = lv.index(cross["lam0"]) if cross["lam0"] in lv else None
    aj = ev.index(cross["eps0"]) if cross["eps0"] in ev else None
    anchor = (ai, aj)
    # AUC 는 값 폭이 좁아(예: 0.72~0.74) 0~1 스케일이면 색차이가 안 보임 → 데이터 범위로 자동 스케일
    _fin = AUm[np.isfinite(AUm)]
    amin, amax = (float(_fin.min()), float(_fin.max())) if _fin.size else (0.0, 1.0)
    if amax - amin < 1e-6: amax = amin + 1e-6

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(5, 6, height_ratios=[3, 1, 1, 1, 1], hspace=.75, wspace=.6)
    fig.suptitle(f"[{a.dataset}] {m['kor']} — Fidelity 십자 + Selection AUC (근사 SV vs inv SV)  "
                 f"| rank/d={a.rank:g}%, tmc={a.tmc}, seeds={a.seeds}",
                 fontsize=13, fontweight="bold")
    ax_sp = fig.add_subplot(gs[0, 0:2]); ax_pe = fig.add_subplot(gs[0, 2:4]); ax_au = fig.add_subplot(gs[0, 4:6])
    draw_heat(ax_sp, SPm, SPs, xl, yl, "Spearman ρ (mean±std)", "viridis", 0.0, 1.0, anchor,
              xlab=f"{m['eps']} (ε)", ylab=f"{m['lam']} (λ)")
    draw_heat(ax_pe, PEm, PEs, xl, yl, "Pearson r (mean±std)", "cividis", 0.0, 1.0, anchor,
              xlab=f"{m['eps']} (ε)", ylab=f"{m['lam']} (λ)")
    draw_heat(ax_au, AUm, AUs, xl, yl, "Selection AUC k1–100% (mean±std)", "viridis", amin, amax, anchor,
              xlab=f"{m['eps']} (ε)", ylab=f"{m['lam']} (λ)", fmt="{:.3f}", sfmt="{:.3f}")
    hi = lv.index(cross["lam0"]) if cross["lam0"] in lv else None
    vj = ev.index(cross["eps0"]) if cross["eps0"] in ev else None
    arms = []
    if hi is not None:
        arms.append(("Spearman — ε sweep (λ 고정)", SPm[hi, :], SPs[hi, :], xl, "viridis"))
        arms.append(("Pearson — ε sweep (λ 고정)",  PEm[hi, :], PEs[hi, :], xl, "cividis"))
    if vj is not None:
        arms.append(("Spearman — λ sweep (ε 고정)", SPm[:, vj], SPs[:, vj], yl, "viridis"))
        arms.append(("Pearson — λ sweep (ε 고정)",  PEm[:, vj], PEs[:, vj], yl, "cividis"))
    for idx, (ttl, mm, ss, labs, cmap) in enumerate(arms[:4]):
        ax = fig.add_subplot(gs[1 + idx // 2, (0 if idx % 2 == 0 else 3):(3 if idx % 2 == 0 else 6)])
        draw_strip(ax, list(mm), list(ss), labs, ttl, cmap, 0, 1)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 페이지: selection 곡선 (eps/λ sweep, seed 평균 선) ============
def page_selection_curves(pdf, a, axis="eps"):
    ks = list(range(1, 101))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"wspace": .6})
    axsym = "ε" if axis == "eps" else "λ"
    fig.suptitle(f"[{a.dataset}] Selection 곡선 — top-k%(1~100), 모든 {axsym} sweep 오버레이 "
                 f"(INV-mode acc, seed 평균)", fontsize=13, fontweight="bold")
    for ax, method in zip(axes, ["nystrom", "eigen"]):
        m = METH[method]; cross = build_cross(a, method)
        if cross is None:
            ax.set_title(m["kor"]); ax.text(.5, .5, "데이터 없음", ha="center", va="center",
                                            transform=ax.transAxes); continue
        if axis == "eps":
            vals = sorted({e for (l, e) in cross["cells"] if cross["close"](l, cross["lam0"])})
            pairs = [(v, cross["lam0"], v) for v in vals]        # (label, lam, eps)
            tagname = f"{m['eps']} (ε)"; fixed = f"λ={_fmt_eps(cross['lam0'])}"
        else:
            vals = sorted({l for (l, e) in cross["cells"] if cross["close"](e, cross["eps0"])})
            pairs = [(v, v, cross["eps0"]) for v in vals]
            tagname = f"{m['lam']} (λ)"; fixed = f"ε={_fmt_eps(cross['eps0'])}"
        ax.set_title(f"{m['kor']} — {axsym} sweep ({fixed} 고정)", fontsize=11)
        ax.set_xlabel("k% (top-k% 사용)", fontsize=10); ax.set_ylabel("val acc (seed 평균)", fontsize=10)
        if not vals: continue
        cmap = plt.get_cmap("tab20"); rnd_drawn = False
        for idx, (lab, lam, eps) in enumerate(pairs):
            mtop, stop, mrnd, n = sel_seeds(a, method, lam, eps)
            if mtop is not None:
                ax.plot(ks[:len(mtop)], mtop, "-", color=cmap(idx % 20), lw=1.6,
                        label=f"{axsym}={_fmt_eps(lab)}")
            if (not rnd_drawn) and mrnd is not None:
                ax.plot(ks[:len(mrnd)], mrnd, "--", color="black", lw=1.4, alpha=.5,
                        label="random", zorder=1); rnd_drawn = True
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, .5),
                  title=tagname, title_fontsize=8.5, ncol=1, handlelength=1.6)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 페이지: selection sweep (acc vs eps, mean±std band) ============
def page_selection_sweep(pdf, a):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"[{a.dataset}] Selection sweep — acc vs ε (top-k%, k={a.sweep_ks}), mean±std band",
                 fontsize=13, fontweight="bold")
    for ax, method in zip(axes, ["nystrom", "eigen"]):
        m = METH[method]; cross = build_cross(a, method)
        ax.set_title(f"{m['kor']} — {m['eps']} sweep (λ={_fmt_eps(cross['lam0']) if cross else '?'})", fontsize=11)
        ax.set_xlabel(f"{m['eps']} (ε)", fontsize=10); ax.set_ylabel("val acc", fontsize=10)
        if cross is None:
            ax.text(.5, .5, "데이터 없음", ha="center", va="center", transform=ax.transAxes); continue
        eps_row = sorted({e for (l, e) in cross["cells"] if cross["close"](l, cross["lam0"])})
        for k in a.sweep_ks:
            xs, ms, ss = [], [], []
            for e in eps_row:
                mm, sd, n = sel_at_k_seeds(a, method, cross["lam0"], e, k)
                if not np.isnan(mm): xs.append(e); ms.append(mm); ss.append(sd)
            if xs:
                xs = np.array(xs); ms = np.array(ms); ss = np.array(ss)
                ln, = ax.plot(xs, ms, "o-", lw=1.8, markersize=5, label=f"k={k}%")
                ax.fill_between(xs, ms - ss, ms + ss, color=ln.get_color(), alpha=.18)
        if eps_row: ax.set_xscale("log")
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 페이지: fidelity–acc coupling (3 metric × 2 acc = 6칸), axis별 ============
def page_coupling(pdf, a, INV, axis):
    """axis='eps' → ε(jitter) sweep 만으로 / axis='lam' → λ(ridge) sweep 만으로 상관."""
    axsym = "ε (jitter)" if axis == "eps" else "λ (ridge)"
    fixed = "λ=1e-2 고정" if axis == "eps" else "ε=anchor 고정"
    fid_metrics = [("sp", "Spearman ρ"), ("pe", "Pearson r"),
                   ("ov", f"top-{a.overlap_pct}% overlap")]
    acc_modes   = [("auc", "acc AUC (전 k 평균)"), ("k30", f"acc (k≤{a.acc_k30}% 평균)")]
    mcol = {"nystrom": "#1f77b4", "eigen": "#d62728"}
    fig, axes = plt.subplots(3, 2, figsize=(14, 16.8))
    fig.suptitle(f"[{a.dataset}] Fidelity–Acc coupling — {axsym} sweep only  ({fixed}, mean±std, seeds={a.seeds})",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.958,
             r"$\mathrm{regret}\;=\;\max_{\theta}\ \mathrm{acc}(\theta)\;-\;\mathrm{acc}\!\left(\theta^{*}_{\mathrm{fid}}\right)"
             r"\qquad \mathrm{where}\ \ \theta^{*}_{\mathrm{fid}}=\arg\max_{\theta}\ \mathrm{fidelity}(\theta)$",
             ha="center", fontsize=13)
    fig.text(0.5, 0.940,
             f"= fidelity 최적 {axsym}로 골랐을 때 놓치는 acc (seed별 argmax→평균±std).  regret≈0 → fidelity=acc 결합 / regret>0 → 어긋남",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.924, f"행=fidelity 지표 · 열=acc 요약 · 점={axsym} · corr·regret은 {axsym} sweep 점만 기준",
             ha="center", fontsize=9, color="gray")
    fig.subplots_adjust(top=0.905, hspace=0.32, wspace=0.24)
    for ri, (fk, fl) in enumerate(fid_metrics):
        for ci, (ak, al) in enumerate(acc_modes):
            ax = axes[ri, ci]
            ax.set_xlabel(fl, fontsize=9); ax.set_ylabel(al, fontsize=9)
            ax.grid(True, alpha=.3)
            corr_txt = []
            for method in ["nystrom", "eigen"]:
                cross = build_cross(a, method)
                if cross is None: continue
                col = mcol[method]; dy = 4 if method == "eigen" else -8
                cells = _arm_cells(cross, axis)
                xs, ys, xe, ye, lbs = [], [], [], [], []
                for (l, e) in cells:
                    fm, fs, _ = fid_seeds(a, method, l, e, INV, fk, ov_pct=a.overlap_pct)
                    am, as_, _ = acc_summary_seeds(a, method, l, e, ak, k30=a.acc_k30)
                    if np.isnan(fm) or np.isnan(am): continue
                    xs.append(fm); ys.append(am); xe.append(fs); ye.append(as_)
                    lbs.append(f"ε={_fmt_eps(e)}" if axis == "eps" else f"λ={_fmt_eps(l)}")
                if not xs: continue
                ax.plot(xs, ys, "-", color=col, lw=.6, alpha=.3, zorder=1)  # sweep 궤적
                for xi, yi, xei, yei, lb in zip(xs, ys, xe, ye, lbs):
                    ax.errorbar(xi, yi, xerr=xei, yerr=yei, fmt="o", ms=5, color=col,
                                ecolor=col, elinewidth=.5, capsize=1.3, alpha=.85, zorder=3)
                    ax.annotate(lb, (xi, yi), fontsize=4.2, color=col, alpha=.85,
                                xytext=(2, dy), textcoords="offset points", zorder=4)
                ax.plot([], [], "o", color=col, label=METH[method]["kor"])
                if len(xs) >= 3:
                    r = float(np.corrcoef(xs, ys)[0, 1])
                    rgm, rgs, _ = regret_seeds(a, method, cross, fk, ak, INV, a.overlap_pct, a.acc_k30, axis)
                    corr_txt.append((METH[method]["kor"], r, rgm, rgs, col))
            for t, (nm, r, rgm, rgs, col) in enumerate(corr_txt):
                rg = f"{rgm:.3f}±{rgs:.3f}" if not np.isnan(rgm) else "—"
                ax.text(.03, .97 - t * .085, f"{nm}: r={r:+.2f}, regret={rg}", transform=ax.transAxes,
                        fontsize=8.5, va="top", fontweight="bold", color=col)
            ax.legend(fontsize=8, loc="lower right")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ 페이지: selection 십자 ============
def page_selection_cross(pdf, a, method):
    m = METH[method]; cross = build_cross(a, method)
    if cross is None: return
    ks = a.sel_ks[:2] if len(a.sel_ks) >= 2 else (a.sel_ks + a.sel_ks)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"[{a.dataset}] {m['kor']} — Selection 십자 (INV-mode top-k% acc, mean±std)",
                 fontsize=13, fontweight="bold")
    for ax, k in zip(axes, ks):
        Mm, Ms, ev, lv = cross_matrices(a, method, cross,
                                        lambda l, e, _k=k: sel_at_k_seeds(a, method, l, e, _k))
        xl = [_fmt_eps(e) for e in ev]; yl = [_fmt_eps(l) for l in lv]
        ai = lv.index(cross["lam0"]) if cross["lam0"] in lv else None
        aj = ev.index(cross["eps0"]) if cross["eps0"] in ev else None
        fin = Mm[np.isfinite(Mm)]
        vmn = float(np.floor(fin.min() * 20) / 20) if fin.size else 0.5
        vmx = float(np.ceil(fin.max() * 20) / 20) if fin.size else 0.9
        draw_heat(ax, Mm, Ms, xl, yl, f"acc @ k={k}%", "magma", vmn, vmx, (ai, aj),
                  xlab=f"{m['eps']} (ε)", ylab=f"{m['lam']} (λ)")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ============ main ============
def main():
    a = parse_args()
    if a.val is None:
        for method in a.methods:
            v = autodetect_val(a, method)
            if v is not None: a.val = v; break
    if a.val is None:
        print("[error] val 자동 감지 실패 — --val 지정 필요"); return
    if a.out is None:
        a.out = f"./jitter_exp/report_{a.dataset}_n{a.num_train}_tmc{a.tmc}.pdf"

    print(f"[cfg] dataset={a.dataset} seeds={a.seeds} n={a.num_train} val={a.val} tmc={a.tmc} rank={a.rank}")
    INV = {s: load_sv(inv_sv_path(a, s)) for s in a.seeds}
    got = [s for s in a.seeds if INV[s][0] is not None]
    print(f"[inv] 존재 seed: {got}")

    with PdfPages(a.out) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.72, f"Report — {a.dataset}", ha="center", fontsize=22, fontweight="bold")
        info = (f"model={a.model}, n_train={a.num_train}, val={a.val}, tmc={a.tmc}\n"
                f"seeds={a.seeds} (inv 존재: {got})\n"
                f"rank/d = {a.rank:g}%,  invlam={a.invlam}\n"
                f"methods = {', '.join(METH[mm]['kor'] for mm in a.methods)}\n"
                f"지표: 'inv mode / top:' (근사 SV 랭킹을 INV 커널로 평가)\n"
                f"모든 지표 = seed별 계산 후 평균±std. 미완성 셀/곡선은 빈칸/생략")
        fig.text(0.5, 0.5, info, ha="center", va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for method in a.methods:
            page_fidelity(pdf, a, method, INV)
        page_selection_curves(pdf, a, "eps")
        page_selection_curves(pdf, a, "lam")
        page_selection_sweep(pdf, a)
        page_coupling(pdf, a, INV, "eps")
        page_coupling(pdf, a, INV, "lam")
        for method in a.methods:
            page_selection_cross(pdf, a, method)

    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()

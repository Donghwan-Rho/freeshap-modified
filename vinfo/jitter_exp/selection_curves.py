# -*- coding: utf-8 -*-
"""selection(top-k 추가) 곡선을 결과 파일에서 직접 읽어오는 로더/수집기.

speedup_er.ipynb 의 셀 1~31 에 하드코딩돼 있던 리터럴을 대체한다.
노트북에서:

    import sys; sys.path.insert(0, "..")          # jitter_exp 가 보이는 위치로
    from selection_curves import load_into
    load_into(globals())

호출하면 아래 이름들이 globals() 에 주입된다 (기존 리터럴과 동일한 이름/형식):

    {prefix}_inv_seed{S}          = [100개 int]   # val acc x 1e4, top 1~100% 추가
    {prefix}_eigen_r{R}_seed{S}   = [100개 int]
    {prefix}_inv_seed{S}_base     = int           # 0% 선택 시 base accuracy

prefix 규약 (기존 노트북과 동일):
    bert   : "{ds}_{n}"           예: sst2_5000
    llama  : "llama_{ds}_{n}"     예: llama_sst2_5000
    resnet : "cifar10_5000"

파일이 없는 조합은 주입하지 않는다 -> 노트북에 리터럴이 남아 있으면 그 값이 유지되고,
디스크에 있으면 항상 최신 값으로 덮인다 (예: cifar10 은 3090/λ=1e-6 재측정판).

단독 실행하면 어떤 조합이 디스크에 있는지 표로 보여준다:
    python jitter_exp/selection_curves.py
"""
import os
import re

# 이 파일은 jitter_exp/ 에 있으므로 그 부모가 vinfo 루트.
# 노트북이 reports_*/ 안에서 실행돼도 CWD 와 무관하게 결과를 찾도록 절대경로로 잡는다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# auc_table.py 와 같은 규약: SELECTION_DIR 로 held-out / in-sample 을 전환한다.
SELECTION_DIR = os.environ.get("SELECTION_DIR", "data_selection")
BASE = os.path.join(ROOT, "freeshap_res", SELECTION_DIR)
SEEDS = (2024, 2025, 2026)
RANKS = (1, 5, 10, 15, 20, 25, 30)
VAL = {"sst2": 872, "mrpc": 408, "rte": 277}
SCALE_LEN = 100          # 곡선 길이 (top 1~100%)

# (model, dataset, num_train) -> 노트북 변수 접두
COMBOS = [("bert", ds, n) for ds, ns in [
              ("sst2", (1000, 2000, 5000, 10000)), ("mnli", (1000, 2000, 5000, 10000)),
              ("mr", (1000, 2000, 5000)), ("mrpc", (1000, 2000, 3668)),
              ("qqp", (1000, 2000, 5000)), ("rte", (1000, 2000, 2490)),
              ("ag_news", (1000, 2000, 5000))] for n in ns] + \
         [("llama", ds, n) for ds, n in [("sst2", 5000), ("mnli", 5000), ("mr", 5000),
                                          ("mrpc", 3668), ("qqp", 5000), ("rte", 2490),
                                          ("ag_news", 5000), ("sst2", 1000), ("mnli", 1000), ("qqp", 1000), ("mr", 1000),
                                          ("ag_news", 1000), ("rte", 1000), ("mrpc", 1000),
                                          ("sst2", 2000), ("mnli", 2000), ("qqp", 2000),
                                          ("mr", 2000), ("ag_news", 2000), ("rte", 2000),
                                          ("mrpc", 2000)]] + \
         [("resnet", "cifar10", 5000), ("resnet", "cifar10", 2000)]


def prefix_of(model, ds, n):
    if model == "resnet":
        return f"{ds}_{n}"
    return (f"llama_{ds}_{n}" if model == "llama" else f"{ds}_{n}")


def _paths(model, ds, n, s, rank):
    """신형 -> 구형 파일명 변형 순서 (auc_table 과 동일 규약)."""
    v = VAL.get(ds, 1000)
    tail = "_signFalse_earlystopTrue_tmc500_predictions.txt"
    if rank is None:
        return [f"{BASE}/{ds}/inv/predictions/{model}_seed{s}_num{n}_val{v}_lam1e-06{tail}"]
    h = f"{BASE}/{ds}/eigen/predictions/{model}_seed{s}_num{n}_val{v}"
    return [f"{h}_eig{rank}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32{tail}",
            f"{h}_eig{rank}_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32{tail}",
            f"{h}_eig{rank}.0_lam1e-02_cholesky_float32{tail}",
            f"{h}_eig{rank}_lam1e-02_cholesky_float32{tail}"]


def _curve(path):
    """'inv mode lambda' 블록의 top: 리스트 (val acc x 1e4). 없으면 None.

    txt 는 append 모드로 쓰여서 재실행분이 누적된다. SV pkl 이 중간에 새로 만들어진
    경우 블록마다 값이 다른데, **마지막 블록이 현재 pkl 과 일치**한다 (indices 파일로 검증).
    그래서 항상 마지막 블록을 쓴다.
    """
    try:
        txt = open(path).read()
    except OSError:
        return None
    ms = re.findall(r"inv mode lambda[^\n]*\ntop:\s*\n?\[([^\]]+)\]", txt)
    return None if not ms else [int(x) for x in re.findall(r"\d+", ms[-1])]


def _base(path):
    """base_accuracy txt 의 inv_base_accuracy (val acc x 1e4). 없으면 None."""
    p = path.replace("/predictions/", "/base_accuracy/").replace("_predictions.txt", "_base.txt")
    try:
        txt = open(p).read()
    except OSError:
        return None
    m = re.search(r"inv_base_accuracy:\s*\n?\s*(\d+)", txt)
    return None if not m else int(m.group(1))


def _first(paths, fn):
    for p in paths:
        if os.path.exists(p):
            v = fn(p)
            if v:
                return v
    return None


def curves_for(model, ds, n):
    """한 조합의 {변수명: 값} dict. 없는 항목은 넣지 않는다."""
    pref, out = prefix_of(model, ds, n), {}
    for s in SEEDS:
        ps = _paths(model, ds, n, s, None)
        c = _first(ps, _curve)
        if c:
            out[f"{pref}_inv_seed{s}"] = c
        b = _first(ps, _base)
        if b is not None:
            out[f"{pref}_inv_seed{s}_base"] = b
        for r in RANKS:
            c = _first(_paths(model, ds, n, s, r), _curve)
            if c:
                out[f"{pref}_eigen_r{r}_seed{s}"] = c
    return out


def load_into(g, combos=None, verbose=True):
    """디스크에서 읽어 g(=globals()) 에 주입. 반환: (덮어쓴 수, 새로 넣은 수, 조합별 요약)."""
    over = new = 0
    summary = []
    for model, ds, n in (combos if combos is not None else COMBOS):
        d = curves_for(model, ds, n)
        if not d:
            continue
        o = sum(1 for k in d if k in g)
        over += o
        new += len(d) - o
        summary.append((prefix_of(model, ds, n), model, len(d)))
        g.update(d)
    if verbose:
        print(f"[selection_curves] 디스크에서 {over + new}개 변수 주입 "
              f"(기존 리터럴 덮어씀 {over}, 신규 {new}) / 조합 {len(summary)}개")
        for pref, model, k in sorted(summary):
            print(f"    {pref:<22s} {model:<7s} {k:>3d}개")
    return over, new, summary


if __name__ == "__main__":
    print(f"{'prefix':<22s} {'model':<7s} {'변수수':>6s}  (최대 27 = inv3 + base3 + eigen21)")
    for model, ds, n in COMBOS:
        d = curves_for(model, ds, n)
        if d:
            print(f"{prefix_of(model, ds, n):<22s} {model:<7s} {len(d):>6d}")

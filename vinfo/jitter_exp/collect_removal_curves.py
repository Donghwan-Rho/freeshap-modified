# -*- coding: utf-8 -*-
"""removal 곡선(top_removal, inv-mode 예측) 리터럴 생성기.

data_removing/{ds}/{inv|eigen}/predictions/{stem}_predictions.txt 의
  "inv mode lambda=..." 블록 -> "top_removal:" 다음 리스트 100개 (0~99% 제거)
를 뽑아 파이썬 리터럴로 출력. eigen 파일도 inv-mode 블록만 사용.

사용:
  python jitter_exp/collect_removal_curves.py bert5000    # bert full-size (rte 2490, mrpc 3668)
  python jitter_exp/collect_removal_curves.py llama5000
  python jitter_exp/collect_removal_curves.py resnet5000  # cifar10 (inv λ 는 --inv_lam 으로)
"""
import os, re, sys

BASE = "./freeshap_res/data_removing"
SEEDS = [2024, 2025, 2026]
RANKS = [1, 5, 10, 15, 20, 25, 30]
NUM = {"rte": 2490, "mrpc": 3668}
VAL = {"sst2": 872, "mrpc": 408, "rte": 277}
DS_TEXT = ["sst2", "mr", "qqp", "mnli", "ag_news", "rte", "mrpc"]


def top_removal(path):
    """inv-mode 블록의 top_removal 리스트 (없으면 None)."""
    try:
        txt = open(path).read()
    except OSError:
        return None
    # 신형 "top_removal:" / 구형 "top removal (removing high Shapley):" 둘 다 인식
    # append 모드라 재실행분이 누적될 수 있어 마지막 블록을 쓴다 (selection 과 동일 규약)
    ms = re.findall(r"inv mode lambda[^\n]*\ntop[ _]removal[^\n]*:\s*\n?\[([^\]]+)\]", txt)
    if not ms:
        return None
    return [int(x) for x in re.findall(r"\d+", ms[-1])]


def paths(model, ds, s, kind, r=None, inv_lam="1e-06"):
    n = NUM.get(ds, 5000) if model != "resnet" else 5000
    v = VAL.get(ds, 1000)
    if kind == "inv":
        st = f"{model}_seed{s}_num{n}_val{v}_lam{inv_lam}_signFalse_earlystopTrue_tmc500"
        return f"{BASE}/{ds}/inv/predictions/{st}_predictions.txt"
    st = (f"{model}_seed{s}_num{n}_val{v}_eig{r}.0_eiglam1e-02_eigeps1e-8"
          f"_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500")
    return f"{BASE}/{ds}/eigen/predictions/{st}_predictions.txt"


def fmt(v):
    return "None" if v is None else "[" + ", ".join(map(str, v)) + "]"


def emit(model, datasets, tag, inv_lam="1e-06"):
    print(f"# ==== {tag} — removal 곡선 (top_removal, inv-mode 예측; 0~99% 제거) ====")
    print(f"# seeds 순서: {SEEDS} / eigen rank: {RANKS} (%)")
    print(f"# 값은 val acc x 1e4. 파일 없으면 None.")
    print(f"# 재생성: python jitter_exp/collect_removal_curves.py {sys.argv[1]}")
    print()
    for ds in datasets:
        n = NUM.get(ds, 5000) if model != "resnet" else 5000
        for s in SEEDS:
            v = top_removal(paths(model, ds, s, "inv", inv_lam=inv_lam))
            print(f"{ds}_{n}_inv_removal_seed{s} = {fmt(v)}")
        print()
        for s in SEEDS:
            for r in RANKS:
                v = top_removal(paths(model, ds, s, "eigen", r=r, inv_lam=inv_lam))
                pad = " " if r in (1, 5) else ""
                print(f"{ds}_{n}_eigen_removal_r{r}_seed{s}{pad} = {fmt(v)}")
            print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bert5000"
    if mode == "bert5000":
        emit("bert", DS_TEXT, "bert full-size (n5000; rte 2490, mrpc 3668)")
    elif mode == "llama5000":
        emit("llama", DS_TEXT, "llama full-size (n5000; rte 2490, mrpc 3668)")
    elif mode == "resnet5000":
        lam = sys.argv[2] if len(sys.argv) > 2 else "1e-06"
        emit("resnet", ["cifar10"], f"resnet cifar10 n5000 (inv λ={lam})", inv_lam=lam)
    else:
        raise SystemExit(f"unknown mode: {mode}")

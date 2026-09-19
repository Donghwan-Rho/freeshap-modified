# -*- coding: utf-8 -*-
"""wrong-label detection 곡선 리터럴 생성기.

wrong_label_detection/{ds}/{inv|eigen}/predictions/{stem}_detection.txt 의
  "detection rates (x10000, lowest Shapley first):" 다음 리스트 (inspect 1~100%)
를 뽑아 파이썬 리터럴로 출력.

wld 파일명은 형식 변형이 많아 glob 으로 매칭한다:
  inv   : bert_seed{S}_num{N}_val{V}_lam1e-06_..._poison{P}_ps{S}_detection.txt
  eigen : bert_seed{S}_num{N}_val{V}_eig{r}_lam1e-02[_eigeps1e-8]_..._poison{P}_ps{S}_detection.txt
          (구형 = eigeps 없음, 신형 = eigeps 있음 — 신형 우선)

사용: python jitter_exp/collect_wld_curves.py bert5000 | llama5000
"""
import os, re, sys, glob

# jitter_exp/ 의 부모가 vinfo 루트. 노트북이 reports_*/ 에서 import 해도 찾도록 절대경로.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# wld 결과 폴더. 예전(train 전체) 결과는 freeshap_res/, RTE/MRPC 를 held-out 용으로 쪼갠
# 새 train(rte 1500 / mrpc 3000)으로 다시 돌린 결과는 freeshap_res_wld/ (n09_wld_split.sh).
#   WLD_DIR=freeshap_res_wld python ...   /  노트북: os.environ["WLD_DIR"]=... 후 reload
WLD_DIR = os.environ.get("WLD_DIR", "freeshap_res")
BASE = os.path.join(ROOT, WLD_DIR, "wrong_label_detection")
SEEDS = [2024, 2025, 2026]
RANKS = [1, 5, 10, 15, 20, 25, 30]
# RTE/MRPC 의 train 크기: 예전 폴더는 전체(2490/3668), 새 폴더는 FIXED_SPLIT(1500/3000).
#   WLD_NUM=split|full 로 강제할 수 있다 (기본은 폴더 이름으로 판단).
_WLD_NUM = os.environ.get("WLD_NUM", "split" if WLD_DIR != "freeshap_res" else "full")
NUM = {"rte": 1500, "mrpc": 3000} if _WLD_NUM == "split" else {"rte": 2490, "mrpc": 3668}
VAL = {"sst2": 872, "mrpc": 408, "rte": 277}
DS = ["sst2", "mr", "qqp", "mnli", "ag_news", "rte", "mrpc"]
POISON = 10


def det_rates(path):
    """txt 우선, 없으면 pkl 의 detection_results 에서 곡선 복원."""
    if path.endswith(".pkl"):
        try:
            import pickle
            with open(path, "rb") as fh:
                d = pickle.load(fh)
        except Exception:
            return None
        rs = d.get("detection_results")
        if not rs:
            return None
        return [int(round(r["detection_rate"] * 10000)) for r in rs]
    try:
        txt = open(path).read()
    except OSError:
        return None
    # txt 는 append 모드로 쓰여서 재실행 시 블록이 누적된다 -> 마지막(최신) 블록 사용.
    ms = re.findall(r"detection rates \(x10000[^\)]*\):\s*\n?\[([^\]]+)\]", txt)
    if not ms:
        return None
    return [int(x) for x in re.findall(r"\d+", ms[-1])]


def pick(pats):
    """glob 후보들 중 eigeps 포함(신형) 우선, 같은 조건이면 pkl 을 txt 보다 우선.

    txt 는 append 로 누적돼 어느 블록이 최신인지 모호하고, pkl 은 매 실행마다
    통째로 덮어써져 항상 최신 1회 결과라 기준으로 삼기에 안전하다.
    """
    hits = []
    for p in pats:
        hits += glob.glob(p)
    if not hits:
        return None
    hits.sort(key=lambda f: ("eigeps" not in f, not f.endswith(".pkl"), f))
    return hits[0]


def inv_file(model, ds, s):
    n, v = NUM.get(ds, 5000), VAL.get(ds, 1000)
    return pick([f"{BASE}/{ds}/inv/predictions/{model}_seed{s}_num{n}_val{v}_lam1e-06_*"
                 f"_poison{POISON}_ps{s}_detection.txt",
                 f"{BASE}/{ds}/inv/predictions/{model}_seed{s}_num{n}_val{v}_lam1e-06_*"
                 f"_poison{POISON}_ps{s}_detection.pkl"])


def eigen_file(model, ds, s, r):
    n, v = NUM.get(ds, 5000), VAL.get(ds, 1000)
    pats = []
    for ext in ("txt", "pkl"):
        pats += [f"{BASE}/{ds}/eigen/predictions/{model}_seed{s}_num{n}_val{v}_eig{r}_lam1e-02*"
                 f"_poison{POISON}_ps{s}_detection.{ext}",
                 f"{BASE}/{ds}/eigen/predictions/{model}_seed{s}_num{n}_val{v}_eig{r}.0_*"
                 f"_poison{POISON}_ps{s}_detection.{ext}"]
    return pick(pats)


def fmt(v):
    return "None" if v is None else "[" + ", ".join(map(str, v)) + "]"


def emit(model, tag, datasets=None):
    DS = datasets if datasets is not None else globals()["DS"]
    print(f"# ==== {tag} — wrong-label detection rates (x1e4, lowest-Shapley-first; inspect 1~100%) ====")
    print(f"# seeds 순서: {SEEDS} / eigen rank: {RANKS} (%)")
    print(f"# poison {POISON}% (poison_seed = seed). 파일 없으면 None.")
    print(f"# 재생성: python jitter_exp/collect_wld_curves.py {sys.argv[1] if len(sys.argv)>1 else 'bert5000'}")
    print()
    miss = 0
    for ds in DS:
        n = NUM.get(ds, 5000)
        for s in SEEDS:
            f = inv_file(model, ds, s)
            v = det_rates(f) if f else None
            miss += v is None
            print(f"{ds}_{n}_inv_wld_seed{s} = {fmt(v)}")
        print()
        for s in SEEDS:
            for r in RANKS:
                f = eigen_file(model, ds, s, r)
                v = det_rates(f) if f else None
                miss += v is None
                pad = " " if r in (1, 5) else ""
                print(f"{ds}_{n}_eigen_wld_r{r}_seed{s}{pad} = {fmt(v)}")
            print()
    print(f"# 결측 셀: {miss} / {len(DS) * (3 + 21)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bert5000"
    if mode == "bert5000":
        emit("bert", "bert full-size (n5000; rte 2490, mrpc 3668)")
    elif mode == "llama5000":
        emit("llama", "llama full-size (n5000; rte 2490, mrpc 3668)")
    elif mode == "resnet5000":
        emit("resnet", "resnet cifar10 n5000 (inv λ=1e-06)", datasets=["cifar10"])
    else:
        raise SystemExit(f"unknown mode: {mode}")

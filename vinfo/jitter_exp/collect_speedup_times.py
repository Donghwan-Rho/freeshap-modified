# -*- coding: utf-8 -*-
"""speedup.ipynb 데이터 수집기 — bert full-size(n5000; rte 2490, mrpc 3668) inv/eigen
data_selection predictions txt 에서 A6000 의 shapley_computation 초를 수집해
파이썬 리터럴 코드를 출력. (A6000 아님/파일 없음 -> None)
eigen 은 λ=1e-2, eigeps=1e-8 고정 (구형 파일명 eig{r}_lam1e-02, eigeps 없음도 인정).
재생성: python jitter_exp/collect_speedup_times.py
"""
import os, re, glob

DS   = ["sst2", "mr", "qqp", "mnli", "ag_news", "rte", "mrpc"]
NUM  = {"rte": 2490, "mrpc": 3668}          # 그 외 5000 (full-size: 시간·in-sample)
# held-out 프로토콜(data_selection)의 selection 결과는 RTE/MRPC 를 쪼개 써서 n 이 다르다
NUM_SEL = {"rte": 1500, "mrpc": 3000}       # heldout_common.FIXED_SPLIT
VAL  = {"sst2": 872, "mrpc": 408, "rte": 277}  # 그 외 1000
SEEDS = [2024, 2025, 2026]
RANKS = [1, 5, 10, 15, 20, 25, 30]
# jitter_exp/ 의 부모가 vinfo 루트. 노트북이 reports_*/ 에서 import 해도 찾도록 절대경로.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# selection 결과 폴더는 프로토콜에 따라 갈린다 (auc_table.py 와 같은 규약).
#   data_selection           : held-out 평가 (현재 프로토콜)
#   data_selection_insample  : 점수 집합에서 그대로 평가한 예전 결과
# 정확도(acc)·a0 는 SELECTION_DIR 에서, **시간은 TIMING_DIR 에서** 읽는다.
#   시간은 Shapley pkl 에 기록된 값이라 프로토콜과 무관하고, 예전 파일만 A6000 에서
#   측정됐음이 확인돼 있어 기본값을 data_selection_insample 로 둔다.
SELECTION_DIR = os.environ.get("SELECTION_DIR", "data_selection")
TIMING_DIR    = os.environ.get("TIMING_DIR", "data_selection_insample")
BASE      = os.path.join(ROOT, "freeshap_res", SELECTION_DIR)
BASE_TIME = os.path.join(ROOT, "freeshap_res", TIMING_DIR)

def read_time(path):
    """txt 에서 (gpu_model, shapley_computation 초). 없으면 (None, None)."""
    gpu, sec = None, None
    try:
        with open(path) as f:
            for line in f:
                if gpu is None and "gpu_model:" in line:
                    gpu = line.split("gpu_model:")[1].strip()
                if sec is None and "shapley_computation:" in line:
                    m = re.search(r"shapley_computation:\s*([\d.]+)s", line)
                    if m: sec = float(m.group(1))
                if gpu is not None and sec is not None:
                    break
    except OSError:
        return None, None
    return gpu, sec

def a6000_time(paths):
    """후보 경로들(신형식 우선순) 중 A6000 에서 잰 첫 시간. 없으면 None."""
    for p in paths:
        if not os.path.exists(p):
            continue
        gpu, sec = read_time(p)
        if gpu and "A6000" in gpu and sec is not None:
            return sec
    return None

def _num(ds, base):
    """폴더에 맞는 n: held-out selection 폴더면 NUM_SEL, 그 외(시간 폴더·in-sample)는 full-size."""
    if base == BASE and SELECTION_DIR == "data_selection":
        return NUM_SEL.get(ds, 5000)
    return NUM.get(ds, 5000)

def inv_paths(ds, s, base=None):
    base = BASE if base is None else base
    n, v = _num(ds, base), VAL.get(ds, 1000)
    return [f"{base}/{ds}/inv/predictions/bert_seed{s}_num{n}_val{v}_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt"]

def eigen_paths(ds, s, r, base=None):
    base = BASE if base is None else base
    n, v = _num(ds, base), VAL.get(ds, 1000)
    head = f"{base}/{ds}/eigen/predictions/bert_seed{s}_num{n}_val{v}"
    tail = "_signFalse_earlystopTrue_tmc500_predictions.txt"
    return [  # 신형식(eig{r}.0, eigeps 포함) -> 신형식(eig{r}) -> 구형식(lam, eigeps 없음)
        f"{head}_eig{r}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32{tail}",
        f"{head}_eig{r}_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32{tail}",
        f"{head}_eig{r}.0_lam1e-02_cholesky_float32{tail}",
        f"{head}_eig{r}_lam1e-02_cholesky_float32{tail}",
    ]

def fmt(x):
    return "None" if x is None else f"{x:.4f}"

def print_time():
    lines = ["# ==== bert full-size (n5000; rte 2490, mrpc 3668) — A6000 shapley computation 초 ====",
         "# seeds 순서: [2024, 2025, 2026] / eigen 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             "# λ=1e-2, eigeps=1e-8 (구형 파일명 포함). A6000 아님/파일 없음 = None",
             "# 재생성: python jitter_exp/collect_speedup_times.py", ""]
    for ds in DS:
        inv = [a6000_time(inv_paths(ds, s, BASE_TIME)) for s in SEEDS]
        lines.append(f"{ds}_inv   = [{', '.join(fmt(x) for x in inv)}]")
        rows = []
        for r in RANKS:
            row = [a6000_time(eigen_paths(ds, s, r, BASE_TIME)) for s in SEEDS]
            rows.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append(f"{ds}_eigen = [")
        lines.extend(rows)
        lines.append("]")
        lines.append("")
    print("\n".join(lines))


# ==================== acc 수집 (셀 2용): python ... acc ====================
ACC_K = 20   # selection top-k 평균 구간 (1~ACC_K %) — ER/파레토가 곡선 그림(1~20%)과 창을 공유

def read_inv_top_mean(path):
    """'inv mode lambda' 블록의 top: 리스트 앞 ACC_K 개(1~ACC_K%) 평균. 없으면 None."""
    try:
        txt = open(path).read()
    except OSError:
        return None
    # txt 는 append 모드 -> 마지막 블록이 현재 SV pkl 과 일치 (indices 로 검증)
    ms = re.findall(r"inv mode lambda[^\n]*\ntop:\s*\n?\[([^\]]+)\]", txt)
    if not ms:
        return None
    vals = [int(x) for x in re.findall(r"\d+", ms[-1])][:ACC_K]
    return sum(vals) / len(vals) if vals else None

def a6000_acc(paths):
    """시간과 동일한 우선순위로, (존재하는 첫 파일의) inv-mode top 평균."""
    for p in paths:
        if os.path.exists(p):
            v = read_inv_top_mean(p)
            if v is not None:
                return v
    return None

def print_acc():
    lines = ["# ==== selection 1~20% top-k 곡선의 평균 정확도 (val acc x 1e4, inv-mode 예측) ====",
             "# seeds 순서: [2024, 2025, 2026] / eigen 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             "# = point-addition 곡선의 AUC (우리 리포트의 'Selection AUC k1-100%' x 1e4 와 동일 정의)",
             "# 재생성: python jitter_exp/collect_speedup_times.py acc", ""]
    for ds in DS:
        inv = [a6000_acc(inv_paths(ds, s)) for s in SEEDS]
        lines.append(f"{ds}_inv_acc   = [{', '.join(fmt(x) for x in inv)}]")
        lines.append(f"{ds}_eigen_acc = [")
        for r in RANKS:
            row = [a6000_acc(eigen_paths(ds, s, r)) for s in SEEDS]
            lines.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append("]")
        lines.append("")
    print("\n".join(lines))

if __name__ == "__main__":
    import sys



# ==================== a0 수집 (base accuracy): python ... a0 ====================
def _base_path(pred_path):
    return pred_path.replace("/predictions/", "/base_accuracy/").replace("_predictions.txt", "_base.txt")

def read_base(path):
    """base txt 의 'inv_base_accuracy:' 다음 숫자 (val acc x 1e4). 없으면 None."""
    try:
        txt = open(path).read()
    except OSError:
        return None
    m = re.search(r"inv_base_accuracy:\s*\n?\s*(\d+)", txt)
    return float(m.group(1)) if m else None

def a0_of(paths):
    for p in paths:
        bp = _base_path(p)
        if os.path.exists(bp):
            v = read_base(bp)
            if v is not None:
                return v
    return None

def print_a0():
    lines = ["# ==== 0% selection 의 base accuracy a0 (val acc x 1e4, inv-mode) ====",
             "# seeds 순서: [2024, 2025, 2026] / eigen 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             "# excess rate 용: (AUC - a0) 비율의 분모/분자 보정에 사용",
             "# 재생성: python jitter_exp/collect_speedup_times.py a0", ""]
    for ds in DS:
        inv = [a0_of(inv_paths(ds, s)) for s in SEEDS]
        lines.append(f"{ds}_inv_a0   = [{', '.join(fmt(x) for x in inv)}]")
        lines.append(f"{ds}_eigen_a0 = [")
        for r in RANKS:
            # a0 는 빈 훈련셋(상수 0 예측)의 val 정확도라 method/rank 무관 -> 파일 없으면 inv 값 사용
            row = [a0_of(eigen_paths(ds, s, r)) if a0_of(eigen_paths(ds, s, r)) is not None else inv[j]
                   for j, s in enumerate(SEEDS)]
            lines.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append("]")
        lines.append("")
    print("\n".join(lines))


# ==================== llama n2000 시간 수집: python ... llama2000 ====================
VAL2000 = {"sst2": 872, "mrpc": 408, "rte": 277}   # 그 외 1000 (n2000 캠페인)

def llama_inv_paths_n(ds, s, n):
    v = VAL2000.get(ds, 1000)   # val 규약은 n 과 무관 (sst2 872 / mrpc 408 / rte 277 / 그 외 1000)
    return [f"{BASE}/{ds}/inv/predictions/llama_seed{s}_num{n}_val{v}_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt"]

def llama_eigen_paths_n(ds, s, r, n):
    v = VAL2000.get(ds, 1000)
    return [f"{BASE}/{ds}/eigen/predictions/llama_seed{s}_num{n}_val{v}_eig{r}.0"
            f"_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt"]

def llama_inv_paths(ds, s):            return llama_inv_paths_n(ds, s, 2000)
def llama_eigen_paths(ds, s, r):       return llama_eigen_paths_n(ds, s, r, 2000)
def llama1000_inv_paths(ds, s):        return llama_inv_paths_n(ds, s, 1000)
def llama1000_eigen_paths(ds, s, r):   return llama_eigen_paths_n(ds, s, r, 1000)

def print_llama2000():
    lines = ["# ==== llama n2000 — A6000 shapley computation 초 ====",
             "# seeds 순서: [2024, 2025, 2026] / eigen_time 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             "# λ=1e-2, eigeps=1e-8. A6000 아님/파일 없음 = None",
             "# 재생성: python jitter_exp/collect_speedup_times.py llama2000", ""]
    for ds in DS:
        inv = [a6000_time(llama_inv_paths(ds, s)) for s in SEEDS]
        lines.append(f"{ds}_inv_time   = [{', '.join(fmt(x) for x in inv)}]")
        lines.append(f"{ds}_eigen_time = [")
        for r in RANKS:
            row = [a6000_time(llama_eigen_paths(ds, s, r)) for s in SEEDS]
            lines.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append("]")
        lines.append("")
    print("\n".join(lines))


def print_llama2000_acc():
    lines = ["# ==== llama n2000 — selection 1~20% top-k 곡선의 평균 정확도 (val acc x 1e4, inv-mode 예측) ====",
             "# seeds 순서: [2024, 2025, 2026] / eigen 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             "# 재생성: python jitter_exp/collect_speedup_times.py llama2000_acc", ""]
    for ds in DS:
        inv = [a6000_acc(llama_inv_paths(ds, s)) for s in SEEDS]
        lines.append(f"{ds}_inv_acc   = [{', '.join(fmt(x) for x in inv)}]")
        lines.append(f"{ds}_eigen_acc = [")
        for r in RANKS:
            row = [a6000_acc(llama_eigen_paths(ds, s, r)) for s in SEEDS]
            lines.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append("]")
        lines.append("")
    print("\n".join(lines))

def print_llama2000_a0():
    lines = ["# ==== llama n2000 — 0% selection 의 base accuracy a0 (val acc x 1e4, inv-mode) ====",
             "# seeds 순서: [2024, 2025, 2026] / eigen 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             "# 재생성: python jitter_exp/collect_speedup_times.py llama2000_a0", ""]
    for ds in DS:
        inv = [a0_of(llama_inv_paths(ds, s)) for s in SEEDS]
        lines.append(f"{ds}_inv_a0   = [{', '.join(fmt(x) for x in inv)}]")
        lines.append(f"{ds}_eigen_a0 = [")
        for r in RANKS:
            # a0 는 method/rank 무관 -> eigen 파일 없으면 같은 seed 의 inv a0 사용
            row = [a0_of(llama_eigen_paths(ds, s, r)) if a0_of(llama_eigen_paths(ds, s, r)) is not None else inv[j]
                   for j, s in enumerate(SEEDS)]
            lines.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append("]")
        lines.append("")
    print("\n".join(lines))


# ==================== llama full-size(n5000; rte 2490, mrpc 3668): python ... llama5000[_acc|_a0] ====
def llama5k_inv_paths(ds, s, base=None):
    """bert 의 inv_paths 와 같은 규약: acc/a0 는 BASE(SELECTION_DIR), 시간은 BASE_TIME 에서."""
    base = BASE if base is None else base
    n, v = _num(ds, base), VAL.get(ds, 1000)
    return [f"{base}/{ds}/inv/predictions/llama_seed{s}_num{n}_val{v}_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt"]

def llama5k_eigen_paths(ds, s, r, base=None):
    base = BASE if base is None else base
    n, v = _num(ds, base), VAL.get(ds, 1000)
    return [f"{base}/{ds}/eigen/predictions/llama_seed{s}_num{n}_val{v}_eig{r}.0"
            f"_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt"]

def _print_generic(header, regen, inv_paths_fn, eigen_paths_fn, value_fn, suffix):
    lines = [header,
             "# seeds 순서: [2024, 2025, 2026] / eigen 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             f"# 재생성: python jitter_exp/collect_speedup_times.py {regen}", ""]
    for ds in DS:
        inv = [value_fn(inv_paths_fn(ds, s)) for s in SEEDS]
        _fb = (suffix == "a0")   # a0 는 method/rank 무관 -> 없으면 inv 값으로 폴백
        lines.append(f"{ds}_inv_{suffix}   = [{', '.join(fmt(x) for x in inv)}]")
        lines.append(f"{ds}_eigen_{suffix} = [")
        for r in RANKS:
            row = [value_fn(eigen_paths_fn(ds, s, r)) for s in SEEDS]
            if _fb:
                row = [v if v is not None else inv[j] for j, v in enumerate(row)]
            lines.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append("]")
        lines.append("")
    print("\n".join(lines))

def print_llama5000():
    _print_generic("# ==== llama full-size (n5000; rte 2490, mrpc 3668) — A6000 shapley computation 초 ====",
                   "llama5000", lambda ds, s: llama5k_inv_paths(ds, s, BASE_TIME),
                   lambda ds, s, r: llama5k_eigen_paths(ds, s, r, BASE_TIME), a6000_time, "time")

def print_llama5000_acc():
    _print_generic("# ==== llama full-size — selection 1~20% top-k 평균 정확도 (val acc x 1e4, inv-mode) ====",
                   "llama5000_acc", llama5k_inv_paths, llama5k_eigen_paths, a6000_acc, "acc")

def print_llama5000_a0():
    _print_generic("# ==== llama full-size — 0% selection base accuracy a0 (val acc x 1e4, inv-mode) ====",
                   "llama5000_a0", llama5k_inv_paths, llama5k_eigen_paths, a0_of, "a0")


# ==================== llama n1000: python ... llama1000[_acc|_a0] ====================
# n=1000 스케일 포인트 (speedup_er 의 n 별 열 재료). 러너: N=1000 DO_REMOVAL=0 DO_WLD=0 sh n05.sh
def print_llama1000():
    _print_generic_ds("# ==== llama n1000 — A6000 shapley computation 초 ====",
                      "llama1000", DS, llama1000_inv_paths, llama1000_eigen_paths, a6000_time, "time")

def print_llama1000_acc():
    _print_generic_ds("# ==== llama n1000 — selection 1~20% top-k 곡선의 평균 정확도 (val acc x 1e4, inv-mode 예측) ====",
                      "llama1000_acc", DS, llama1000_inv_paths, llama1000_eigen_paths, a6000_acc, "acc")

def print_llama1000_a0():
    _print_generic_ds("# ==== llama n1000 — 0% selection 의 base accuracy a0 (val acc x 1e4, inv-mode) ====",
                      "llama1000_a0", DS, llama1000_inv_paths, llama1000_eigen_paths, a0_of, "a0")


# ==================== resnet cifar10 n5000: python ... resnet5000[_acc|_a0] ====================
# vision n5000 캠페인은 RTX 3090 에서 재측정됨 (speedup 비교 GPU 통일 + inv λ 를 1e-6 로 통일).
#   옛 2080 Ti / inv λ=1e-2 산출물은 backup_2080ti/ 로 옮겨져 경로에서 잡히지 않는다.
#   아직 3090 으로 안 돈 셀은 None 으로 남는다 (러너: n03.sh).
def resnet5k_inv_paths(ds, s):
    return [f"{BASE}/cifar10/inv/predictions/resnet_seed{s}_num5000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt"]

def resnet5k_eigen_paths(ds, s, r):
    return [f"{BASE}/cifar10/eigen/predictions/resnet_seed{s}_num5000_val1000_eig{r}.0"
            f"_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500_predictions.txt"]

def t3090_time(paths):
    """RTX 3090 에서 잰 shapley_computation 초 (vision 용 GPU 필터).

    다른 GPU 로 잰 값이면 speedup 비교가 오염되므로 None 을 돌려 빈 칸으로 남긴다.
    """
    for p in paths:
        if not os.path.exists(p):
            continue
        gpu, sec = read_time(p)
        if gpu and "3090" in gpu and sec is not None:
            return sec
    return None

def _print_generic_ds(header, regen, dslist, inv_paths_fn, eigen_paths_fn, value_fn, suffix):
    lines = [header,
             "# seeds 순서: [2024, 2025, 2026] / eigen 행 순서: rank [1, 5, 10, 15, 20, 25, 30] (%)",
             f"# 재생성: python jitter_exp/collect_speedup_times.py {regen}", ""]
    for ds in dslist:
        inv = [value_fn(inv_paths_fn(ds, s)) for s in SEEDS]
        _fb = (suffix == "a0")   # a0 는 method/rank 무관 -> 없으면 inv 값으로 폴백
        lines.append(f"{ds}_inv_{suffix}   = [{', '.join(fmt(x) for x in inv)}]")
        lines.append(f"{ds}_eigen_{suffix} = [")
        for r in RANKS:
            row = [value_fn(eigen_paths_fn(ds, s, r)) for s in SEEDS]
            if _fb:
                row = [v if v is not None else inv[j] for j, v in enumerate(row)]
            lines.append(f"    [{', '.join(fmt(x) for x in row)}],   # rank {r}%")
        lines.append("]")
        lines.append("")
    print("\n".join(lines))

def print_resnet5000():
    _print_generic_ds("# ==== resnet cifar10 n5000 — RTX 3090 shapley computation 초 (inv 는 lam1e-06) ====",
                      "resnet5000", ["cifar10"], resnet5k_inv_paths, resnet5k_eigen_paths, t3090_time, "time")

def print_resnet5000_acc():
    _print_generic_ds("# ==== resnet cifar10 n5000 — selection 1~20% top-k 평균 정확도 (val acc x 1e4, inv-mode) ====",
                      "resnet5000_acc", ["cifar10"], resnet5k_inv_paths, resnet5k_eigen_paths, a6000_acc, "acc")

def print_resnet5000_a0():
    _print_generic_ds("# ==== resnet cifar10 n5000 — 0% selection base accuracy a0 (val acc x 1e4, inv-mode) ====",
                      "resnet5000_a0", ["cifar10"], resnet5k_inv_paths, resnet5k_eigen_paths, a0_of, "a0")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "acc":
        print_acc()
    elif len(sys.argv) > 1 and sys.argv[1] == "a0":
        print_a0()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama2000":
        print_llama2000()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama2000_acc":
        print_llama2000_acc()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama2000_a0":
        print_llama2000_a0()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama5000":
        print_llama5000()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama5000_acc":
        print_llama5000_acc()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama5000_a0":
        print_llama5000_a0()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama1000":
        print_llama1000()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama1000_acc":
        print_llama1000_acc()
    elif len(sys.argv) > 1 and sys.argv[1] == "llama1000_a0":
        print_llama1000_a0()
    elif len(sys.argv) > 1 and sys.argv[1] == "resnet5000":
        print_resnet5000()
    elif len(sys.argv) > 1 and sys.argv[1] == "resnet5000_acc":
        print_resnet5000_acc()
    elif len(sys.argv) > 1 and sys.argv[1] == "resnet5000_a0":
        print_resnet5000_a0()
    else:
        print_time()

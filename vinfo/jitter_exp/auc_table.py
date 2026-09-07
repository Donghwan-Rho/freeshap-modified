# -*- coding: utf-8 -*-
"""selection / removal / wrong-label-detection 곡선의 요약 표 생성기.

(sel_rem_auc_table.py + wld_auc_table.py 를 합친 것. 세 태스크가 같은 정의를
 쓰므로 한 파일에서 동일한 표 형식으로 뽑는다.)

지표 정의: 검사 구간에서의 **평균값** (mean accuracy / mean detection rate, [0,1] 스케일).
  곡선이 1%p 등간격 격자라 "정규화된 곡선 아래 면적"(사다리꼴적분/구간길이) 과 사실상 같지만,
  AUC 는 관례상 ROC-AUC 를 뜻해 오해 소지가 있으므로 표기는 평균(mean)으로 통일한다.
  값은 seed 2024/2025/2026 각각의 평균을 구한 뒤 mean ± std(ddof=1).
  ※ random baseline 이 창 길이에 따라 달라지므로(wld: 1~20% 0.105 vs 1~100% 0.505)
    서로 다른 창의 값끼리는 직접 비교하지 말 것.

태스크별 규약 (창 index 가 다르니 주의):
  selection : 값 = val accuracy. 곡선 index 0 = top 1% 추가.  창 1~L% = [:L].
              높을수록 좋음.
  removal   : 값 = val accuracy. 곡선 index i = i% 제거.       창 1~L% = [1:L+1].
              index 0 (0% 제거) 은 방법 무관 공통 상수라 창에서 빼고 참고값으로만 출력.
              곡선이 100 점(0~99%) 이라 L=100 요청은 99 로 클램프. **낮을수록 좋음**.
  wld       : 값 = detection rate. 곡선 index 0 = inspect 1%.  창 1~L% = [:L].
              높을수록 좋음. random baseline = mean(p/100) 을 같이 출력.

데이터 출처 (전부 결과 파일 직접 파싱 — 노트북 리터럴에 의존하지 않음):
  selection : data_selection/{ds}/{inv|eigen}/predictions/*_predictions.txt  "top:"
  removal   : data_removing/{ds}/{inv|eigen}/predictions/*_predictions.txt   "top_removal:"
              (둘 다 eigen 파일에서도 inv-mode 블록만 사용)
  wld       : wrong_label_detection/{ds}/{inv|eigen}/predictions/*_detection.{pkl,txt}
              (collect_wld_curves 의 glob + pkl 우선 로직 재사용)

모델: bert / llama 는 NLP 7종, resnet 은 cifar10 (vision) 한 패널.
  자료가 없는 칸은 '--', seed 가 3개 미만인 칸은 있는 seed 로만 평균 내고 표 아래에 목록을 찍는다
  (실험이 아직 도는 중에도 실행 가능 — 대신 seed 수가 다른 칸끼리 비교하면 안 된다).

사용:
  python jitter_exp/auc_table.py                            # 3태스크 x bert/llama/resnet x 1~20%,1~100%
  python jitter_exp/auc_table.py --task wld
  python jitter_exp/auc_table.py --model resnet
  python jitter_exp/auc_table.py --task selection removal --pct 20
  python jitter_exp/auc_table.py --task removal --model bert --pct 20 --latex
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_wld_curves as wld_src   # inv_file / eigen_file / det_rates (pkl 우선)

SEEDS = (2024, 2025, 2026)
RANKS = (1, 5, 10, 15, 20, 25, 30)
NUM = {"rte": 2490, "mrpc": 3668}
VAL = {"sst2": 872, "mrpc": 408, "rte": 277}
PANELS = [("sst2", "SST-2"), ("mnli", "MNLI"), ("ag_news", "AG News"),
          ("mr", "MR"), ("qqp", "QQP"), ("rte", "RTE"), ("mrpc", "MRPC")]
# 모델별 데이터셋. bert/llama 는 NLP 7종, resnet(vision)은 cifar10 하나뿐.
PANELS_BY_MODEL = {"resnet": [("cifar10", "CIFAR-10")]}
MODELS = ("bert", "llama", "resnet")
SCALE = 10000.0
TASKS = ("selection", "removal", "wld")


def panels_for(model):
    return PANELS_BY_MODEL.get(model, PANELS)

# 이 파일은 jitter_exp/ 에 있으므로 그 부모가 vinfo 루트.
# 노트북이 reports_*/ 안에서 실행돼도 CWD 와 무관하게 결과를 찾도록 절대경로로 잡는다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = {"selection": os.path.join(ROOT, "freeshap_res", "data_selection"),
        "removal": os.path.join(ROOT, "freeshap_res", "data_removing")}
FIELD = {"selection": r"top", "removal": r"top[ _]removal[^\n]*"}
HIGHER_IS_BETTER = {"selection": True, "removal": False, "wld": True}
SPAN = {"selection": "top 1~{L}% 추가", "removal": "1~{L}% 제거", "wld": "inspect 1~{L}%"}
UNIT = {"selection": "val accuracy", "removal": "val accuracy", "wld": "detection rate"}

# ---- LaTeX caption/label 용 어휘 ----
# 모델명은 configs/dshap/*/ 의 yaml 에 적힌 것과 동일하게 (bert: ntk_prompt.yaml 의 model,
#   llama: ntk_llama.yaml 의 model_name, resnet: ntk_vision.yaml 의 model).
MODEL_TEX = {"bert": r"\texttt{bert-base-uncased}",
             "llama": r"\texttt{meta-llama/Llama-2-7b-hf}",
             "resnet": r"\texttt{resnet-18\_pretrained}"}
# caption 본문: "The mean {METRIC} of {MODEL} when the {AXIS} is 1-{L}\%."
METRIC_TEX = {"selection": "accuracy", "removal": "accuracy", "wld": "detection rate"}
AXIS_TEX = {"selection": "selection percentage",
            "removal": "removal percentage",
            "wld": "inspection percentage"}


def tex_caption(task, model, L):
    return (f"The mean {METRIC_TEX[task]} of {MODEL_TEX.get(model, model)} "
            f"when the {AXIS_TEX[task]} is 1-{L}\\%.")


def tex_label(task, model, L):
    metric = METRIC_TEX[task].replace(" ", "_")
    return f"tab:mean_{metric}_{task}_{model}_{L}"


# ---------------------------------------------------------------- 곡선 읽기
def _acc_curve(path, task):
    """selection/removal txt 의 inv-mode 블록 곡선 (없으면 None).

    txt 는 append 모드라 재실행분이 누적된다. SV pkl 이 중간에 재생성된 경우 블록마다
    값이 다른데 **마지막 블록이 현재 pkl 과 일치**하므로(indices 파일로 검증) 마지막을 쓴다.
    (2026-08 기준 selection 의 llama ag_news/mr/mrpc eigen 60개가 여기 해당)
    """
    try:
        txt = open(path).read()
    except OSError:
        return None
    ms = re.findall(rf"inv mode lambda[^\n]*\n{FIELD[task]}:\s*\n?\[([^\]]+)\]", txt)
    return None if not ms else [int(x) for x in re.findall(r"\d+", ms[-1])]


def _acc_candidates(task, model, ds, s, rank):
    """존재할 수 있는 파일명 변형들 (신형 -> 구형 순)."""
    base, n, v = BASE[task], NUM.get(ds, 5000), VAL.get(ds, 1000)
    tail = "_signFalse_earlystopTrue_tmc500_predictions.txt"
    if rank is None:
        return [f"{base}/{ds}/inv/predictions/{model}_seed{s}_num{n}_val{v}_lam1e-06{tail}"]
    head = f"{base}/{ds}/eigen/predictions/{model}_seed{s}_num{n}_val{v}"
    return [f"{head}_eig{rank}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32{tail}",
            f"{head}_eig{rank}_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32{tail}",
            f"{head}_eig{rank}.0_lam1e-02_cholesky_float32{tail}",
            f"{head}_eig{rank}_lam1e-02_cholesky_float32{tail}"]


def get_curve(task, model, ds, s, rank):
    """rank=None 이면 inv(FreeShap). 없으면 None."""
    if task == "wld":
        f = (wld_src.inv_file(model, ds, s) if rank is None
             else wld_src.eigen_file(model, ds, s, rank))
        return None if not f else wld_src.det_rates(f)
    for p in _acc_candidates(task, model, ds, s, rank):
        if os.path.exists(p):
            c = _acc_curve(p, task)
            if c:
                return c
    return None


# ---------------------------------------------------------------- AUC 계산
def window(c, task, L):
    a = np.array(c, dtype=float) / SCALE
    seg = a[1:L + 1] if task == "removal" else a[:L]
    return None if len(seg) == 0 else seg.mean()


def auc_cell(task, model, ds, rank, L):
    vs = []
    for s in SEEDS:
        c = get_curve(task, model, ds, s, rank)
        if c is not None:
            w = window(c, task, L)
            if w is not None:
                vs.append(w)
    if not vs:
        return None
    a = np.array(vs)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def zero_removal(model, ds):
    """removal 곡선 index 0 (0% 제거) — 같은 seed 안에서는 방법 무관 상수.

    (mean, 방법간 최대편차). 편차는 seed 간 차이가 섞인 값이라 0 이 아닐 수 있다.
    """
    vals = []
    for rank in [None] + list(RANKS):
        for s in SEEDS:
            c = get_curve("removal", model, ds, s, rank)
            if c:
                vals.append(c[0] / SCALE)
    return None if not vals else (float(np.mean(vals)), float(np.max(vals) - np.min(vals)))


# ---------------------------------------------------------------- 출력
def rank_labels():
    """첫 rank 열만 'Rank 1%', 나머지는 '5%', '10%' … (표 안에서 한 번만 단어를 쓴다)."""
    return [f"Rank {r}%" if i == 0 else f"{r}%" for i, r in enumerate(RANKS)]


def table(task, model, L, latex=False, caption=None):
    panels = panels_for(model)
    cols = ["FreeShap"] + rank_labels()
    better = "높을수록 좋음" if HIGHER_IS_BETTER[task] else "낮을수록 좋음"
    W, width = 14, 11 + 14 * (len(RANKS) + 1)
    print()
    print("=" * width)
    print(f"{model} — {task}: {SPAN[task].format(L=L)} 구간의 평균 {UNIT[task]}, "
          f"mean ± std over seeds {list(SEEDS)}   [{better}]")
    if task == "wld":
        print(f"random baseline = {np.arange(1, L + 1).mean() / 100.0:.4f}")
    print("=" * width)
    print(f"{'dataset':>9s} " + "".join(f"{c:>{W}s}" for c in cols))
    colacc = {c: [] for c in cols}
    missing, partial = [], []          # 자료 없는 셀 / seed 가 덜 찬 셀
    for ds, disp in panels:
        line = f"{disp:>9s} "
        for name, rank in zip(cols, [None] + list(RANKS)):
            c = auc_cell(task, model, ds, rank, L)
            if c is None:
                line += f"{'--':>{W}s}"
                missing.append(f"{disp}/{name}")
            else:
                line += f"{c[0]:>8.4f}±{c[1]:.3f}"
                colacc[name].append(c[0])
                if c[2] < len(SEEDS):
                    partial.append(f"{disp}/{name}({c[2]})")
        print(line)
    print("-" * width)
    print(f"{'평균':>8s} " + "".join(
        f"{'--':>{W}s}" if not colacc[c] else f"{np.mean(colacc[c]):>{W}.4f}" for c in cols))
    if task == "removal":
        z = [t for t in (zero_removal(model, ds) for ds, _ in panels) if t]
        if z:
            print(f"  참고: 0% 제거 시 정확도(창에서 제외, 같은 seed 안에서는 방법 무관 상수) "
                  f"평균 {np.mean([t[0] for t in z]):.4f}, "
                  f"최대편차 {max(t[1] for t in z):.4f}")
    # 자료가 덜 쌓인 상태에서도 실행되므로, 어디가 비었는지 반드시 함께 알린다.
    if missing:
        print(f"  ※ 자료 없음({len(missing)}칸, '--'): " + ", ".join(missing))
    if partial:
        print(f"  ※ seed 일부만({len(partial)}칸, 괄호=seed 수): " + ", ".join(partial)
              + "  -> seed 수가 다른 칸끼리는 직접 비교 주의")
    if latex:
        print()
        print(f"% {model} {task}, {SPAN[task].format(L=L)} 구간의 평균 {UNIT[task]} "
              f"({better}), seeds {list(SEEDS)} 평균")
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\scriptsize")
        print(r"\begin{tabular}{l" + "c" * len(cols) + "}")
        print(r"\toprule")
        print("Dataset & FreeShap & " + " & ".join(rank_labels()).replace("%", r"\%") + r" \\")
        print(r"\midrule")
        for ds, disp in panels:
            cs = [auc_cell(task, model, ds, rank, L) for rank in [None] + list(RANKS)]
            print(f"{disp} & " + " & ".join("--" if c is None else f"{c[0]:.4f}" for c in cs) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(rf"\caption{{{caption if caption is not None else tex_caption(task, model, L)}}}")
        print(rf"\label{{{tex_label(task, model, L)}}}")
        print(r"\end{table}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="selection/removal/wld 곡선의 요약 표")
    ap.add_argument("--task", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--model", nargs="+", default=list(MODELS), choices=MODELS,
                    help="bert/llama = NLP 7종, resnet = cifar10 (기본: 전부)")
    ap.add_argument("--pct", type=int, nargs="+", default=[20, 100],
                    help="검사 구간 상한 (기본 20 과 100 둘 다)")
    ap.add_argument("--latex", action="store_true",
                    help="LaTeX table 환경(booktabs, scriptsize, caption/label)도 출력")
    ap.add_argument("--caption", default=None,
                    help="LaTeX \\caption 내용을 직접 지정 (생략하면 태스크/모델/구간에서 자동 생성)")
    a = ap.parse_args()
    for task in a.task:
        for model in a.model:
            for L in a.pct:
                # removal 곡선은 100 점(0~99%) 뿐이라 1~100% 요청은 1~99% 로 클램프
                table(task, model, min(L, 99) if task == "removal" else L,
                      latex=a.latex, caption=a.caption)

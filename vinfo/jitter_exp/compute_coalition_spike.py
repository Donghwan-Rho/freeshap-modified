# -*- coding: utf-8 -*-
"""coalition-스파이크 사전계산: 부분집합(coalition) 크기 n 별로
|acc_approx(S) − acc_inv(S)| (val acc, %p) 를 rank×method 그리드로 계산해 npz 저장.

배경: 저랭크 근사는 |S| ≈ d(=rank%×N) 보간 임계에서 부분집합 회귀가 준특이가 되어
유틸리티가 크게 어긋남 (double-descent 스파이크). TMC 는 모든 coalition 크기를
적분하므로 이 오염 구간의 총량이 SV 오차를 지배 — rank↑ 시 nys SV 오차가
정체/상승하는 원인. build_eigen_pinv_lev_report p7 이 이 npz 를 읽어 그림.

크기 그리드: rank 별 d 를 중심으로 배수 {0.3,0.6,0.85,1.0,1.15,1.4,2.0} + 공통 anchor.
inv 예측은 (size, trial) 별 캐시 (rank/method 무관 재사용).

모델 로드 필요 (bert/resnet 은 CPU 가능, llama 는 llama 있는 서버에서).
사용: python jitter_exp/compute_coalition_spike.py --dataset qqp [--model bert]
출력: ./jitter_exp/coalition_spike/{model}_{dataset}_seed{S}.npz
"""
import os, sys, glob, pickle, argparse, re
import numpy as np
import torch
import yaml

VINFO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(VINFO)
sys.path.insert(0, VINFO)
sys.path.insert(0, os.path.join(VINFO, "jitter_exp"))

from dataset import *          # noqa: F401,F403  — YAML tag 등록
from probe import *            # noqa: F401,F403
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: F401
from entks.ntk_regression import shapleyNTKRegression
try:
    from vision.vision_dataset import VisionReader, VisionDataset  # noqa: F401
    from vision.vision_probe   import NTKVisionProbe                # noqa: F401
except ImportError:
    pass

MODEL2CONFIG = {"bert": "ntk_prompt", "llama": "ntk_llama", "resnet": "ntk_vision"}
NUM = 2000
METHODS = ["eigen", "nystrom_pinv", "nystrom_lev"]
FACTORS = [0.3, 0.6, 0.85, 1.0, 1.15, 1.4, 2.0]   # d 대비 coalition 크기 배수
ANCHORS = [50, 1000, 2000]                          # 공통 기준점


def out_path(model, dataset, seed):
    return f"./jitter_exp/coalition_spike/{model}_{dataset}_seed{seed}.npz"


def setup(dataset, seed, val, model_name, config):
    """yaml→probe/dataset, NTK 캐시 로드, train/val 라벨. (CPU fallback 포함)"""
    torch.manual_seed(seed); np.random.seed(seed)
    yaml_path = f"../configs/dshap/{dataset}/{config}.yaml"
    txt = open(yaml_path).read()
    if not torch.cuda.is_available():
        txt = txt.replace("device: cuda:0", "device: cpu")
        print("[info] CUDA 없음 → device: cpu")
    ya = yaml.load(txt, Loader=yaml.Loader)
    lds, probe = ya["dataset"], ya["probe_com"]
    if hasattr(lds, "label_word_list") and hasattr(probe.model, "init"):
        probe.model.init(lds.label_word_list)   # prompt 모델(bert/llama)만
    ntk_path = f"./freeshap_res/ntk/{dataset}/{model_name}_seed{seed}_num{NUM}_val{val}_signFalse.pkl"
    bundle = pickle.load(open(ntk_path, "rb"))
    sidx, svidx = np.array(bundle["sampled_idx"]), np.array(bundle["sampled_val_idx"])
    probe.get_cached_ntk(bundle["ntk"])
    probe.get_train_labels(lds.get_idx_dataset(sidx, split="train"))
    val_set = lds.get_idx_dataset(svidx, split="val")
    y_val = np.array([i["label"] for i in val_set])
    return probe, y_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", type=str, default="bert")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024])
    ap.add_argument("--ranks", type=float, nargs="+", default=[1, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--trials", type=int, default=4, help="coalition 크기당 랜덤 부분집합 수")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    config = args.config or MODEL2CONFIG.get(args.model, "ntk_prompt")

    for seed in args.seeds:
        outp = out_path(args.model, args.dataset, seed)
        if os.path.exists(outp) and not args.overwrite:
            print(f"[skip] {outp} 존재"); continue
        g = glob.glob(f"./freeshap_res/ntk/{args.dataset}/{args.model}_seed{seed}_num{NUM}_val*_signFalse.pkl")
        if not g:
            print(f"[skip] seed{seed}: NTK 캐시 없음"); continue
        val = int(re.search(r"_val(\d+)_", os.path.basename(g[0])).group(1))

        print(f"[setup] {args.dataset} {args.model} seed{seed} val{val} — probe/NTK 로드...")
        probe, y_val = setup(args.dataset, seed, val, args.model, config)
        ntk = probe.ntk; n = ntk.size(2)

        # ---- 크기 그리드 (rank 별 d 중심 + anchor 합집합) ----
        rank_sizes = {}
        all_sizes = set(a for a in ANCHORS if a <= n)
        for r in args.ranks:
            d = int(NUM * r / 100)
            ss = sorted({min(n, max(10, int(round(f * d)))) for f in FACTORS} |
                        {a for a in ANCHORS if a <= n})
            rank_sizes[r] = ss; all_sizes |= set(ss)
        all_sizes = sorted(all_sizes)

        # ---- inv 예측 캐시: (size, trial) — rank/method 무관 ----
        rng = np.random.RandomState(seed)
        subsets = {sz: [rng.choice(n, size=sz, replace=False) for _ in range(args.trials)]
                   for sz in all_sizes}
        acc = lambda p: float((p.argmax(1) == y_val).mean())   # noqa: E731
        inv_acc = {}
        for sz in all_sizes:
            for t in range(args.trials):
                idx = subsets[sz][t]
                k_tr = ntk[:, idx[:, None], idx]
                kr = shapleyNTKRegression(k_tr, probe.train_labels[idx], probe.num_labels,
                                          None, reg=1e-6)
                p = kr(ntk[:, n:, :][:, :, idx])
                p = p[0] if isinstance(p, tuple) else p
                inv_acc[(sz, t)] = acc(np.asarray(p.detach().cpu()))
            print(f"  [inv] n={sz} done")

        # ---- method × rank 곡선 ----
        out = {"ranks": np.array(args.ranks, float), "trials": args.trials,
               "lam": args.lam, "eps": args.eps, "num": NUM, "val": val}
        for m in METHODS:
            for r in args.ranks:
                d_act = int(NUM * r / 100)
                if m == "eigen":
                    probe.approximate("eigen")
                    probe.nystrom_use_pinv = False; probe.nystrom_use_lev = False
                    probe.set_eigen_params(rank=d_act, lam=args.lam, solver="cholesky",
                                           dtype="float32", seed=seed, floor=args.eps)
                    probe.eigen_regression = None
                    probe.prepare_eigen_regression(); reg = probe.eigen_regression
                else:
                    probe.approximate("nystrom")
                    probe.nystrom_use_pinv = (m == "nystrom_pinv")
                    probe.nystrom_use_lev = (m == "nystrom_lev")
                    probe.set_nystrom_params(d=d_act, lam=args.lam, solver="cholesky",
                                             dtype="float32", landmark_seed=seed, jitter=args.eps)
                    probe.prepare_nystrom_regression(); reg = probe.nystrom_regression
                sizes = rank_sizes[r]; vals = []
                for sz in sizes:
                    diffs = []
                    for t in range(args.trials):
                        idx = subsets[sz][t]
                        pa = np.asarray(reg(idx).detach().cpu())
                        diffs.append(abs(acc(pa) - inv_acc[(sz, t)]))
                    vals.append(100 * float(np.mean(diffs)))
                out[f"{m}_r{r:g}_sizes"] = np.array(sizes, int)
                out[f"{m}_r{r:g}_vals"] = np.array(vals, float)
                print(f"  [ok] {m:13s} r{r:g} (d={d_act}) peak={max(vals):.1f}%p")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        np.savez_compressed(outp, **out)
        print(f"[write] {outp}")
        del probe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()

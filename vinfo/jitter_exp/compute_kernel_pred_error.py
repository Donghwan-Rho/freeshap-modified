# -*- coding: utf-8 -*-
"""각 (λ,eps) 셀에서 val-set 예측 logits 를 커널별(inv / eigen / nystrom_pinv)로 계산해 npz 저장.

목적: build_pinv_selection_report 맨 뒤에 붙는 "커널 예측 상대오차" 히트맵의 사전계산.
  - 예측 = train 2000 전체로 학습 → val set 예측 (logits, val×n_class).
    task_data_selection 과 동일한 probe 경로(NTK 정규화 포함)를 재사용하되,
    최종 예측 커널만 inv / eigen / nystrom_pinv 로 각각 수행.
  - inv 기준 λ=1e-6 고정. eigen/nys_pinv 는 16칸 grid: λ∈{1e-3..1} × eps∈{1e-8..1e-2}.
  - 저장 경로: data_selection 경로에서 폴더만 kernel_prediction 으로 교체 + .npz
      eigen    -> ./jitter_exp/res/kernel_prediction/{ds}/eigen/predictions/{stem}_predictions.npz
      nys_pinv -> ./jitter_exp/nys_pinv_res/kernel_prediction/{ds}/nystrom_pinv/predictions/...
      inv      -> ./freeshap_res/kernel_prediction/{ds}/inv/predictions/...
  - 상대오차 자체는 리포트가 npz(logits)에서 계산 (Frobenius / L2).

데이터셋 라벨 로드 필요 → 인터넷 되는 노드(mathcluster)에서 실행.
사용: python jitter_exp/compute_kernel_pred_error.py --dataset rte --seeds 2024 2025 2026
"""
import os, sys, glob, pickle, argparse, types, re
import numpy as np
import torch
import yaml

VINFO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(VINFO)
sys.path.insert(0, VINFO)
sys.path.insert(0, os.path.join(VINFO, "jitter_exp"))

import build_pinv_selection_report as R      # 경로/태그 함수 재사용 (파일명 일치 보장)
from dataset import *          # noqa: F401,F403  — YAML tag (!EasyReader 등) 등록
from probe import *            # noqa: F401,F403  — YAML tag (!NTKProbe 등) 등록
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: F401 — YAML tag 등록
from entks.ntk_regression import (shapleyNTKRegression, EigenNTKRegression,
                                  NystromPinvNTKRegression, NystromLevNTKRegression)
try:  # vision(cifar10) 지원: yaml tag (!VisionReader 등) 등록 — 없으면 텍스트 전용으로 동작
    from vision.vision_dataset import VisionReader, VisionDataset  # noqa: F401
    from vision.vision_probe   import NTKVisionProbe                # noqa: F401
except ImportError:
    pass

# --model 에 따른 기본 config 매핑 (--config 로 명시 덮어쓰기 가능)
MODEL2CONFIG = {"bert": "ntk_prompt", "llama": "ntk_llama", "resnet": "ntk_vision"}

GRID_LAMS = [1e-3, 1e-2, 1e-1, 1.0]
GRID_EPS  = [1e-8, 1e-6, 1e-4, 1e-2]
RANK_PCT  = 20.0          # eig/nys_pinv rank·d 퍼센트 (파일명 태그 eig20.0 / nyspinv20.0)
NUM       = 2000
TMC       = 500
INVLAM    = "1e-06"       # inv 기준 λ (고정)


ROOTS = dict(res_root="./jitter_exp/res", pinv_root="./jitter_exp/nys_pinv_res",
             inv_root="./freeshap_res", lev_root="./jitter_exp/nys_lev_res")   # 리포트와 일치 필수

def make_a(dataset, val, model="bert", rank=RANK_PCT):
    """리포트 경로 함수(R.pred_path/inv_sv_path)가 참조하는 최소 네임스페이스."""
    return types.SimpleNamespace(
        dataset=dataset, model=model, num_train=NUM, val=val, tmc=TMC,
        rank=rank, invlam=INVLAM, **ROOTS,
    )


kp_npz_path = R.kp_npz_path   # 경로 정의는 리포트 모듈과 단일 소스 공유


def setup_probe(dataset, seed, val, model_name="bert", config=None):
    """task_data_selection 셋업 재사용: yaml→probe, NTK 캐시 로드(정규화 포함), train 라벨."""
    torch.manual_seed(seed); np.random.seed(seed)
    if config is None:
        config = MODEL2CONFIG.get(model_name, "ntk_prompt")
    yaml_path = f"../configs/dshap/{dataset}/{config}.yaml"
    txt = open(yaml_path).read()
    if not torch.cuda.is_available():
        # GPU 없는 노드(mathcluster 등): NTK 는 캐시라 커널회귀는 CPU 로 충분
        txt = txt.replace("device: cuda:0", "device: cpu")
        print("[info] CUDA 없음 → device: cpu 로 로드")
    ya = yaml.load(txt, Loader=yaml.Loader)
    list_dataset = ya["dataset"]; probe = ya["probe_com"]
    if hasattr(list_dataset, "label_word_list") and hasattr(probe.model, "init"):
        probe.model.init(list_dataset.label_word_list)   # prompt 모델(bert/llama)만 해당; vision 은 skip
    ntk_path = f"./freeshap_res/ntk/{dataset}/{model_name}_seed{seed}_num{NUM}_val{val}_signFalse.pkl"
    bundle = pickle.load(open(ntk_path, "rb"))
    sidx = np.array(bundle["sampled_idx"])
    probe.get_cached_ntk(bundle["ntk"])
    train_set = list_dataset.get_idx_dataset(sidx, split="train")
    probe.get_train_labels(train_set)
    return probe


def _K_sym_from_ntk(ntk):
    """정규화된 NTK 에서 train kernel 의 대칭화 (2000x2000, float64) — 파이프라인 클래스 내부와 동일."""
    ntk0 = (ntk[0] if ntk.ndim == 3 else ntk).detach().to("cpu", dtype=torch.float64)
    n_train = ntk0.shape[1]
    K = ntk0[:n_train, :].numpy()
    return 0.5 * (K + K.T)

def kernel_relerr(K_sym, phi_tr):
    """||K - Phi Phi^T||_F / ||K||_F  (커널 근사 자체의 상대오차; λ 무관, eps 만 관여)."""
    P = np.asarray(phi_tr.detach().cpu(), dtype=np.float64)
    return float(np.linalg.norm(K_sym - P @ P.T) / np.linalg.norm(K_sym))


def predict_inv(probe, all_idx):
    ntk = probe.ntk
    idx = np.asarray(all_idx, dtype=int)
    k_train = ntk[:, idx[:, None], idx]
    y_train = probe.train_labels[idx]
    kr = shapleyNTKRegression(k_train, y_train, probe.num_labels, None, reg=float(INVLAM))
    k_test = ntk[:, ntk.size(2):, :][:, :, idx]
    preds = kr(k_test)
    if isinstance(preds, tuple):
        preds = preds[0]
    return np.asarray(preds.detach().cpu(), dtype=np.float32)


def predict_eigen(probe, all_idx, rank_actual, lam, eps, seed):
    probe.approximate("eigen"); probe.nystrom_use_pinv = False; probe.nystrom_use_lev = False
    probe.set_eigen_params(rank=rank_actual, lam=lam, solver="cholesky",
                           dtype="float32", seed=seed, floor=eps)
    probe.prepare_eigen_regression()
    return np.asarray(probe.eigen_regression(all_idx).detach().cpu(), dtype=np.float32)


def predict_pinv(probe, all_idx, d_actual, lam, eps, seed):
    probe.approximate("nystrom"); probe.nystrom_use_pinv = True; probe.nystrom_use_lev = False
    probe.set_nystrom_params(d=d_actual, lam=lam, solver="cholesky", dtype="float32",
                             landmark_seed=seed, jitter=eps)
    probe.prepare_nystrom_regression()
    return np.asarray(probe.nystrom_regression(all_idx).detach().cpu(), dtype=np.float32)


def predict_lev(probe, all_idx, d_actual, lam, eps, seed):
    probe.approximate("nystrom"); probe.nystrom_use_pinv = False; probe.nystrom_use_lev = True
    probe.set_nystrom_params(d=d_actual, lam=lam, solver="cholesky", dtype="float32",
                             landmark_seed=seed, jitter=eps)
    probe.prepare_nystrom_regression()
    return np.asarray(probe.nystrom_regression(all_idx).detach().cpu(), dtype=np.float32)


def _cells_for(args):
    """(rank, method, lam, eps) 조합 목록 (inv 는 rank 무관이라 별도 처리)."""
    return [(r, m, l, e) for r in args.ranks
            for m in args.methods
            for l in args.lams for e in args.epss]


def backfill_kernel(args):
    """기존 npz 에 kernel_relerr 필드만 추가 (NTK 캐시만 필요; 데이터셋/모델 로드 없음)."""
    for seed in args.seeds:
        g = glob.glob(f"./freeshap_res/ntk/{args.dataset}/{args.model}_seed{seed}_num{NUM}_val*_signFalse.pkl")
        if not g:
            print(f"[skip] seed{seed}: NTK 없음"); continue
        val = int(re.search(r"_val(\d+)_", os.path.basename(g[0])).group(1))
        todo = []
        for (r, mth, l, e) in _cells_for(args):
            p = kp_npz_path(make_a(args.dataset, val, model=args.model, rank=r), mth, l, e, seed)
            if os.path.exists(p) and (args.overwrite or "kernel_relerr" not in np.load(p).files):
                todo.append((r, mth, l, e, p))
        ip = kp_npz_path(make_a(args.dataset, val, model=args.model), "inv", None, None, seed)
        if os.path.exists(ip) and "kernel_relerr" not in np.load(ip).files:
            d = dict(np.load(ip)); d["kernel_relerr"] = 0.0
            np.savez_compressed(ip, **d)
        if not todo:
            print(f"[skip] seed{seed}: backfill 불필요"); continue
        print(f"[backfill] {args.dataset} seed{seed}: {len(todo)}칸 (NTK 만 로드)")
        ntk = pickle.load(open(g[0], "rb"))["ntk"]
        if abs(ntk.mean()) > 1000:   # get_cached_ntk 와 동일 정규화
            ntk = ntk / 10000
        K_sym = _K_sym_from_ntk(ntk)
        y_dummy = torch.zeros(K_sym.shape[0], dtype=torch.long)   # phi 구성에 라벨 미사용
        for (r, mth, l, e, p) in todo:
            d_act = int(NUM * r / 100)
            if mth == "eigen":
                reg = EigenNTKRegression(ntk, y_dummy, 2, rank=d_act, lam=l,
                                         solver="cholesky", dtype=torch.float32, device="cpu",
                                         eigen_decom_mode="top", seed=seed, floor=e)
            elif mth == "nystrom_lev":
                reg = NystromLevNTKRegression(ntk, y_dummy, 2, rank=d_act, lam=l,
                                              solver="cholesky", dtype=torch.float32, device="cpu",
                                              landmark_seed=seed, jitter=e)
            else:
                reg = NystromPinvNTKRegression(ntk, y_dummy, 2, rank=d_act, lam=l,
                                               solver="cholesky", dtype=torch.float32, device="cpu",
                                               landmark_seed=seed, jitter=e)
            kerr = kernel_relerr(K_sym, reg.phi_tr)
            d = dict(np.load(p)); d["kernel_relerr"] = kerr
            np.savez_compressed(p, **d)
            print(f"  [ok] r{r:g} {mth:12s} lam={l} eps={e} kernel_relerr={kerr:.4f}")
    print("[done backfill]", args.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--ranks", type=float, nargs="+", default=[RANK_PCT],
                    help="rank/d 퍼센트 목록 (기본 20). rank sweep 시 여러 개 (예: 1 5 10 15 20 25 30)")
    ap.add_argument("--lams", type=float, nargs="+", default=GRID_LAMS)
    ap.add_argument("--epss", type=float, nargs="+", default=GRID_EPS)
    ap.add_argument("--methods", type=str, nargs="+", default=["eigen", "nystrom_pinv"],
                    help="approx method 목록 (eigen/nystrom_pinv/nystrom_lev)")
    ap.add_argument("--model", type=str, default="bert",
                    help="모델 이름 (파일명/NTK 캐시 prefix): bert / llama / resnet")
    ap.add_argument("--config", type=str, default=None,
                    help="YAML config 이름 (기본: model 에 따라 ntk_prompt/ntk_llama/ntk_vision 자동)")
    ap.add_argument("--overwrite", action="store_true", help="기존 npz 있어도 다시 계산")
    ap.add_argument("--backfill_kernel", action="store_true",
                    help="기존 npz 에 kernel_relerr 만 추가 (예측 재계산 없음, 네트워크 불필요)")
    ap.add_argument("--res_root", type=str, default=ROOTS["res_root"])
    ap.add_argument("--pinv_root", type=str, default=ROOTS["pinv_root"])
    ap.add_argument("--inv_root", type=str, default=ROOTS["inv_root"])
    ap.add_argument("--lev_root", type=str, default=ROOTS["lev_root"])
    args = ap.parse_args()
    ROOTS.update(res_root=args.res_root, pinv_root=args.pinv_root, inv_root=args.inv_root,
                 lev_root=args.lev_root)
    if args.backfill_kernel:
        backfill_kernel(args); return

    for seed in args.seeds:
        g = glob.glob(f"./freeshap_res/ntk/{args.dataset}/{args.model}_seed{seed}_num{NUM}_val*_signFalse.pkl")
        if not g:
            print(f"[skip] seed{seed}: NTK 캐시 없음"); continue
        val = int(re.search(r"_val(\d+)_", os.path.basename(g[0])).group(1))

        todo = []
        if args.overwrite or not os.path.exists(
                kp_npz_path(make_a(args.dataset, val, model=args.model), "inv", None, None, seed)):
            todo.append((args.ranks[0], "inv", None, None))
        for (r, mth, l, e) in _cells_for(args):
            if args.overwrite or not os.path.exists(
                    kp_npz_path(make_a(args.dataset, val, model=args.model, rank=r), mth, l, e, seed)):
                todo.append((r, mth, l, e))
        if not todo:
            print(f"[skip] seed{seed}: 전부 존재"); continue

        print(f"[setup] {args.dataset} seed{seed} val{val}  (todo {len(todo)}칸) — probe/NTK 로드...")
        probe = setup_probe(args.dataset, seed, val, model_name=args.model, config=args.config)
        n_train = probe.ntk.size(2)
        all_idx = np.arange(n_train, dtype=int)
        K_sym = _K_sym_from_ntk(probe.ntk)   # 커널 근사 오차용 (train kernel)

        for (r, mth, l, e) in todo:
            a = make_a(args.dataset, val, model=args.model, rank=r)
            out = kp_npz_path(a, mth, l, e, seed)
            d_act = int(NUM * r / 100)
            try:
                if mth == "inv":
                    logits = predict_inv(probe, all_idx); kerr = 0.0
                elif mth == "eigen":
                    logits = predict_eigen(probe, all_idx, d_act, l, e, seed)
                    kerr = kernel_relerr(K_sym, probe.eigen_regression.phi_tr)
                elif mth == "nystrom_lev":
                    logits = predict_lev(probe, all_idx, d_act, l, e, seed)
                    kerr = kernel_relerr(K_sym, probe.nystrom_regression.phi_tr)
                else:
                    logits = predict_pinv(probe, all_idx, d_act, l, e, seed)
                    kerr = kernel_relerr(K_sym, probe.nystrom_regression.phi_tr)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                np.savez_compressed(out, logits=logits, method=mth, seed=seed,
                                    lam=(l if l is not None else -1.0),
                                    eps=(e if e is not None else -1.0),
                                    kernel_relerr=kerr)
                print(f"  [ok] r{r:g} {mth:12s} lam={l} eps={e} -> logits{logits.shape} kerr={kerr:.4f}")
            except Exception as ex:
                print(f"  [FAIL] r{r:g} {mth} lam={l} eps={e}: {type(ex).__name__}: {ex}")
        del probe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("[done]", args.dataset)


if __name__ == "__main__":
    main()

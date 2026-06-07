"""task_imbalance_shapley.py — Shapley for controlled-imbalance experiment.

Single script that runs both LRFShap (top-r by lambda) and A1 (top-r by
supervised score) over an imbalanced training NTK.

Usage (from vinfo/):
    python ./freeshap_res/claude_research/experiments/task_imbalance_shapley.py \
        --seed 2026 --dataset_name sst2 --num_train_dp 2000 \
        --val_sample_num 872 --pos_ratio 0.7 \
        --approximate eigen --eigen_rank 10 \
        --method a1 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import os
import sys
import time
import random
import argparse

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml

# ----- monkey-patch (only applied when --method a1) -----
import entks.ntk_regression as _ntk_reg

_ORIG_PRECOMPUTE = _ntk_reg.EigenNTKRegression._precompute_eigen_features


def _label_aware_precompute(self, ntk_full):
    """A1: top-r by s_i = (lam_i / (lam_i + rho))^2 * || u_i^T Y_centered ||^2."""
    if getattr(self, "eigen_decom_mode", "top") != "top":
        return _ORIG_PRECOMPUTE(self, ntk_full)

    print("[A1 label-aware] Computing eigen decomposition with supervised selection ...")
    t0 = time.time()

    if ntk_full.ndim == 3:
        ntk0 = ntk_full[0].detach()
    else:
        ntk0 = ntk_full.detach()
    ntk0 = ntk0.to("cpu", dtype=torch.float64)
    n_total, n_train = ntk0.shape

    K_trtr = ntk0[:n_train, :].numpy()
    K_tetr = ntk0[n_train:, :].numpy()
    K_sym = 0.5 * (K_trtr + K_trtr.T)

    d = int(min(self.rank, n_train))
    evals, evecs = np.linalg.eigh(K_sym)

    y_int = self.y.detach().cpu().numpy() if torch.is_tensor(self.y) else np.asarray(self.y)
    y_int = y_int.astype(np.int64)
    if y_int.ndim != 1:
        raise ValueError(f"Expected 1-D integer labels, got shape {y_int.shape}")
    n = y_int.shape[0]
    if n != n_train:
        raise ValueError(f"len(y) {n} != n_train {n_train}")

    C = int(getattr(self, "n_class", None) or (int(y_int.max()) + 1))
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y_int] = 1.0
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    coeffs = evecs.T @ Y_centered
    c2 = (coeffs ** 2).sum(axis=1)

    evals_pos = np.maximum(evals, 0.0)
    rho_filter = float(self.lam)
    filt = (evals_pos / (evals_pos + rho_filter)) ** 2
    score = filt * c2

    idx = np.argsort(score)[::-1][:d]

    lam_kept = evals[idx]
    U_kept = evecs[:, idx]
    lam_kept = np.clip(lam_kept, 1e-8, None)

    Phi_tr = U_kept * np.sqrt(lam_kept)[None, :]
    Phi_te = K_tetr @ (U_kept / np.sqrt(lam_kept)[None, :])

    decomp_time = time.time() - t0
    self.eigen_decomposition_time = decomp_time
    print(
        f"[A1 label-aware] Done (n_train={n_train}, d={d}) in {decomp_time:.4f}s, "
        f"top-1 score={score[idx[0]]:.3e}, bottom-of-top score={score[idx[-1]]:.3e}"
    )

    phi_tr = torch.from_numpy(Phi_tr).to(device=self.device, dtype=self.dtype)
    phi_te = torch.from_numpy(Phi_te).to(device=self.device, dtype=self.dtype)
    return phi_tr, phi_te


sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *           # noqa: E402,F401,F403
from probe import *             # noqa: E402,F401,F403
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: E402,F401

IMBALANCE_ROOT = "./freeshap_res/claude_research/data_selection_test/imbalance"


def parse_ratios(pos_ratio, class_ratios_str):
    if class_ratios_str is not None:
        ratios = [float(x) for x in class_ratios_str.split(",")]
    elif pos_ratio is not None:
        ratios = [1.0 - pos_ratio, pos_ratio]
    else:
        raise ValueError("Need --pos_ratio or --class_ratios.")
    if abs(sum(ratios) - 1.0) > 1e-3:
        raise ValueError(f"class_ratios must sum to 1, got {sum(ratios)}: {ratios}")
    return ratios


def ratio_tag(class_ratios):
    if len(class_ratios) == 2:
        return f"pos{int(round(class_ratios[1] * 100)):02d}"
    return "cls" + "_".join(f"{int(round(r * 100)):02d}" for r in class_ratios)


def log_gpu_info():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"[gpu] CUDA_VISIBLE_DEVICES={cuda_visible}")
    if not torch.cuda.is_available():
        print("[gpu] CUDA is not available. Running on CPU.")
        return
    device_count = torch.cuda.device_count()
    print(f"[gpu] device_count={device_count}")
    for gpu_idx in range(device_count):
        props = torch.cuda.get_device_properties(gpu_idx)
        print(f"[gpu] cuda:{gpu_idx} {props.name}, "
              f"{props.total_memory / (1024**3):.2f} GB")


def get_gpu_info():
    gi = {}
    if not torch.cuda.is_available():
        gi['device_type'] = 'CPU'
        return gi
    idx = torch.cuda.current_device()
    gi['used_gpu_count'] = 1
    gi['gpu_index'] = f"cuda:{idx}"
    gi['gpu_model'] = torch.cuda.get_device_name(idx)
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv:
        gi['cuda_visible_devices'] = cv
    return gi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["sst2", "mr", "qqp", "mnli", "ag_news", "mrpc", "rte"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_train_dp", type=int, default=2000)
    parser.add_argument("--val_sample_num", type=int, required=True)
    parser.add_argument("--pos_ratio", type=float, default=None)
    parser.add_argument("--class_ratios", type=str, default=None)
    parser.add_argument("--val_balance", action="store_true",
                        help="Must match the value used for task_imbalance_ntk.py.")
    parser.add_argument("--val_pos_ratio", type=float, default=None,
                        help="Must match the value used for task_imbalance_ntk.py "
                             "(binary-only imbalanced val).")
    parser.add_argument("--val_class_ratios", type=str, default=None,
                        help="Must match the value used for task_imbalance_ntk.py "
                             "(multi-class imbalanced val).")
    parser.add_argument("--method", type=str, required=True,
                        choices=["lrfshap", "a1"])
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="eigen",
                        choices=["inv", "eigen", "nystrom"])
    parser.add_argument("--eigen_rank", type=float, default=10)
    parser.add_argument("--inv_lambda_", type=float, default=1e-6)
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2)
    # Nystrom-specific
    parser.add_argument("--nystrom_d", type=int, default=512,
                        help="Number of random landmarks for Nystrom approximation")
    parser.add_argument("--landmark_seed", type=int, default=1234,
                        help="RNG seed for landmark sampling")
    parser.add_argument("--nystrom_lambda_", type=float, default=1e-2,
                        help="Ridge regularization on Nystrom features (analogue of eigen_lambda_)")
    parser.add_argument("--config", type=str, default="ntk_prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    log_gpu_info()

    if args.method == "a1":
        _ntk_reg.EigenNTKRegression._precompute_eigen_features = _label_aware_precompute
        print("[imbalance] method=a1 -> monkey-patch active.")
    else:
        print("[imbalance] method=lrfshap -> top-r by lambda (no patch).")

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    class_ratios = parse_ratios(args.pos_ratio, args.class_ratios)
    tmc_iter = args.tmc_iter
    approximate = args.approximate
    eigen_rank_pct = args.eigen_rank
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_

    timing_info = {'gpu_info': get_gpu_info()}

    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% -> actual rank={eigen_rank}")

    prompt = True
    signgd = False
    eigen_solver = "cholesky"
    eigen_dtype = "float32"
    per_point = True
    early_stopping = "True"

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"
    rtag = ratio_tag(class_ratios)
    base_path = f"{IMBALANCE_ROOT}/shapley/{dataset_name}/{rtag}/{args.method}"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    probe_model = yaml_args["probe_com"]
    dshap_com = yaml_args["dshap_com"]
    dshap_com.log_early_stopping = True

    if prompt:
        probe_model.model.init(list_dataset.label_word_list)

    probe_model.approximate(approximate)
    if approximate == "eigen":
        probe_model.set_eigen_params(
            rank=eigen_rank,
            lam=eigen_lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed,
        )
    elif approximate == "nystrom":
        probe_model.set_nystrom_params(
            d=int(args.nystrom_d),
            lam=float(args.nystrom_lambda_),
            solver=eigen_solver,
            dtype=eigen_dtype,
            landmark_seed=int(args.landmark_seed),
            jitter=1e-8,
        )
    else:
        probe_model.set_inv_params(lam=inv_lambda_)

    if signgd:
        probe_model.signgd()

    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    else:
        model_name = 'model'

    # Imbalance NTK cache (shared across lrfshap/a1 methods).
    if args.val_pos_ratio is not None:
        val_tag = f"valimb{val_sample_num}_pos{int(round(args.val_pos_ratio * 100)):02d}"
    elif args.val_class_ratios is not None:
        parts = [int(round(float(r) * 100)) for r in args.val_class_ratios.split(",")]
        val_tag = f"valimb{val_sample_num}_cls" + "_".join(f"{p:02d}" for p in parts)
    elif args.val_balance:
        val_tag = f"valbal{val_sample_num}"
    else:
        val_tag = f"val{val_sample_num}"
    ntk_path = (
        f"./freeshap_res/claude_research/imbalance_ntk/{dataset_name}/{rtag}/"
        f"{model_name}_seed{seed}_num{num_train_dp}_{val_tag}"
        f"_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")

    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)
    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError(f"Bad NTK cache format at {ntk_path}")

    ntk = bundle["ntk"]
    sampled_idx = np.array(bundle["sampled_idx"])
    sampled_val_idx = np.array(bundle["sampled_val_idx"])
    print(f"[info] |train|={len(sampled_idx)} |val|={len(sampled_val_idx)}")

    probe_model.get_cached_ntk(ntk)
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    if approximate == "eigen":
        print("[info] Preparing eigen regression features ...")
        t_eig = time.time()
        probe_model.prepare_eigen_regression()
        eig_total = time.time() - t_eig
        if hasattr(probe_model, 'eigen_regression') and probe_model.eigen_regression is not None:
            ed_time = probe_model.eigen_regression.eigen_decomposition_time
            timing_info['eigendecomposition'] = ed_time
            timing_info['eigen_preparation_total'] = eig_total
    elif approximate == "nystrom":
        print("[info] Preparing Nystrom regression features ...")
        t_eig = time.time()
        probe_model.prepare_nystrom_regression()
        eig_total = time.time() - t_eig
        if hasattr(probe_model, 'nystrom_regression') and probe_model.nystrom_regression is not None:
            ed_time = probe_model.nystrom_regression.eigen_decomposition_time
            timing_info['nystrom_feature_time'] = ed_time
            timing_info['nystrom_preparation_total'] = eig_total

    method_dir = approximate
    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (
            f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_invlam{inv_lam_str}"
            f"_{eigen_solver}_{eigen_dtype}_{args.method}"
        )
    elif approximate == "nystrom":
        nys_lam_str = f"{float(args.nystrom_lambda_):.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (
            f"_nys{int(args.nystrom_d)}_landseed{int(args.landmark_seed)}"
            f"_nyslam{nys_lam_str}_invlam{inv_lam_str}"
            f"_{eigen_solver}_{eigen_dtype}_{args.method}"
        )
    else:
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}_{args.method}"

    shapley_path = (
        f"{base_path}/{method_dir}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_{val_tag}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")
    dshap_com._log_iter = 0

    try:
        with open(shapley_path, "rb") as f:
            result = pickle.load(f)
        print(f"[info] loaded Shapley from cache")
        dv_result = result["dv_result"]
    except Exception as e:
        print(f"[info] cache miss -> computing Shapley   (reason: {e})")
        t_s = time.time()
        dv_result = dshap_com.run(
            data_idx=sampled_idx.tolist(),
            val_data_idx=sampled_val_idx.tolist(),
            iteration=tmc_iter,
            use_cache_ntk=True,
            prompt=prompt,
            seed=seed,
            num_dp=num_train_dp,
            checkpoint=False,
            per_point=per_point,
            early_stopping=early_stopping,
        )
        shap_time = time.time() - t_s
        timing_info['shapley_computation'] = shap_time
        print(f"[TIMING] Shapley computation: {shap_time:.2f}s")

        es_records = None
        if hasattr(dshap_com, 'early_stopping_records') and dshap_com.early_stopping_records:
            es_records = dshap_com.early_stopping_records

        result = {
            "dv_result": dv_result,
            "sampled_idx": sampled_idx,
            "sampled_val_idx": sampled_val_idx,
            "args": vars(args),
            "timing_info": timing_info,
            "early_stopping_records": es_records,
            "selection_scheme": args.method,
        }
        os.makedirs(os.path.dirname(shapley_path), exist_ok=True)
        with open(shapley_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[info] saved Shapley to {shapley_path}")

    dv_result = np.array(dv_result)
    print("dv_result shape:", dv_result.shape)


if __name__ == "__main__":
    main()

"""task_label_shapley.py — A1 (label-aware) Shapley computation.

Mirror of `vinfo/task_shapley.py` with two differences:

1. Monkey-patches `entks.ntk_regression.EigenNTKRegression._precompute_eigen_features`
   so that the rank-r eigen selection uses

       s_i = (lambda_i / (lambda_i + rho))^2 * || u_i^T Y_centered ||^2

   instead of the original `top-r by lambda_i`. Here Y_centered is the
   one-hot-encoded centered training label matrix; rho is `self.lam`
   (i.e., `--eigen_lambda_` from the CLI). For multi-class, the Frobenius
   norm across classes is used. For `eigen_decom_mode == "random"`, the
   original behaviour is preserved.

2. Saves Shapley pickles under

       ./freeshap_res/claude_research/data_selection_test/shapley/{dataset}/{method}/

   (instead of `./freeshap_res/shapley/{dataset}/{method}/`), and appends
   "_label" to the `extra_tag` so the output filename is distinguishable
   from the baseline `task_shapley.py` output.

The script is intended to be invoked from the `vinfo/` directory just like
`task_shapley.py`:

    python ./freeshap_res/claude_research/experiments/task_label_shapley.py \
        --seed 2026 --dataset_name rte --num_train_dp 2490 \
        --val_sample_num 277 --approximate eigen --eigen_rank 10 \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --tmc_iter 500
"""

import pickle
import os
import sys
import time
import random
import argparse

# Ensure cwd (expected to be vinfo/) is on sys.path so the upstream modules
# `entks`, `dataset`, `probe`, `dvutils` are importable. The script's own
# directory (`experiments/`) is sys.path[0] when invoked by file path, so
# without this insertion the `entks` import below would fail.
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml

# ----- monkey-patch BEFORE importing probe / Data_Shapley -----
# The patched function replaces top-r-by-lambda with top-r-by-supervised-score.

import entks.ntk_regression as _ntk_reg

_ORIG_PRECOMPUTE = _ntk_reg.EigenNTKRegression._precompute_eigen_features


def _label_aware_precompute(self, ntk_full):
    """A1 selection: top-r by  s_i = (lambda/(lambda+rho))^2 * || u_i^T Y_centered ||^2.

    Falls back to the original implementation when `eigen_decom_mode != "top"`.
    """
    # Preserve original behaviour for non-top modes (e.g. "random").
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

    # We need the FULL eigendecomposition to score every mode.
    evals, evecs = np.linalg.eigh(K_sym)
    # evals ascending by default.

    # Training labels as a 1-D integer vector.
    y_int = self.y.detach().cpu().numpy() if torch.is_tensor(self.y) else np.asarray(self.y)
    y_int = y_int.astype(np.int64)
    if y_int.ndim != 1:
        raise ValueError(f"Expected 1-D integer labels, got shape {y_int.shape}")
    n = y_int.shape[0]
    if n != n_train:
        raise ValueError(f"len(y) {n} != n_train {n_train}")

    C = int(getattr(self, "n_class", None) or (int(y_int.max()) + 1))

    # One-hot + column centering.
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y_int] = 1.0
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Mode-wise || u_i^T Y_centered ||_F^2 across classes.
    coeffs = evecs.T @ Y_centered                # (n, C)
    c2 = (coeffs ** 2).sum(axis=1)                # (n,)

    # Filter (use rho = self.lam, clip negative numerical eigvals).
    evals_pos = np.maximum(evals, 0.0)
    rho_filter = float(self.lam)
    filt = (evals_pos / (evals_pos + rho_filter)) ** 2

    score = filt * c2

    idx = np.argsort(score)[::-1][:d]
    method = f"a1_label(top-{d})"

    lam_kept = evals[idx]
    U_kept = evecs[:, idx]
    lam_kept = np.clip(lam_kept, 1e-8, None)

    # Feature matrices (same form as the original).
    Phi_tr = U_kept * np.sqrt(lam_kept)[None, :]
    Phi_te = K_tetr @ (U_kept / np.sqrt(lam_kept)[None, :])

    decomp_time = time.time() - t0
    self.eigen_decomposition_time = decomp_time
    print(
        f"[A1 label-aware] Done via {method} (n_train={n_train}, d={d}, "
        f"d/n_train(%)={d / n_train * 100:.2f}%) in {decomp_time:.4f}s"
    )
    print(f"[A1 label-aware] rho_filter={rho_filter:.3e}, "
          f"top-1 score = {score[idx[0]]:.3e}, bottom-of-top score = {score[idx[-1]]:.3e}")

    phi_tr = torch.from_numpy(Phi_tr).to(device=self.device, dtype=self.dtype)
    phi_te = torch.from_numpy(Phi_te).to(device=self.device, dtype=self.dtype)
    return phi_tr, phi_te


_ntk_reg.EigenNTKRegression._precompute_eigen_features = _label_aware_precompute
print("[A1 label-aware] EigenNTKRegression._precompute_eigen_features monkey-patched.")

# ----- rest of imports, matching task_shapley.py -----

from datasets import load_dataset  # noqa: E402,F401  (kept for parity)

# Ensure imports resolve when the script is invoked from vinfo/
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

from dataset import *           # noqa: E402,F401,F403
from probe import *             # noqa: E402,F401,F403
from dvutils.Data_Shapley import Fast_Data_Shapley  # noqa: E402,F401


# ----- output redirection -----
# Everything related to this experimental scheme is kept under claude_research/.
LABEL_OUTPUT_ROOT = "./freeshap_res/claude_research/data_selection_test"


# ===== GPU helpers (verbatim from task_shapley.py) =====
def log_gpu_info():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"[gpu] CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(f"[gpu] torch.cuda.is_available()={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[gpu] CUDA is not available. Running on CPU.")
        return
    device_count = torch.cuda.device_count()
    print(f"[gpu] torch.cuda.device_count()={device_count}")
    for gpu_idx in range(device_count):
        props = torch.cuda.get_device_properties(gpu_idx)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"[gpu] cuda:{gpu_idx} name={props.name}, total_memory={total_mem_gb:.2f} GB")
    current_idx = torch.cuda.current_device()
    print(f"[gpu] torch.cuda.current_device()={current_idx}")
    print(f"[gpu] current_device_name={torch.cuda.get_device_name(current_idx)}")


def get_gpu_info():
    gpu_info = {}
    if not torch.cuda.is_available():
        gpu_info['device_type'] = 'CPU'
        gpu_info['used_gpu_count'] = 0
        return gpu_info
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    current_idx = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_idx)
    gpu_info['used_gpu_count'] = 1
    gpu_info['gpu_index'] = f"cuda:{current_idx}"
    gpu_info['gpu_model'] = current_device_name
    if cuda_visible:
        gpu_info['cuda_visible_devices'] = cuda_visible
    return gpu_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="rte")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_train_dp", type=int, default=2490)
    parser.add_argument("--val_sample_num", type=int, default=277)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="eigen",
                        choices=["inv", "eigen", "none"])
    parser.add_argument("--eigen_rank", type=float, default=10,
                        help="Eigen rank as percentage of num_train_dp.")
    parser.add_argument("--inv_lambda_", type=float, default=1e-6)
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2)
    parser.add_argument("--config", type=str, default="ntk_prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    log_gpu_info()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num
    tmc_iter = args.tmc_iter

    approximate = args.approximate
    eigen_rank_pct = args.eigen_rank
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_

    timing_info = {}
    gpu_info = get_gpu_info()
    timing_info['gpu_info'] = gpu_info

    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% of num_dp={num_train_dp} -> actual rank={eigen_rank}")

    prompt = True
    signgd = False

    eigen_solver = "cholesky"
    eigen_dtype = "float32"

    per_point = True
    early_stopping = "True"

    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"
    # Redirected output root for the A1 experimental scheme.
    base_path = f"{LABEL_OUTPUT_ROOT}/shapley/{dataset_name}"

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

    if approximate != "none":
        probe_model.approximate(approximate)

    if approximate == "eigen":
        probe_model.set_eigen_params(
            rank=eigen_rank,
            lam=eigen_lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed,
        )
    elif approximate == "inv":
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

    # NTK cache lives in the original location and is shared.
    ntk_path = (
        f"./freeshap_res/ntk/{dataset_name}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")

    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)

    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError(
            "NTK cache format is old (ntk only). Regenerate with the updated script."
        )

    ntk = bundle["ntk"]
    sampled_idx = np.array(bundle["sampled_idx"])
    sampled_val_idx = np.array(bundle["sampled_val_idx"])
    print(f"[info] loaded NTK + indices: |train|={len(sampled_idx)}, |val|={len(sampled_val_idx)}")

    probe_model.get_cached_ntk(ntk)

    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    if approximate == "eigen":
        print("[info] Preparing eigen regression features (label-aware) ...")
        eigen_start_time = time.time()
        probe_model.prepare_eigen_regression()
        eigen_total_time = time.time() - eigen_start_time

        if hasattr(probe_model, 'eigen_regression') and probe_model.eigen_regression is not None:
            eigendecom_time = probe_model.eigen_regression.eigen_decomposition_time
            timing_info['eigendecomposition'] = eigendecom_time
            timing_info['eigen_preparation_total'] = eigen_total_time
            timing_info['eigen_preparation_overhead'] = eigen_total_time - eigendecom_time
            print(f"[TIMING] Eigendecomposition time: {eigendecom_time:.4f}s")
            print(f"[TIMING] Total eigen preparation time: {eigen_total_time:.4f}s")

    print("len(train_set) =", len(train_set))
    print("len(val_set)   =", len(val_set))

    method_dir = approximate  # 'inv' or 'eigen'

    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        # Append `_label` so the output filename is distinguishable from baseline.
        extra_tag = (
            f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_invlam{inv_lam_str}"
            f"_{eigen_solver}_{eigen_dtype}_label"
        )
    else:
        # `inv` mode is baseline FreeShap (full kernel inverse); no label-aware
        # selection applies, but we still tag the file so it sits under the
        # claude_research/ tree and does not collide with the upstream pkl.
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}_label"

    shapley_path = (
        f"{base_path}/{method_dir}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")

    dshap_com._log_iter = 0

    shapley_computation_time = None
    try:
        with open(shapley_path, "rb") as f:
            result = pickle.load(f)
        print(f"[info] loaded Shapley from {shapley_path}")
        dv_result = result["dv_result"]
        if "sampled_idx" in result:
            sampled_idx = np.array(result["sampled_idx"])
        if "sampled_val_idx" in result:
            sampled_val_idx = np.array(result["sampled_val_idx"])
        if "timing_info" in result:
            timing_info = result["timing_info"]
    except Exception as e:
        print(f"[info] shapley cache miss -> computing Shapley   (reason: {e})")
        shapley_start_time = time.time()
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
        shapley_computation_time = time.time() - shapley_start_time
        timing_info['shapley_computation'] = shapley_computation_time
        print(f"[TIMING] Shapley computation time: {shapley_computation_time:.4f}s")

        early_stopping_records = None
        if hasattr(dshap_com, 'early_stopping_records') and dshap_com.early_stopping_records:
            early_stopping_records = dshap_com.early_stopping_records

        result = {
            "dv_result": dv_result,
            "sampled_idx": sampled_idx,
            "sampled_val_idx": sampled_val_idx,
            "args": vars(args),
            "timing_info": timing_info,
            "early_stopping_records": early_stopping_records,
            "selection_scheme": "a1_label",
        }
        os.makedirs(os.path.dirname(shapley_path), exist_ok=True)
        with open(shapley_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[info] saved Shapley to {shapley_path}")

    dv_result = np.array(dv_result)
    print("dv_result shape:", dv_result.shape)

    if timing_info:
        print("\n" + "=" * 80)
        print("TIMING SUMMARY (A1 label-aware)")
        print("=" * 80)
        if 'gpu_info' in timing_info:
            print("\nGPU Information:")
            for k, v in timing_info['gpu_info'].items():
                print(f"{k:.<40} {v}")
        print("\nTiming Measurements:")
        for k, v in timing_info.items():
            if k != 'gpu_info':
                print(f"{k:.<40} {v:.4f}s")
        print("=" * 80)


if __name__ == "__main__":
    main()

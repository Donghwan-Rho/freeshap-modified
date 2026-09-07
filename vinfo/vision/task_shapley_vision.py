import pickle
import numpy as np
import yaml
import torch
import random
from datasets import load_dataset
import time

import sys, os
sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')

# vision 폴더에서 실행 → vinfo/를 sys.path에 추가해 probe/dataset 임포트 가능하게
_HERE  = os.path.dirname(os.path.abspath(__file__))
_VINFO = os.path.dirname(_HERE)
if _VINFO not in sys.path:
    sys.path.insert(0, _VINFO)

from dataset import *
from probe import *
from dvutils.Data_Shapley import Fast_Data_Shapley  # YAML tag 해석용

# vision 신규 클래스 (yaml_tag 등록)
from vision.vision_dataset import VisionReader, VisionDataset  # noqa: F401
from vision.vision_probe   import NTKVisionProbe                # noqa: F401

import argparse

from landmark_sv import resolve_landmark_sv, add_landmark_args   # SV 기반 Nystrom landmark (nystrom_q0..q4)


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
        print(
            f"[gpu] cuda:{gpu_idx} name={props.name}, total_memory={total_mem_gb:.2f} GB"
        )

    current_idx = torch.cuda.current_device()
    print(f"[gpu] torch.cuda.current_device()={current_idx}")
    print(f"[gpu] current_device_name={torch.cuda.get_device_name(current_idx)}")


def get_gpu_info():
    """Collect GPU information for timing records"""
    gpu_info = {}
    
    if not torch.cuda.is_available():
        gpu_info['device_type'] = 'CPU'
        gpu_info['used_gpu_count'] = 0
        return gpu_info
    
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    current_idx = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_idx)
    
    gpu_info['used_gpu_count'] = 1  # Single GPU usage (no DataParallel)
    gpu_info['gpu_index'] = f"cuda:{current_idx}"
    gpu_info['gpu_model'] = current_device_name
    if cuda_visible:
        gpu_info['cuda_visible_devices'] = cuda_visible
    
    return gpu_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="mr")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_train_dp", type=int, default=8530)
    parser.add_argument("--val_sample_num", type=int, default=1066)
    parser.add_argument("--tmc_iter", type=int, default=500)
    parser.add_argument("--approximate", type=str, default="inv",
                        choices=["inv", "eigen", "nystrom", "nystrom_pinv", "nystrom_lev",
                                 "nystrom_q4", "nystrom_q3", "nystrom_q2",
                                 "nystrom_q1", "nystrom_q0", "none"])
    parser.add_argument("--eigen_rank", type=float, default=30,
                        help="Eigen rank as percentage of num_train_dp (e.g., 10 means 10%% of data)")
    parser.add_argument("--inv_lambda_", type=float, default=1e-6,
                        help="Lambda (regularization parameter) for INV mode")
    parser.add_argument("--eigen_lambda_", type=float, default=1e-2,
                        help="Lambda (regularization parameter) for Eigen mode")
    parser.add_argument("--nystrom_d", type=float, default=30,
                        help="Nystrom landmark count as percentage of num_train_dp (e.g., 10 means 10%% of data), same convention as --eigen_rank")
    parser.add_argument("--nystrom_lambda_", type=float, default=1e-3,
                        help="Lambda (ridge regularization) for Nystrom mode (friend's default = 1e-3)")
    parser.add_argument("--eigeps", type=str, default="1e-8",
                        help="Eigen eigenvalue floor (filename tag: eigeps)")
    parser.add_argument("--nyseps", type=str, default="1e-8",
                        help="Nystrom jitter/eps (filename tag: nyseps)")
    parser.add_argument("--config", type=str, default="ntk_prompt",
                        help="YAML config name without .yaml extension (e.g., ntk_prompt, ntk_llama)")
    parser.add_argument("--out_root", type=str, default="./freeshap_res",
                        help="shapley/data_selection 출력 루트 (NTK 는 항상 ./freeshap_res/ntk 에서 읽는다).")
    add_landmark_args(parser)
    return parser.parse_args()


def _fmt_nyseps(eps):
    """1e+02 → '1e+2', 1e-08 → '1e-8' (text 판 파일명 포맷과 동일)."""
    return f"{eps:.0e}".replace('e+0', 'e+').replace('e-0', 'e-')


def _quantize_nyseps(eps):
    """파일명 표기와 실제 사용값을 일치시키기 위한 quantize (text 판과 동일)."""
    return float(f"{eps:.0e}")


def main():
    args = parse_args()
    log_gpu_info()

    dataset_name = args.dataset_name
    seed = args.seed
    num_train_dp = args.num_train_dp
    val_sample_num = args.val_sample_num

    tmc_iter = args.tmc_iter

    approximate = args.approximate
    # nystrom_pinv/nystrom_lev: alias to "nystrom" for all param/tag/path logic; only the
    # regression CLASS (probe_model.nystrom_use_pinv/lev) and method_dir differ. (text 판과 동일)
    _is_pinv = (approximate == "nystrom_pinv")
    _is_lev = (approximate == "nystrom_lev")   # leverage-score landmarks (pinv construction)
    # SV 기반 결정적 landmark (oracle diagnostic). 구성은 pinv 와 동일하고 landmark 선택만 다르다.
    _lm_mode = ("q4" if approximate == "nystrom_q4"
                else "q0" if approximate == "nystrom_q0"
                else "q2" if approximate == "nystrom_q2"
                else "q1" if approximate == "nystrom_q1"
                else "q3" if approximate == "nystrom_q3" else "uniform")
    _is_svlm = (_lm_mode != "uniform")
    if _is_pinv or _is_lev or _is_svlm:
        approximate = "nystrom"
    eigen_rank_pct = args.eigen_rank  # Now interpreted as percentage of num_dp
    inv_lambda_ = args.inv_lambda_
    eigen_lambda_ = args.eigen_lambda_

    # eps 처리 (파일명 태그와 실제 사용값 일치; text 판과 동일)
    eigen_eps = _quantize_nyseps(float(args.eigeps))
    eigeps_str = _fmt_nyseps(eigen_eps)
    nystrom_eps = _quantize_nyseps(float(args.nyseps))
    nyseps_str = _fmt_nyseps(nystrom_eps)

    # Initialize timing info dictionary
    timing_info = {}
    
    # Collect GPU information
    gpu_info = get_gpu_info()
    timing_info['gpu_info'] = gpu_info
    
    # Calculate actual eigen rank from percentage
    eigen_rank = int(num_train_dp * eigen_rank_pct / 100)
    print(f"[info] eigen_rank={eigen_rank_pct}% of num_dp={num_train_dp} -> actual rank={eigen_rank}")

    # Nystrom landmark count: interpret --nystrom_d as percentage of num_dp (same convention as eigen_rank)
    nystrom_d_pct = args.nystrom_d
    nystrom_d = int(num_train_dp * nystrom_d_pct / 100)
    if approximate == "nystrom":
        print(f"[info] nystrom_d={nystrom_d_pct}% of num_dp={num_train_dp} -> actual landmarks={nystrom_d}")

    prompt = False   # vision은 prompt 없음
    signgd = False

    # eigen 파라미터
    eigen_solver = "cholesky"  # or "lstsq"
    eigen_dtype = "float32"

    per_point = True
    early_stopping = "True"

    # Dynamic path construction based on dataset_name
    yaml_path = f"../configs/dshap/{dataset_name}/{args.config}.yaml"
    base_path = f"{args.out_root}/shapley/{dataset_name}"  # Shapley results (out_root override 가능)

    # seed 고정 (재현성 보장)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN 최적화 비활성화
    np.random.seed(seed)
    random.seed(seed)

    # ===== YAML 로드 =====
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    probe_model = yaml_args["probe_com"]
    dshap_com = yaml_args["dshap_com"]
    
    # Enable early stopping logging
    dshap_com.log_early_stopping = True

    if prompt:
        probe_model.model.init(list_dataset.label_word_list)

    if approximate != "none":
        probe_model.approximate(approximate)
    probe_model.nystrom_use_pinv = _is_pinv   # pinv -> pseudoinverse-Nystrom class
    probe_model.nystrom_use_lev = _is_lev    # lev -> leverage-score Nystrom class
    probe_model.nystrom_landmark_mode = _lm_mode   # q0..q4 -> SV 기반 결정적 landmark

    # # 정책: ntk_normalize는 모든 approximate에 적용
    # if hasattr(probe_model, "normalize_ntk"):
    #     probe_model.normalize_ntk()

    if approximate == "nystrom":
        probe_model.set_nystrom_params(
            d=nystrom_d,
            lam=float(args.nystrom_lambda_),
            solver=eigen_solver,
            dtype=eigen_dtype,
            landmark_seed=seed,
            jitter=nystrom_eps,
        )
    elif approximate == "eigen":
        probe_model.set_eigen_params(
            rank=eigen_rank,
            lam=eigen_lambda_,
            solver=eigen_solver,
            dtype=eigen_dtype,
            seed=seed,
            floor=eigen_eps
        )
    elif approximate == "inv":
        probe_model.set_inv_params(lam=inv_lambda_)

    if signgd:
        probe_model.signgd()

    # ===== model_name 결정 =====
    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    elif 'resnet' in probe_model.args['model']:
        model_name = 'resnet'
    elif 'resnext' in probe_model.args['model']:
        model_name = 'resnext'
    else:
        model_name = 'model'

    # ===== SV 기반 landmark: 같은 seed 의 inv Shapley 를 읽어 probe 에 주입 =====
    #   nyseps auto 는 랜덤 subset 으로 sigma_min(W) 를 추정하므로 실제 landmark 와 어긋난다.
    if _is_svlm:
        if str(args.nyseps).lower() == "auto":
            raise SystemExit("[nystrom_q0..q4] --nyseps auto 는 지원하지 않습니다. "
                             "--nyseps 1e-8 처럼 명시하세요.")
        probe_model.nystrom_landmark_sv = resolve_landmark_sv(
            args, dataset_name, model_name, seed, num_train_dp,
            val_sample_num, inv_lambda_, tmc_iter=tmc_iter)

    # ===== 1) NTK 캐시 로드 (indices도 같이 로드) =====
    ntk_path = (
        f"./freeshap_res/ntk/{dataset_name}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}_sign{signgd}.pkl"
    )
    print(f"[info] ntk_path = {ntk_path}")

    with open(ntk_path, "rb") as f:
        bundle = pickle.load(f)

    if not (isinstance(bundle, dict) and "ntk" in bundle):
        raise RuntimeError(
            "NTK cache format is old (ntk only). "
            "Please regenerate NTK with the updated script that stores indices."
        )

    ntk = bundle["ntk"]
    sampled_idx = np.array(bundle["sampled_idx"])
    sampled_val_idx = np.array(bundle["sampled_val_idx"])
    print(f"[info] loaded NTK + indices: |train|={len(sampled_idx)}, |val|={len(sampled_val_idx)}")

    # probe에 NTK 주입
    probe_model.get_cached_ntk(ntk)

    # ===== 2) train/val set 구성 (indices 재샘플링 금지) =====
    train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
    val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
    probe_model.get_train_labels(train_set)

    # eigen이면 feature 준비 (train labels 설정 후에 호출해야 함)
    if approximate == "eigen":
        print("[info] Preparing eigen regression features...")
        import time as time_module
        eigen_start_time = time_module.time()
        probe_model.prepare_eigen_regression()
        eigen_total_time = time_module.time() - eigen_start_time

        # Get eigendecomposition time from the regression model
        if hasattr(probe_model, 'eigen_regression') and probe_model.eigen_regression is not None:
            eigendecom_time = probe_model.eigen_regression.eigen_decomposition_time
            timing_info['eigendecomposition'] = eigendecom_time
            timing_info['eigen_preparation_total'] = eigen_total_time
            timing_info['eigen_preparation_overhead'] = eigen_total_time - eigendecom_time
            print(f"[TIMING] Eigendecomposition time: {eigendecom_time:.4f}s")
            print(f"[TIMING] Total eigen preparation time: {eigen_total_time:.4f}s")
            print(f"[TIMING] Overhead (non-decomposition): {eigen_total_time - eigendecom_time:.4f}s")
    elif approximate == "nystrom":
        print("[info] Preparing Nystrom regression features...")
        import time as time_module
        nys_start_time = time_module.time()
        probe_model.prepare_nystrom_regression()
        nys_total_time = time_module.time() - nys_start_time
        if hasattr(probe_model, 'nystrom_regression') and probe_model.nystrom_regression is not None:
            feat_time = probe_model.nystrom_regression.eigen_decomposition_time
            timing_info['nystrom_feature_time'] = feat_time
            timing_info['nystrom_preparation_total'] = nys_total_time
            print(f"[TIMING] Nystrom feature time: {feat_time:.4f}s")

    print("len(train_set) =", len(train_set))
    print("len(val_set)   =", len(val_set))

    # ===== 3) Shapley 결과 경로 =====
    method_dir = (f"nystrom_{_lm_mode}" if _is_svlm else
                  "nystrom_lev" if _is_lev else
                  ("nystrom_pinv" if _is_pinv else approximate))

    # Format lambda in scientific notation for filename (always use 1e-X format)
    if approximate == "eigen":
        eigen_lam_str = f"{eigen_lambda_:.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        # Use the percentage value itself (eigen_rank_pct) in filename
        extra_tag = f"_eig{eigen_rank_pct}_eiglam{eigen_lam_str}_eigeps{eigeps_str}_invlam{inv_lam_str}_{eigen_solver}_{eigen_dtype}"
    elif approximate == "nystrom":
        nys_lam_str = f"{float(args.nystrom_lambda_):.0e}"
        inv_lam_str = f"{inv_lambda_:.0e}"
        extra_tag = (
            f"_nys{nystrom_d_pct}"
            f"_nyslam{nys_lam_str}_nyseps{nyseps_str}_invlam{inv_lam_str}_{eigen_solver}_{eigen_dtype}"
        )
        if _is_pinv:
            extra_tag = extra_tag.replace("_nys", "_nyspinv", 1)  # filename marker: pseudoinverse variant
        elif _is_lev:
            extra_tag = extra_tag.replace("_nys", "_nyslev", 1)   # filename marker: leverage-score variant
        elif _is_svlm:
            extra_tag = extra_tag.replace("_nys", f"_nys{_lm_mode}", 1)
    else:
        lambda_str = f"{inv_lambda_:.0e}"
        extra_tag = f"_lam{lambda_str}"

    shapley_path = (
        f"{base_path}/{method_dir}/{model_name}"
        f"_seed{seed}_num{num_train_dp}_val{val_sample_num}"
        f"{extra_tag}_sign{signgd}_earlystop{early_stopping}"
        f"_tmc{tmc_iter}.pkl"
    )
    print(f"[info] shapley_path = {shapley_path}")
    
    # Setup early stopping logging
    dshap_com._log_iter = 0  # Counter for iteration number
    
    # ===== 4) Shapley 로드 or 계산 =====
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
        
        # Load timing info if available
        if "timing_info" in result:
            timing_info = result["timing_info"]
            print(f"[info] loaded timing info from cache: {timing_info}")

    except Exception as e:
        print("[info] shapley cache miss -> computing Shapley")
        print(f"[info] reason: {e}")
        
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
            early_stopping=early_stopping
        )
        
        shapley_computation_time = time.time() - shapley_start_time
        timing_info['shapley_computation'] = shapley_computation_time
        print(f"[TIMING] Shapley computation time: {shapley_computation_time:.4f}s")

        # Collect early stopping records from memory
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
        }
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(shapley_path), exist_ok=True)
        with open(shapley_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[info] saved Shapley to {shapley_path}")

    dv_result = np.array(dv_result)
    print("dv_result shape:", dv_result.shape)
    
    # ===== Print timing summary =====
    if timing_info:
        print("\n" + "="*80)
        print("TIMING SUMMARY")
        print("="*80)
        
        if 'gpu_info' in timing_info:
            gpu_info = timing_info['gpu_info']
            print("\nGPU Information:")
            print("-" * 40)
            for key, value in gpu_info.items():
                print(f"{key:.<40} {value}")
            print("-" * 40 + "\n")
        
        print("Timing Measurements:")
        print("-" * 40)
        for key, value in timing_info.items():
            if key != 'gpu_info':
                print(f"{key:.<40} {value:.4f}s")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()

"""Extract timing info (inv + 7 eigen ranks) from prediction .txt files.

For a given (dataset_name, seed, num_train_dp), reads the inv prediction
file and 7 eigen prediction files (ranks 1, 5, 10, 15, 20, 25, 30 % of
num_train_dp), checks that each was run on an NVIDIA RTX A6000, and prints:

    inv: <shapley_computation>
    <eigendecomp_r1>  <shapley_computation_r1>
    ...
    <eigendecomp_r30> <shapley_computation_r30>

Other parameters (val_sample_num, tmc_iter, inv_lambda_, eigen_lambda_,
signgd, early_stop) are fixed to the conventions used in the experiments;
val_sample_num is auto-detected by globbing.
"""
import argparse
import re
import sys
from pathlib import Path

BASE_DIR = Path("/extdata1/donghwan/freeshap/vinfo/freeshap_res/data_selection")
EXPECTED_GPU = "NVIDIA RTX A6000"
RANKS = [1, 5, 10, 15, 20, 25, 30]

# Fixed experimental conventions
TMC_ITER = 500
SIGN_STR = "signFalse"
EARLYSTOP_STR = "earlystopTrue"
# val_sample_num, inv_lambda_, eigen_lambda_ are auto-detected via glob.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_train_dp", type=int, required=True)
    return parser.parse_args()


def read_or_die(path: Path) -> str:
    if not path.exists():
        sys.exit(f"error: file not found: {path}")
    return path.read_text()


def extract_a6000_blocks(content: str, path: Path) -> list[dict]:
    """Find every 'timing info ...' block in the file and keep only the ones run
    on EXPECTED_GPU. Returns list of dicts with 'shapley' (float) and
    'eigendecomp' (float or None).

    Files often contain multiple appended runs (same setting on different GPUs
    or at different times); we filter to A6000 runs only.
    """
    blocks = re.findall(
        r"timing info \(from shapley pkl\):.*?(?=\n=+\n|\Z)",
        content,
        flags=re.DOTALL,
    )
    runs = []
    for blk in blocks:
        gpu_m = re.search(r"gpu_model:\s*(.+?)\s*(?:\n|$)", blk)
        if gpu_m is None or gpu_m.group(1).strip() != EXPECTED_GPU:
            continue
        shap_m = re.search(r"shapley_computation:\s*([\d.]+)s", blk)
        if shap_m is None:
            continue
        eig_m = re.search(r"eigendecomposition:\s*([\d.]+)s", blk)
        runs.append({
            "shapley": float(shap_m.group(1)),
            "eigendecomp": float(eig_m.group(1)) if eig_m else None,
        })
    if not runs:
        sys.exit(
            f"error: no '{EXPECTED_GPU}' run found in {path} "
            f"(file has {len(blocks)} timing block(s), all on a different GPU)"
        )
    return runs


def best_run(content: str, path: Path, need_eigendecomp: bool) -> dict:
    """Among A6000 runs in the file, pick the one with smallest shapley_computation."""
    runs = extract_a6000_blocks(content, path)
    if need_eigendecomp:
        runs = [r for r in runs if r["eigendecomp"] is not None]
        if not runs:
            sys.exit(f"error: no A6000 run with eigendecomposition timing in {path}")
    return min(runs, key=lambda r: r["shapley"])


def glob_unique(directory: Path, pattern: str) -> Path:
    """Find the single file in `directory` matching `pattern`. Exit on 0 or >1 matches."""
    if not directory.exists():
        sys.exit(f"error: directory not found: {directory}")
    matches = list(directory.glob(pattern))
    if not matches:
        sys.exit(f"error: no file matching {pattern} under {directory}")
    if len(matches) > 1:
        joined = "\n  ".join(str(m) for m in matches)
        sys.exit(f"error: multiple files match {pattern} under {directory}:\n  {joined}")
    return matches[0]


def inv_path(args) -> Path:
    pattern = (
        f"bert_seed{args.seed}"
        f"_num{args.num_train_dp}"
        f"_val*"
        f"_lam*"
        f"_{SIGN_STR}_{EARLYSTOP_STR}"
        f"_tmc{TMC_ITER}"
        f"_predictions.txt"
    )
    return glob_unique(BASE_DIR / args.dataset_name / "inv" / "predictions", pattern)


def eigen_path(args, rank: int) -> Path:
    """Try both eig{rank}.0 / eig{rank} forms and both old/new lambda layouts.

    Old layout: ..._eig{r}_eiglam1e-02_invlam*_cholesky_float32_...
    New layout: ..._eig{r}_lam1e-02_cholesky_float32_...
    Eigen lambda is always 1e-02 (constraint).
    """
    base = BASE_DIR / args.dataset_name / "eigen" / "predictions"
    if not base.exists():
        sys.exit(f"error: directory not found: {base}")
    head = (
        f"bert_seed{args.seed}"
        f"_num{args.num_train_dp}"
        f"_val*"
    )
    tail_common = f"_cholesky_float32_{SIGN_STR}_{EARLYSTOP_STR}_tmc{TMC_ITER}_predictions.txt"
    tried = []
    for eig_str in [f"eig{rank}.0", f"eig{rank}"]:
        for lam_part in ["_eiglam1e-02_invlam*", "_lam1e-02"]:
            pattern = f"{head}_{eig_str}{lam_part}{tail_common}"
            tried.append(pattern)
            matches = list(base.glob(pattern))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                joined = "\n  ".join(str(m) for m in matches)
                sys.exit(f"error: multiple files match {pattern}:\n  {joined}")
    joined = "\n  ".join(tried)
    sys.exit(
        f"error: eigen file not found for rank {rank} under {base}; tried:\n  {joined}"
    )


def main():
    args = parse_args()

    p = inv_path(args)
    content = read_or_die(p)
    run = best_run(content, p, need_eigendecomp=False)
    print(f"inv: {run['shapley']}")

    for r in RANKS:
        p = eigen_path(args, r)
        content = read_or_die(p)
        run = best_run(content, p, need_eigendecomp=True)
        print(f"{run['eigendecomp']} {run['shapley']}")


if __name__ == "__main__":
    main()

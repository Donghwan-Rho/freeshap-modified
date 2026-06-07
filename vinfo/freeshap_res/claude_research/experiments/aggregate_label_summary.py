"""aggregate_label_summary.py — single 8-line comparison txt for a (dataset, seed) run.

After `run_rte_label.sh` finishes, every per-rank `task_label_data_selection.py`
invocation has dropped a sidecar JSON under

    ./freeshap_res/claude_research/data_selection_test/data_selection/{ds}/
        {inv,eigen}/sidecar/{setting_name}.json

This script glob-reads those sidecars for one (dataset, seed, num_train_dp,
val_sample_num, tmc_iter, inv_lambda_, eigen_lambda_) tuple and writes a
single text file with eight Python-list assignments, in the format the user
asked for:

    inv_lam1e_6            = [...]
    r1_eigen_lam_inv1e_6   = [...]
    r5_eigen_lam_inv1e_6   = [...]
    r10_eigen_lam_inv1e_6  = [...]
    r15_eigen_lam_inv1e_6  = [...]
    r20_eigen_lam_inv1e_6  = [...]
    r25_eigen_lam_inv1e_6  = [...]
    r30_eigen_lam_inv1e_6  = [...]

Line 1 is the inv baseline (`top_results_inv` from the `approximate=inv` run).
Lines 2-8 are the dual-mode evaluations: eigen-A1 Shapley ranking, evaluated
with the INV (full-kernel) predictor — i.e. `top_results_inv` from each
`approximate=eigen --eigen_rank R` run. The 7 ranks {1, 5, 10, 15, 20, 25, 30}
are the defaults; configurable via `--ranks`.

Each list has length 100 (selection percentages 1% through 100%, in order).

Invoke from `vinfo/`:

    python ./freeshap_res/claude_research/experiments/aggregate_label_summary.py \
        --dataset_name rte --seed 2026 --num_train_dp 2490 --val_sample_num 277 \
        --tmc_iter 500 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 \
        --ranks 1 5 10 15 20 25 30
"""

import argparse
import ast
import json
import os
import re
import sys
import glob


LABEL_OUTPUT_ROOT = "./freeshap_res/claude_research/data_selection_test"
# Upstream baseline (FreeShap) outputs live here. The INV-mode baseline run by
# `task_shapley.py` + `task_data_selection.py` (no monkey-patch, no `_label`
# suffix) is functionally identical to a hypothetical A1 INV run, so we just
# reuse it instead of rerunning under claude_research/.
UPSTREAM_OUTPUT_ROOT = "./freeshap_res"


def fmt_lam(x: float) -> str:
    """Upstream pkl/predictions filename convention. `:.0e` -> '1e-06'."""
    return f"{x:.0e}"


def fmt_lam_short(x: float) -> str:
    """Strip the leading zero from the exponent: 1e-6 -> '1e-6' (matches the
    user's manual ipynb filename convention like `seed*_eiglam1e-2_iter500.ipynb`).
    """
    s = fmt_lam(x)  # '1e-06'
    if 'e' in s:
        mantissa, exp = s.split('e')
        sign = exp[0] if exp[0] in '+-' else '+'
        digits = exp[1:] if exp[0] in '+-' else exp
        digits = digits.lstrip('0') or '0'
        return f"{mantissa}e{sign}{digits}"
    return s


def fmt_lam_var(x: float) -> str:
    """Variable-name tag matching the user's ipynb convention: 1e-6 -> '1e_6'."""
    return fmt_lam_short(x).replace("-", "_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num_train_dp", type=int, required=True)
    p.add_argument("--val_sample_num", type=int, required=True)
    p.add_argument("--tmc_iter", type=int, default=500)
    p.add_argument("--inv_lambda_", type=float, default=1e-6)
    p.add_argument("--eigen_lambda_", type=float, default=1e-2)
    p.add_argument("--ranks", type=float, nargs="+",
                   default=[1, 5, 10, 15, 20, 25, 30],
                   help="Rank percentages to aggregate.")
    p.add_argument("--output_path", type=str, default=None,
                   help="If unset, defaults to "
                        "{LABEL_OUTPUT_ROOT}/data_selection/{ds}/summary_seed{S}_"
                        "n{N}_val{V}_eiglam{el}_invlam{il}_tmc{T}_label.txt")
    p.add_argument("--lrfshap_ipynb", type=str, default=None,
                   help="Path to the upstream user-maintained ipynb containing "
                        "the LRFShap (top-r by lambda) baseline lists for this "
                        "(dataset, num_dp). If unset, the script globs for "
                        "freeshap_res/data_selection/{ds}/seed{S}_eiglam*_iter{T}.ipynb*.")
    return p.parse_args()


def load_sidecar(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_ipynb_lrfshap(ipynb_path: str, num_dp: int):
    """Extract LRFShap (top-r by lambda) per-rank lists + inv baseline from
    the user-maintained ipynb. Looks for the code cell that contains
    `num_dp = {num_dp}` and parses every `r{N}_eigen_lam_inv*` assignment.

    Returns (inv_list_or_None, {rank_pct -> list_of_int}).
    """
    try:
        with open(ipynb_path, "r") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"[WARN] could not open ipynb {ipynb_path}: {e}", file=sys.stderr)
        return None, {}

    target_src = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if re.search(rf"num_dp\s*=\s*{num_dp}\b", src):
            target_src = src
            break

    if target_src is None:
        return None, {}

    inv_list = None
    inv_match = re.search(r"^\s*inv_lam\w+\s*=\s*(\[[^\]]*\])",
                           target_src, re.MULTILINE)
    if inv_match:
        try:
            inv_list = list(ast.literal_eval(inv_match.group(1)))
        except (ValueError, SyntaxError):
            pass

    rank_lists = {}
    # r{N}_eigen_lam_inv1e_6 = [...]   (LRFShap top-r by lambda + inv predict)
    for m in re.finditer(
        r"^\s*r(\d+)_eigen_lam_inv\w+\s*=\s*(\[[^\]]*\])",
        target_src, re.MULTILINE,
    ):
        rank = int(m.group(1))
        try:
            rank_lists[rank] = list(ast.literal_eval(m.group(2)))
        except (ValueError, SyntaxError):
            pass

    return inv_list, rank_lists


def parse_top_list_from_predictions(path: str):
    """Parse the LAST `top:\n[...]` block from an upstream predictions.txt.

    The upstream `task_data_selection.py` appends each run's results, so a file
    may contain multiple `top:` blocks; we use the most recent one. Returns a
    list of ints, or None if no list is found.
    """
    try:
        text = open(path, "r").read()
    except FileNotFoundError:
        return None

    # `[…]` on a single line of digits/commas/spaces is what Python's list repr
    # produces for an int list of moderate length. Tolerate optional whitespace.
    matches = re.findall(r"top:\s*\n(\[[^\n\]]*\])", text)
    if not matches:
        return None
    try:
        return list(ast.literal_eval(matches[-1]))
    except (ValueError, SyntaxError):
        return None


def main():
    args = parse_args()

    ds = args.dataset_name
    inv_lam_str = fmt_lam(args.inv_lambda_)     # e.g., "1e-06" or "1e-6"
    eigen_lam_str = fmt_lam(args.eigen_lambda_)
    inv_lam_var = fmt_lam_var(args.inv_lambda_) # e.g., "1e_6"

    ds_base = f"{LABEL_OUTPUT_ROOT}/data_selection/{ds}"

    # ----- find inv baseline -----
    # Two possible locations, in order of preference:
    #   (a) claude_research sidecar (if user happened to run task_label_*.py inv,
    #       e.g., for a clean self-contained record)
    #   (b) upstream predictions.txt produced by the standard
    #       task_shapley.py + task_data_selection.py inv pipeline.
    # Since INV mode never invokes the eigen monkey-patch, both are identical.
    inv_list = None
    inv_source = None

    # (a) claude_research sidecar.
    inv_glob_cr = (
        f"{ds_base}/inv/sidecar/*_seed{args.seed}_num{args.num_train_dp}"
        f"_val{args.val_sample_num}_lam*{inv_lam_str}*_label_signFalse_earlystopTrue"
        f"_tmc{args.tmc_iter}.json"
    )
    inv_matches_cr = sorted(glob.glob(inv_glob_cr))
    if inv_matches_cr:
        if len(inv_matches_cr) > 1:
            print(f"[WARN] multiple CR-inv sidecars; using {inv_matches_cr[0]}",
                  file=sys.stderr)
        inv_data = load_sidecar(inv_matches_cr[0])
        inv_list = inv_data["top_results_inv"]
        inv_source = inv_matches_cr[0]

    # (b) Fall back to upstream predictions.txt.
    if inv_list is None:
        upstream_inv_glob = (
            f"{UPSTREAM_OUTPUT_ROOT}/data_selection/{ds}/inv/predictions/"
            f"*_seed{args.seed}_num{args.num_train_dp}_val{args.val_sample_num}"
            f"_lam{inv_lam_str}_signFalse_earlystopTrue_tmc{args.tmc_iter}"
            f"_predictions.txt"
        )
        upstream_matches = sorted(glob.glob(upstream_inv_glob))
        if upstream_matches:
            if len(upstream_matches) > 1:
                print(f"[WARN] multiple upstream inv predictions.txt; "
                      f"using {upstream_matches[0]}", file=sys.stderr)
            parsed = parse_top_list_from_predictions(upstream_matches[0])
            if parsed is not None:
                inv_list = parsed
                inv_source = upstream_matches[0]
            else:
                print(f"[WARN] could not parse top: list from "
                      f"{upstream_matches[0]}", file=sys.stderr)
        else:
            print(f"[WARN] no upstream inv predictions.txt match (glob:\n  "
                  f"{upstream_inv_glob})", file=sys.stderr)

    if inv_list is None:
        print(f"[WARN] inv baseline NOT found", file=sys.stderr)
    else:
        print(f"[info] inv baseline: {inv_source}  (len {len(inv_list)})")

    # ----- find LRFShap (top-r by lambda) baseline lists from the upstream
    # ipynb maintained manually by the user. -----
    lrfshap_lists = {}
    lrfshap_ipynb_used = None
    eigen_lam_short = fmt_lam_short(args.eigen_lambda_)   # e.g. '1e-2'
    ipynb_candidates = []
    if args.lrfshap_ipynb:
        ipynb_candidates.append(args.lrfshap_ipynb)
    # Glob common locations (try short-exp first, then long-exp).
    ipynb_candidates.extend(sorted(glob.glob(
        f"{UPSTREAM_OUTPUT_ROOT}/data_selection/{ds}/seed{args.seed}"
        f"_eiglam{eigen_lam_short}_iter{args.tmc_iter}.ipynb*"
    )))
    ipynb_candidates.extend(sorted(glob.glob(
        f"{UPSTREAM_OUTPUT_ROOT}/data_selection/{ds}/seed{args.seed}"
        f"_eiglam{eigen_lam_str}_iter{args.tmc_iter}.ipynb*"
    )))
    # De-dup while preserving order.
    seen = set()
    ipynb_candidates = [p for p in ipynb_candidates if not (p in seen or seen.add(p))]

    for path in ipynb_candidates:
        _inv_ipynb, lrfshap = parse_ipynb_lrfshap(path, args.num_train_dp)
        if lrfshap:
            lrfshap_lists = lrfshap
            lrfshap_ipynb_used = path
            print(f"[info] LRFShap source: {path}  "
                  f"(ranks found: {sorted(lrfshap.keys())})")
            break

    if not lrfshap_lists:
        print(f"[WARN] LRFShap (top-r by lambda) results NOT found in ipynb.\n"
              f"  Tried: {ipynb_candidates}", file=sys.stderr)

    # ----- find each eigen-rank sidecar -----
    rank_lists = {}      # rank_pct -> list
    rank_paths = {}
    for r in args.ranks:
        # task_label_data_selection.py writes eigen_rank_pct as the literal CLI
        # value (float, e.g. 10 or 10.0). We try both representations.
        candidates = []
        for tag in (str(int(r)) if float(r).is_integer() else None, str(float(r))):
            if tag is None:
                continue
            eigen_extra_tag = (
                f"_eig{tag}_eiglam{eigen_lam_str}_invlam{inv_lam_str}"
                f"_cholesky_float32_label"
            )
            pattern = (
                f"{ds_base}/eigen/sidecar/*_seed{args.seed}_num{args.num_train_dp}"
                f"_val{args.val_sample_num}{eigen_extra_tag}"
                f"_signFalse_earlystopTrue_tmc{args.tmc_iter}.json"
            )
            candidates.extend(sorted(glob.glob(pattern)))
        candidates = sorted(set(candidates))
        if not candidates:
            print(f"[WARN] no eigen sidecar for rank={r}%", file=sys.stderr)
            rank_lists[r] = None
            continue
        if len(candidates) > 1:
            print(f"[WARN] multiple matches for rank={r}%; using {candidates[0]}",
                  file=sys.stderr)
        data = load_sidecar(candidates[0])
        rank_lists[r] = data["top_results_inv"]
        rank_paths[r] = candidates[0]
        print(f"[info] rank={r}%: {candidates[0]}  (len {len(rank_lists[r])})")

    # ----- write 8-line summary -----
    out_path = args.output_path or (
        f"{ds_base}/summary_seed{args.seed}_n{args.num_train_dp}"
        f"_val{args.val_sample_num}_eiglam{eigen_lam_str}_invlam{inv_lam_str}"
        f"_tmc{args.tmc_iter}_label.txt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Header explains what each line is.
    header = (
        f"# A1 label-aware vs LRFShap baseline — 3-way comparison summary\n"
        f"# dataset={ds}  seed={args.seed}  n_train={args.num_train_dp}  "
        f"val={args.val_sample_num}  tmc_iter={args.tmc_iter}\n"
        f"# inv_lambda_={args.inv_lambda_}  eigen_lambda_={args.eigen_lambda_}\n"
        f"# Each list is length 100, indexing selection-percentage 1%..100%.\n"
        f"# Values are int(acc * 10000) — divide by 10000 for [0,1] accuracy.\n"
        f"#\n"
        f"# Line 1 (inv_lam{inv_lam_var}):\n"
        f"#   FreeShap (full kernel inverse) Shapley + INV prediction.\n"
        f"# Lines 2-8 (rR_eigen_lam_inv{inv_lam_var}):\n"
        f"#   LRFShap top-r by lambda Shapley (rank R%) + INV prediction.\n"
        f"#   (Pulled from upstream user-maintained ipynb.)\n"
        f"# Lines 9-15 (rR_eigen_label_lam_inv{inv_lam_var}):\n"
        f"#   A1 label-aware top-r by s_i Shapley (rank R%) + INV prediction.\n"
        f"#   (Pulled from CR sidecar JSON written by task_label_data_selection.py.)\n"
        f"#\n"
        f"# All 15 lines share the same INV-prediction evaluation axis — only the\n"
        f"# Shapley ranking (and thus the selected subset at each top-k%) changes.\n\n"
    )

    lines = [header]

    # Line 1 — inv baseline.
    if inv_list is None:
        lines.append(f"inv_lam{inv_lam_var:<10} = None  # MISSING\n")
    else:
        lines.append(f"inv_lam{inv_lam_var:<10} = {list(inv_list)}\n")
    lines.append("\n")

    # Lines 2-8 — LRFShap (top-r by lambda).
    lines.append(f"# --- LRFShap (top-r by lambda) ---\n")
    for r in args.ranks:
        var_tag = int(r) if float(r).is_integer() else r
        var_name = f"r{var_tag}_eigen_lam_inv{inv_lam_var}"
        lst = lrfshap_lists.get(int(var_tag)) if isinstance(var_tag, int) else None
        if lst is None:
            lines.append(f"{var_name:<32} = None  # MISSING from ipynb\n")
        else:
            lines.append(f"{var_name:<32} = {list(lst)}\n")
    lines.append("\n")

    # Lines 9-15 — A1 label-aware.
    lines.append(f"# --- A1 label-aware (top-r by s_i = (lambda/(lambda+rho))^2 * |u^T y|^2) ---\n")
    for r in args.ranks:
        var_tag = int(r) if float(r).is_integer() else r
        var_name = f"r{var_tag}_eigen_label_lam_inv{inv_lam_var}"
        lst = rank_lists.get(r)
        if lst is None:
            lines.append(f"{var_name:<32} = None  # MISSING\n")
        else:
            lines.append(f"{var_name:<32} = {list(lst)}\n")

    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"\n[saved] {out_path}")
    print(f"        ({sum(1 for l in lines if l.strip() and not l.startswith('#'))} non-comment lines)")


if __name__ == "__main__":
    main()

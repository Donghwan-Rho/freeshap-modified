#!/bin/sh
# A1 label-aware Shapley + data selection for RTE (n=2490, val=277, seed=2026).
# Mirrors the upstream `n04_*.sh` convention but routes everything through
# the A1 task scripts under claude_research/experiments/.
#
# Invoke from vinfo/ (the directory that contains task_*.py and freeshap_res/):
#   cd /extdata1/donghwan/freeshap/vinfo
#   bash freeshap_res/claude_research/experiments/run_rte_label.sh
#
# Outputs land under:
#   freeshap_res/claude_research/data_selection_test/shapley/rte/
#   freeshap_res/claude_research/data_selection_test/data_selection/rte/

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
SHAPLEY=$SCRIPT_DIR/task_label_shapley.py
SELECT=$SCRIPT_DIR/task_label_data_selection.py

COMMON_ARGS="--seed 2026 --dataset_name rte --num_train_dp 2490 --val_sample_num 277 --tmc_iter 500"

# Note: the INV baseline (--approximate inv) is identical to the upstream
# FreeShap run -- no eigendecomposition happens, so the monkey-patch has no
# effect. Skip it here; the user is expected to have run the upstream
# task_shapley.py + task_data_selection.py with --approximate inv already
# (or to run it separately into the standard freeshap_res/ location).
# The aggregator below pulls the inv result from that upstream output.

# A1 label-aware Eigen, rank sweep matching the upstream pattern.
for R in 1 5 10 15 20 25 30; do
    python $SHAPLEY $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON_ARGS --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

# Aggregate the 8 sidecars into one comparison txt:
#   inv_lam1e_6           = [...]    (inv baseline FreeShap)
#   r1_eigen_lam_inv1e_6  = [...]    (eigen-A1 Shapley + INV prediction, rank 1%)
#   ...
#   r30_eigen_lam_inv1e_6 = [...]
python $SCRIPT_DIR/aggregate_label_summary.py \
    --dataset_name rte --seed 2026 --num_train_dp 2490 --val_sample_num 277 \
    --tmc_iter 500 --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 \
    --ranks 1 5 10 15 20 25 30

echo "[run_rte_label.sh] done"

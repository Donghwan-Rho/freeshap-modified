#!/bin/sh
# Fill the 3 missing lrfshap eigen sub-runs in mrpc valimb experiments:
#   mrpc/pos50/valimb300_pos70 : lrfshap r=30%
#   mrpc/pos50/valimb300_pos90 : lrfshap r=15% and r=30%
# These were missing because the original launcher exited early.

SCRIPT_DIR=freeshap_res/claude_research/experiments
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

run_one () {
    VPR=$1; R=$2
    COMMON="--seed 2026 --dataset_name mrpc --num_train_dp 2300 --val_sample_num 300 \
            --tmc_iter 500 --val_pos_ratio $VPR --pos_ratio 0.5 \
            --method lrfshap --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2"
    echo "==============================================================="
    echo "[mrpc-valimb-fill] val_pos=$VPR lrfshap eigen r=$R%"
    echo "==============================================================="
    python $SHAPLEY $COMMON
    python $SELECT  $COMMON
}

run_one 0.7 30    # mrpc valimb300_pos70 의 r=30
run_one 0.9 15    # mrpc valimb300_pos90 의 r=15
run_one 0.9 30    # mrpc valimb300_pos90 의 r=30

echo "[run_mrpc_valimb_partial_fill.sh] all done"

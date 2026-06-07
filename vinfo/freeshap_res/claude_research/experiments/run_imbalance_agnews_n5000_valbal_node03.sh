#!/bin/sh
# node03 — AG News n=5000, val=1000 (balanced 25/25/25/25 stratified), 3 ratios.
#   baseline:  0.25,0.25,0.25,0.25  (= 1250 per class)
#   mild:      0.55,0.15,0.15,0.15  (label0 majority 2750, others 750 each)
#   extreme:   0.85,0.05,0.05,0.05  (label0 majority 4250, others 250 each)
# val 강제 균등: each class 250 (1000//4).

# NOTE: `set -e` intentionally removed — the NTK step finishes successfully
# but exits with a multiprocessing semaphore-leak segfault on cleanup,
# which would otherwise abort the entire grid even though the NTK pkl is
# already written. Without set -e the script proceeds to the next step
# (Shapley/Selection) which then hits the cache and runs normally.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"

run_grid () {
    RATIOS=$1
    TAG=$2
    echo "==============================================================="
    echo "[agnews-n5000-valbal] $TAG (class_ratios=$RATIOS) — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --class_ratios $RATIOS

    # INV baseline once per ratio (LR == A1 in inv mode)
    echo "---------------------------------------------------------------"
    echo "[agnews-n5000-valbal] $TAG INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --class_ratios $RATIOS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[agnews-n5000-valbal] $TAG method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON --class_ratios $RATIOS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --class_ratios $RATIOS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

run_grid "0.25,0.25,0.25,0.25"   cls25_25_25_25
run_grid "0.55,0.15,0.15,0.15"   cls55_15_15_15
run_grid "0.85,0.05,0.05,0.05"   cls85_05_05_05

echo "[run_imbalance_agnews_n5000_valbal_node03.sh] all done"

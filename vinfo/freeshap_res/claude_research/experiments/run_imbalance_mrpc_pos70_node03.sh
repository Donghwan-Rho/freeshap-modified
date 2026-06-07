#!/bin/sh
# node03 — MRPC forced 70/30 (label 1 = natural majority at 70%).
# n=2000, val=408. pos_ratio = 0.7 (label-1 fraction).

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mrpc --num_train_dp 2000 --val_sample_num 408 --tmc_iter 500"

echo "==============================================================="
echo "[node03-mrpc] pos_ratio=0.7 (label1 maj 70%) — computing NTK ..."
echo "==============================================================="
python $NTK $COMMON --pos_ratio 0.7

for METHOD in lrfshap a1; do
    echo "---------------------------------------------------------------"
    echo "[node03-mrpc] pos70 method=$METHOD inv baseline"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio 0.7 --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio 0.7 --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6

    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[node03-mrpc] pos70 method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio 0.7 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio 0.7 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_mrpc_pos70_node03.sh] all done"

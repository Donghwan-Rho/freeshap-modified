#!/bin/sh
# node04 — QQP forced 70/30 (label 0 = natural majority at 70%).
# n=2000, val=1000. pos_ratio = 0.3 (label-1 fraction).

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name qqp --num_train_dp 2000 --val_sample_num 1000 --tmc_iter 500"

echo "==============================================================="
echo "[node04-qqp] pos_ratio=0.3 (label0 maj 70%) — computing NTK ..."
echo "==============================================================="
python $NTK $COMMON --pos_ratio 0.3

for METHOD in lrfshap a1; do
    echo "---------------------------------------------------------------"
    echo "[node04-qqp] pos30 method=$METHOD inv baseline"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio 0.3 --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio 0.3 --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6

    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[node04-qqp] pos30 method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio 0.3 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio 0.3 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_qqp_pos30_node04.sh] all done"

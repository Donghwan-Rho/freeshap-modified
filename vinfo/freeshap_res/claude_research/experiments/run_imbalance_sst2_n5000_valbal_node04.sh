#!/bin/sh
# node04 — SST-2 n=5000, val=856 (balanced 50/50 stratified), 3 train ratios.
# pos_ratio = label-1 (positive sentiment) fraction; natural majority.
#   50/50: 2500+2500, 70/30: 3500+1500, 90/10: 4500+500.
# val 강제 balanced: each class 428 (= minority pool of GLUE sst2 validation).
# INV baseline once per ratio (LR == A1 in inv mode).

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 856 --tmc_iter 500 --val_balance"

for POS in 0.5 0.7 0.9; do
    echo "==============================================================="
    echo "[sst2-n5000-valbal] pos_ratio=$POS — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --pos_ratio $POS

    # INV baseline once per ratio (LR == A1 in inv mode)
    echo "---------------------------------------------------------------"
    echo "[sst2-n5000-valbal] pos=$POS INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[sst2-n5000-valbal] pos=$POS method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --pos_ratio $POS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
done

echo "[run_imbalance_sst2_n5000_valbal_node04.sh] all done"

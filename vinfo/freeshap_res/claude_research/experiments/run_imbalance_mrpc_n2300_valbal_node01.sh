#!/bin/sh
# node01 — MRPC n=2300, val=258 (balanced 50/50), 3 train ratios.
# pos_ratio = label-1 fraction; label-1 = natural majority (equivalent paraphrase).
#   50/50: label1=1150 + label0=1150 (label0 pool 1194 의 96% 사용)
#   70/30: label1=1610 + label0=690
#   90/10: label1=2070 + label0=230
# val 강제 50/50: label0=129 + label1=129 (label0 minority pool 129 max).

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mrpc --num_train_dp 2300 --val_sample_num 258 --tmc_iter 500 --val_balance"

for POS in 0.5 0.7 0.9; do
    echo "==============================================================="
    echo "[mrpc-n2300-valbal] pos_ratio=$POS — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --pos_ratio $POS

    # INV baseline once per ratio (LR == A1 in inv mode)
    echo "---------------------------------------------------------------"
    echo "[mrpc-n2300-valbal] pos=$POS INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[mrpc-n2300-valbal] pos=$POS method=$METHOD eigen r=$R%"
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

echo "[run_imbalance_mrpc_n2300_valbal_node01.sh] all done"

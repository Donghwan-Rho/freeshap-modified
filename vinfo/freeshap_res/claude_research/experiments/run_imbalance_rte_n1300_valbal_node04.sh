#!/bin/sh
# node04 — RTE n=1300, val=262 (balanced 50/50 strictly), 3 ratios.
# pos_ratio = label-1 fraction; label1 forced to be the majority direction.
#   pos50: label1=650 + label0=650 (≈ natural balanced)
#   pos70: label1=910 + label0=390
#   pos90: label1=1170 + label0=130 (binding constraint — label0 pool is 1249)
# val 강제 50/50: each class 131 (val_pool: label0=146, label1=131 → cap=2*131=262).
#
# Per ratio: NTK -> INV baseline once (LR == A1 in INV mode) -> lrfshap eigen×7
#            -> a1 eigen×7  (a1 INV intentionally skipped; identical to lrfshap INV).

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name rte --num_train_dp 1300 --val_sample_num 262 --tmc_iter 500 --val_balance"

for POS in 0.5 0.7 0.9; do
    echo "==============================================================="
    echo "[rte-n1300-valbal] pos_ratio=$POS — NTK"
    echo "==============================================================="
    python $NTK $COMMON --pos_ratio $POS

    echo "---------------------------------------------------------------"
    echo "[rte-n1300-valbal] pos=$POS INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[rte-n1300-valbal] pos=$POS method=$METHOD eigen r=$R%"
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

echo "[run_imbalance_rte_n1300_valbal_node04.sh] all done"

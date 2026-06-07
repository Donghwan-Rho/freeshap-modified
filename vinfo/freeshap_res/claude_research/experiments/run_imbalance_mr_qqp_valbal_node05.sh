#!/bin/sh
# node05 — MR n=4500 (3 pos ratios) + QQP n=5000 (3 pos ratios), val=1000 balanced.
# Replaces the MNLI+QQP grid (MNLI INV-baseline was estimated at ~20h per ratio).
#
# MR pool: rotten_tomatoes train ~ 4265 per class -> max n=4500 with imbalance.
#   pos50  -> 2250/2250
#   pos70  -> 3150/1350  (label0 majority dir)
#   pos90  -> 4050/450   (label0 majority dir)
#   val_balance: per_class=500 (total 1000)
#
# QQP pool: n=5000 (same as before)
#   pos50/pos30/pos10
#
# NTK already cached for both MR (n=4500) and QQP (n=5000) ratios.
# Order: MR pos70 -> MR pos50 -> MR pos90 -> QQP pos50 -> pos30 -> pos10.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON_MR="--seed 2026 --dataset_name mr --num_train_dp 4500 --val_sample_num 1000 --tmc_iter 500 --val_balance"
COMMON_QQP="--seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"

run_grid_binary () {
    COMMON="$1"
    POS="$2"
    DS_LABEL="$3"
    echo "==============================================================="
    echo "[$DS_LABEL] pos_ratio=$POS — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --pos_ratio $POS

    echo "---------------------------------------------------------------"
    echo "[$DS_LABEL] pos=$POS INV baseline (method-independent)"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
        --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[$DS_LABEL] pos=$POS method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --pos_ratio $POS --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

# ===== MR n=4500 =====
run_grid_binary "$COMMON_MR"  0.7  mr-n4500-valbal
run_grid_binary "$COMMON_MR"  0.5  mr-n4500-valbal
run_grid_binary "$COMMON_MR"  0.9  mr-n4500-valbal

# ===== QQP n=5000 =====
run_grid_binary "$COMMON_QQP" 0.5  qqp-n5000-valbal
run_grid_binary "$COMMON_QQP" 0.3  qqp-n5000-valbal
run_grid_binary "$COMMON_QQP" 0.1  qqp-n5000-valbal

echo "[run_imbalance_mr_qqp_valbal_node05.sh] all done"

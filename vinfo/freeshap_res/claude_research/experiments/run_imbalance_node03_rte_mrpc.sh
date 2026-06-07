#!/bin/sh
# node03 — RTE (n=1300) + MRPC (n=2000) sequential, RTE first.
#
# RTE:
#   n_train = 1300 (reduced from 2490 because of minority pool ceiling at 90/10).
#   Majority = label_0 (entailment) — natural full-pool majority (50.2% > 49.8%).
#   ratios: 70/30  -> label_0 910 + label_1 390 (pos_ratio = 0.3)
#           90/10  -> label_0 1170 + label_1 130 (pos_ratio = 0.1)
#   val = 277 (full GLUE rte validation).
#
# MRPC:
#   n_train = 2000.
#   Majority = label_1 (equivalent) — natural majority 67.9%.
#   ratios: 50/50  -> label_1 1000 + label_0 1000 (pos_ratio = 0.5)
#           90/10  -> label_1 1800 + label_0 200 (pos_ratio = 0.9)
#   val = 408 (full GLUE mrpc validation).

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

run_one_grid() {
    DS=$1
    POS_RATIO=$2   # binary, label-1 fraction
    N_TRAIN=$3
    VAL=$4
    COMMON="--seed 2026 --dataset_name $DS --num_train_dp $N_TRAIN --val_sample_num $VAL --tmc_iter 500"

    echo "==============================================================="
    echo "[node03] $DS n=$N_TRAIN pos_ratio=$POS_RATIO — computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON --pos_ratio $POS_RATIO

    for METHOD in lrfshap a1; do
        echo "---------------------------------------------------------------"
        echo "[node03] $DS pos_ratio=$POS_RATIO method=$METHOD inv baseline"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio $POS_RATIO --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6
        python $SELECT  $COMMON --pos_ratio $POS_RATIO --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6

        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[node03] $DS pos_ratio=$POS_RATIO method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON --pos_ratio $POS_RATIO --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON --pos_ratio $POS_RATIO --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

# RTE first — n=1300, val=277
run_one_grid rte 0.3 1300 277   # 70/30 with label_0 majority
run_one_grid rte 0.1 1300 277   # 90/10 with label_0 majority

# MRPC — n=2000, val=408
run_one_grid mrpc 0.5 2000 408   # 50/50
run_one_grid mrpc 0.9 2000 408   # 90/10 with label_1 majority

echo "[run_imbalance_node03_rte_mrpc.sh] all done"

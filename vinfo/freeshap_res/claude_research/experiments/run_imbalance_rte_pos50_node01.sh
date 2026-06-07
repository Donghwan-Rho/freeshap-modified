#!/bin/sh
# node01 — RTE forced 50/50 balance.  Same n/val as pos30 + pos10.
# n=1300, val=277.  pos_ratio = 0.5  (label-1 fraction).
# Same majority convention as pos30/pos10: label 0 = entailment (natural majority).

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name rte --num_train_dp 1300 --val_sample_num 277 --tmc_iter 500"

echo "==============================================================="
echo "[node01-rte] pos_ratio=0.5 (forced balance, n=1300) — computing NTK ..."
echo "==============================================================="
python $NTK $COMMON --pos_ratio 0.5

for METHOD in lrfshap a1; do
    echo "---------------------------------------------------------------"
    echo "[node01-rte] pos50 method=$METHOD inv baseline"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --pos_ratio 0.5 --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6
    python $SELECT  $COMMON --pos_ratio 0.5 --method $METHOD \
        --approximate inv --inv_lambda_ 1e-6

    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[node01-rte] pos50 method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio 0.5 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio 0.5 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_rte_pos50_node01.sh] all done"

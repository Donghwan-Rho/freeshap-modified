#!/bin/sh
# node01 — QQP pos30 (= label0 majority 70/30) n=5000 valbal grid.
# Offloaded from node05 mr_qqp queue (still on MR pos50 INV; QQP not reached
# for many hours).  When node05 later reaches QQP pos30, it will cache-hit
# the sidecar this script writes — no double work.
#
# Per ratio: NTK (cache hit expected) -> INV baseline once (LR == A1 in INV)
#            -> lrfshap eigen x7 -> a1 eigen x7  (a1 INV intentionally skipped).

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"
POS=0.3

echo "==============================================================="
echo "[qqp-n5000-valbal] pos_ratio=$POS — NTK (cache hit expected)"
echo "==============================================================="
python $NTK $COMMON --pos_ratio $POS

echo "---------------------------------------------------------------"
echo "[qqp-n5000-valbal] pos=$POS INV baseline (method-independent)"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[qqp-n5000-valbal] pos=$POS method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_qqp_pos30_n5000_valbal_node01.sh] all done"

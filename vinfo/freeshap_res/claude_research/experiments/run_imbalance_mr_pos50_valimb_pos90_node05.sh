#!/bin/sh
# node05 — MR train 50/50 (n=4500) + val IMBALANCED (n=500, label1 90%).
# Controlled label-shift symmetry experiment: complements the original
# valbal sweep (train imbalanced + val balanced) with the opposite direction
# (train balanced + val extremely imbalanced).
#
# Per ratio: NTK (new — cache miss) -> INV baseline (LR == A1 in INV mode)
#            -> lrfshap eigen × 7 -> a1 eigen × 7.  Total 15 sub-runs.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mr --num_train_dp 4500 --val_sample_num 500 --tmc_iter 500 --val_pos_ratio 0.9"
POS=0.5

echo "==============================================================="
echo "[mr-n4500-valimb] train pos=$POS, val pos=0.9 — NTK"
echo "==============================================================="
python $NTK $COMMON --pos_ratio $POS

echo "---------------------------------------------------------------"
echo "[mr-n4500-valimb] train pos=$POS val pos=0.9 INV baseline (method-independent)"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[mr-n4500-valimb] train pos=$POS val pos=0.9 method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_mr_pos50_valimb_pos90_node05.sh] all done"

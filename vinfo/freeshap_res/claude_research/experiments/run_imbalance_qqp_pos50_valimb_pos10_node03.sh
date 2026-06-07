#!/bin/sh
# node03 — QQP train 50/50 (n=5000) + val IMBALANCED (n=1000, label0 90% =
# val_pos_ratio 0.1 since pos_ratio = label1 frac and QQP's natural majority
# is label0).  val=1000 to match qqp/pos50 valbal control (fair comparison).
#
# Per ratio: NTK (new — cache miss) -> INV baseline (LR == A1 in INV mode)
#            -> lrfshap eigen × 7 -> a1 eigen × 7.  Total 15 sub-runs.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_pos_ratio 0.1"
POS=0.5

echo "==============================================================="
echo "[qqp-n5000-valimb] train pos=$POS, val pos=0.1 (label0 maj 90%) — NTK"
echo "==============================================================="
python $NTK $COMMON --pos_ratio $POS

echo "---------------------------------------------------------------"
echo "[qqp-n5000-valimb] train pos=$POS val pos=0.1 INV baseline"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[qqp-n5000-valimb] train pos=$POS val pos=0.1 method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_qqp_pos50_valimb_pos10_node03.sh] all done"

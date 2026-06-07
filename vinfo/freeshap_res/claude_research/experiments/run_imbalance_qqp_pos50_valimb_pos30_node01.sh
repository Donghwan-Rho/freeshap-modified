#!/bin/sh
# node01 — QQP train 50/50 (n=5000) + val IMBALANCED (n=1000, label0 70% =
# val_pos_ratio 0.3 since pos_ratio = label1 frac, QQP natural majority = label0).

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_pos_ratio 0.3"
POS=0.5

echo "===============================================================" ; echo "[qqp-n5000-valimb] train pos=$POS val pos=0.3 (label0 maj 70%) — NTK" ; echo "==============================================================="
python $NTK $COMMON --pos_ratio $POS

echo "[qqp-n5000-valimb] train pos=$POS val pos=0.3 INV baseline"
python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --pos_ratio $POS --method lrfshap --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "[qqp-n5000-valimb] train pos=$POS val pos=0.3 method=$METHOD eigen r=$R%"
        python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio $POS --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_qqp_pos50_valimb_pos30_node01.sh] all done"

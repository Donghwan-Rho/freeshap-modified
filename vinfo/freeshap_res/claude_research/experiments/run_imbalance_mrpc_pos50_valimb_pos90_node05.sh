#!/bin/sh
# node05 — MRPC train 50/50 (n=2300) + val IMBALANCED (n=300, label1 90%).

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mrpc --num_train_dp 2300 --val_sample_num 300 --tmc_iter 500 --val_pos_ratio 0.9"
POS=0.5

echo "===============================================================" ; echo "[mrpc-n2300-valimb] train pos=$POS val pos=0.9 — NTK" ; echo "==============================================================="
python $NTK $COMMON --pos_ratio $POS

echo "[mrpc-n2300-valimb] train pos=$POS val pos=0.9 INV baseline"
python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --pos_ratio $POS --method lrfshap --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "[mrpc-n2300-valimb] train pos=$POS val pos=0.9 method=$METHOD eigen r=$R%"
        python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio $POS --method $METHOD --approximate eigen --eigen_rank $R --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_mrpc_pos50_valimb_pos90_node05.sh] all done"

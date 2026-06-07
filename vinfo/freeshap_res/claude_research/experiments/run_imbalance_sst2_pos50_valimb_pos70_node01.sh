#!/bin/sh
# node01 — SST-2 train 50/50 (n=5000) + val IMBALANCED (n=400, label1 70%).
# Same label-shift symmetry experiment as MR, on SST-2.
#
# Per ratio: NTK (new — cache miss) -> INV baseline (LR == A1 in INV mode)
#            -> lrfshap eigen × 7 -> a1 eigen × 7.  Total 15 sub-runs.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 400 --tmc_iter 500 --val_pos_ratio 0.7"
POS=0.5

echo "==============================================================="
echo "[sst2-n5000-valimb] train pos=$POS, val pos=0.7 — NTK"
echo "==============================================================="
python $NTK $COMMON --pos_ratio $POS

echo "---------------------------------------------------------------"
echo "[sst2-n5000-valimb] train pos=$POS val pos=0.7 INV baseline"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --pos_ratio $POS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[sst2-n5000-valimb] train pos=$POS val pos=0.7 method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio $POS --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_sst2_pos50_valimb_pos70_node01.sh] all done"

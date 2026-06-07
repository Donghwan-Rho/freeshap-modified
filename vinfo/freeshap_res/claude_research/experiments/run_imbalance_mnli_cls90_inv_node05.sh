#!/bin/sh
# node05 — MNLI cls90_05_05 (label0 90/5/5) n=5000 valbal=1000.
# LRFShap INV (full kernel) ONLY — same INV baseline used for all other
# datasets (mathematically eigen rank=100%; LR == A1 in INV mode).
# Heavy step (~20h).  node01 handles lrfshap eigen × 7; node04 handles a1
# eigen × 7.  NTK already cached.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"
RATIOS="0.9,0.05,0.05"

echo "==============================================================="
echo "[mnli-n5000-valbal] cls90_05_05 — NTK (cache hit expected)"
echo "==============================================================="
python $NTK $COMMON --class_ratios $RATIOS

echo "---------------------------------------------------------------"
echo "[mnli-n5000-valbal] cls90_05_05 method=lrfshap INV (full rank)"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --class_ratios $RATIOS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --class_ratios $RATIOS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6

echo "[run_imbalance_mnli_cls90_inv_node05.sh] all done"

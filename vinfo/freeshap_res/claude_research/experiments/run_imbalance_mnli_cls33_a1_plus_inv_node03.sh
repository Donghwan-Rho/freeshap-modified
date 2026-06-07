#!/bin/sh
# node03 — MNLI cls33_33_33 (balanced) n=5000 valbal=1000.
# Per user request: a1 eigen × 7 (the core A1 cells) THEN lrfshap INV
# (the FreeShap baseline; LR == A1 in INV mode, same as other datasets).
# Total = 8 sub-runs.  NTK already cached.

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 --tmc_iter 500 --val_balance"
RATIOS="0.333,0.333,0.334"

echo "==============================================================="
echo "[mnli-n5000-valbal] cls33_33_33 — NTK (cache hit expected)"
echo "==============================================================="
python $NTK $COMMON --class_ratios $RATIOS

# 1) A1 eigen × 7 (core A1 cells; lightweight ~5-15 min each)
for R in 1 5 10 15 20 25 30; do
    echo "---------------------------------------------------------------"
    echo "[mnli-n5000-valbal] cls33_33_33 method=a1 eigen r=$R%"
    echo "---------------------------------------------------------------"
    python $SHAPLEY $COMMON --class_ratios $RATIOS --method a1 \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --class_ratios $RATIOS --method a1 \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

# 2) LRFShap INV (full kernel) — the FreeShap baseline (heavy, ~20h)
echo "---------------------------------------------------------------"
echo "[mnli-n5000-valbal] cls33_33_33 method=lrfshap INV (full rank)"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --class_ratios $RATIOS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --class_ratios $RATIOS --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6

echo "[run_imbalance_mnli_cls33_a1_plus_inv_node03.sh] all done"

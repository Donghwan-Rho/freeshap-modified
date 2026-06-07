#!/bin/sh
# node01 RESUME — MRPC n=2300 valbal.
# Skips redundant a1 INV (LR == A1 in INV mode). Picks up:
#   pos70: a1 eigen × 7 ranks (lrfshap done already).
#   pos90: lrfshap INV + lrfshap eigen×7 + a1 eigen×7  (no a1 INV).
# NTK is already cached for both ratios (n=2300 valbal258).

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON="--seed 2026 --dataset_name mrpc --num_train_dp 2300 --val_sample_num 258 --tmc_iter 500 --val_balance"

# ===== pos70 — finish a1 eigen only =====
for R in 1 5 10 15 20 25 30; do
    echo "==============================================================="
    echo "[mrpc-n2300-valbal] pos=0.7 method=a1 eigen r=$R% (resume)"
    echo "==============================================================="
    python $SHAPLEY $COMMON --pos_ratio 0.7 --method a1 \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    python $SELECT  $COMMON --pos_ratio 0.7 --method a1 \
        --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
done

# ===== pos90 — full grid (no a1 INV) =====
echo "==============================================================="
echo "[mrpc-n2300-valbal] pos_ratio=0.9 — NTK (cache hit expected)"
echo "==============================================================="
python $NTK $COMMON --pos_ratio 0.9

echo "---------------------------------------------------------------"
echo "[mrpc-n2300-valbal] pos=0.9 INV baseline (method-independent)"
echo "---------------------------------------------------------------"
python $SHAPLEY $COMMON --pos_ratio 0.9 --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6
python $SELECT  $COMMON --pos_ratio 0.9 --method lrfshap \
    --approximate inv --inv_lambda_ 1e-6

for METHOD in lrfshap a1; do
    for R in 1 5 10 15 20 25 30; do
        echo "---------------------------------------------------------------"
        echo "[mrpc-n2300-valbal] pos=0.9 method=$METHOD eigen r=$R%"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON --pos_ratio 0.9 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        python $SELECT  $COMMON --pos_ratio 0.9 --method $METHOD \
            --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
    done
done

echo "[run_imbalance_mrpc_n2300_valbal_node01_resume.sh] all done"

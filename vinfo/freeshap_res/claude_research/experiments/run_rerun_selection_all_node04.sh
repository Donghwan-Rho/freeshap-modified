#!/bin/sh
# node04 — Re-run SELECTION step only for all valbal + trainbal ratios
# to compute and store balanced accuracy alongside naive accuracy.
#
# Coverage (25 ratios total):
#   valbal  : 21 ratios (mrpc, sst2, mr, qqp, rte, ag_news, mnli)
#   trainbal: 4 ratios (mr×2, sst2×2)
#     - qqp/pos50/valbal duplicates the valbal entry — not repeated here
#     - qqp/pos50/valimb already excluded upstream
#
# Per ratio: 15 sub-runs
#   - 1 INV  (lrfshap, --approximate inv)
#   - 7 LR eigen (lrfshap, --approximate eigen, ranks 1/5/10/15/20/25/30)
#   - 7 A1 eigen (a1,      --approximate eigen, ranks 1/5/10/15/20/25/30)
# Total: 25 × 15 = 375 selection runs.
#
# Shapley pickles + NTK caches are already on disk — only the SELECT step is
# rerun.  task_imbalance_data_selection.py has been updated to compute
# balanced accuracy and append it to predictions.txt + sidecar JSON
# (backward-compatible: existing keys preserved).
#
# Usage:
#   cd /extdata1/donghwan/freeshap/vinfo
#   CUDA_VISIBLE_DEVICES=1 nohup bash freeshap_res/claude_research/experiments/run_rerun_selection_all_node04.sh \
#       > freeshap_res/claude_research/experiments/run_rerun_selection_all_node04.log 2>&1 &
#   disown

SCRIPT_DIR=freeshap_res/claude_research/experiments
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

RANKS="1 5 10 15 20 25 30"

run_block() {
    # $1 : tag (for log)
    # $@ : COMMON arguments (one big string already split)
    TAG="$1"; shift
    COMMON="$*"
    echo "==============================================================="
    echo "[$TAG] INV baseline (method-independent)"
    echo "==============================================================="
    python $SELECT $COMMON --method lrfshap --approximate inv --inv_lambda_ 1e-6

    for METHOD in lrfshap a1; do
        for R in $RANKS; do
            echo "---------------------------------------------------------------"
            echo "[$TAG] method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SELECT $COMMON --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
}

T0=$(date +%s)

# =============================================================================
# VALBAL (21 ratios)
# =============================================================================

# ---- mrpc pos50 / pos70 / pos90 ----
run_block "mrpc-n2300-valbal258 pos50" \
    --seed 2026 --dataset_name mrpc --num_train_dp 2300 --val_sample_num 258 \
    --tmc_iter 500 --val_balance --pos_ratio 0.5
run_block "mrpc-n2300-valbal258 pos70" \
    --seed 2026 --dataset_name mrpc --num_train_dp 2300 --val_sample_num 258 \
    --tmc_iter 500 --val_balance --pos_ratio 0.7
run_block "mrpc-n2300-valbal258 pos90" \
    --seed 2026 --dataset_name mrpc --num_train_dp 2300 --val_sample_num 258 \
    --tmc_iter 500 --val_balance --pos_ratio 0.9

# ---- sst2 pos50 / pos70 / pos90 ----
run_block "sst2-n5000-valbal856 pos50" \
    --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 856 \
    --tmc_iter 500 --val_balance --pos_ratio 0.5
run_block "sst2-n5000-valbal856 pos70" \
    --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 856 \
    --tmc_iter 500 --val_balance --pos_ratio 0.7
run_block "sst2-n5000-valbal856 pos90" \
    --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 856 \
    --tmc_iter 500 --val_balance --pos_ratio 0.9

# ---- mr pos50 / pos70 / pos90 ----
run_block "mr-n4500-valbal1000 pos50" \
    --seed 2026 --dataset_name mr --num_train_dp 4500 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --pos_ratio 0.5
run_block "mr-n4500-valbal1000 pos70" \
    --seed 2026 --dataset_name mr --num_train_dp 4500 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --pos_ratio 0.7
run_block "mr-n4500-valbal1000 pos90" \
    --seed 2026 --dataset_name mr --num_train_dp 4500 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --pos_ratio 0.9

# ---- qqp pos50 / pos30 / pos10 ----
run_block "qqp-n5000-valbal1000 pos50" \
    --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --pos_ratio 0.5
run_block "qqp-n5000-valbal1000 pos30" \
    --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --pos_ratio 0.3
run_block "qqp-n5000-valbal1000 pos10" \
    --seed 2026 --dataset_name qqp --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --pos_ratio 0.1

# ---- rte pos50 / pos70 / pos90 ----
run_block "rte-n1300-valbal262 pos50" \
    --seed 2026 --dataset_name rte --num_train_dp 1300 --val_sample_num 262 \
    --tmc_iter 500 --val_balance --pos_ratio 0.5
run_block "rte-n1300-valbal262 pos70" \
    --seed 2026 --dataset_name rte --num_train_dp 1300 --val_sample_num 262 \
    --tmc_iter 500 --val_balance --pos_ratio 0.7
run_block "rte-n1300-valbal262 pos90" \
    --seed 2026 --dataset_name rte --num_train_dp 1300 --val_sample_num 262 \
    --tmc_iter 500 --val_balance --pos_ratio 0.9

# ---- ag_news cls25/25/25/25, cls55/15/15/15, cls85/05/05/05 ----
run_block "agnews-n5000-valbal1000 cls25_25_25_25" \
    --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --class_ratios 0.25,0.25,0.25,0.25
run_block "agnews-n5000-valbal1000 cls55_15_15_15" \
    --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --class_ratios 0.55,0.15,0.15,0.15
run_block "agnews-n5000-valbal1000 cls85_05_05_05" \
    --seed 2026 --dataset_name ag_news --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --class_ratios 0.85,0.05,0.05,0.05

# ---- mnli cls33_33_33, cls60_20_20, cls90_05_05 ----
run_block "mnli-n5000-valbal1000 cls33_33_33" \
    --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --class_ratios 0.333,0.333,0.334
run_block "mnli-n5000-valbal1000 cls60_20_20" \
    --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --class_ratios 0.6,0.2,0.2
run_block "mnli-n5000-valbal1000 cls90_05_05" \
    --seed 2026 --dataset_name mnli --num_train_dp 5000 --val_sample_num 1000 \
    --tmc_iter 500 --val_balance --class_ratios 0.9,0.05,0.05

# =============================================================================
# TRAINBAL (4 ratios — train balanced, val imbalanced)
# =============================================================================

# ---- mr pos50 train, val pos=0.7 / 0.9 ----
run_block "mr-n4500-valimb500_pos70 trainpos50" \
    --seed 2026 --dataset_name mr --num_train_dp 4500 --val_sample_num 500 \
    --tmc_iter 500 --val_pos_ratio 0.7 --pos_ratio 0.5
run_block "mr-n4500-valimb500_pos90 trainpos50" \
    --seed 2026 --dataset_name mr --num_train_dp 4500 --val_sample_num 500 \
    --tmc_iter 500 --val_pos_ratio 0.9 --pos_ratio 0.5

# ---- sst2 pos50 train, val pos=0.7 / 0.9 ----
run_block "sst2-n5000-valimb400_pos70 trainpos50" \
    --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 400 \
    --tmc_iter 500 --val_pos_ratio 0.7 --pos_ratio 0.5
run_block "sst2-n5000-valimb400_pos90 trainpos50" \
    --seed 2026 --dataset_name sst2 --num_train_dp 5000 --val_sample_num 400 \
    --tmc_iter 500 --val_pos_ratio 0.9 --pos_ratio 0.5

T1=$(date +%s)
echo "[run_rerun_selection_all_node04.sh] all done in $((T1 - T0)) seconds"

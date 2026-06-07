#!/bin/sh
# Controlled imbalance experiment for SST-2 (iter 04 part 1).
#   train n=2000, seed=2026, val=872, single seed.
#   ratios: 0.7 (70/30) and 0.9 (90/10).  50/50 baseline is from upstream.
#   methods: lrfshap (top-r by lambda) + a1 (top-r by supervised).
#   approximate: inv + eigen rank in {1,5,10,15,20,25,30}%.
#
# Invoke from vinfo/:
#   cd /extdata1/donghwan/freeshap/vinfo
#   bash freeshap_res/claude_research/experiments/run_imbalance_sst2.sh

set -e

SCRIPT_DIR=freeshap_res/claude_research/experiments
NTK=$SCRIPT_DIR/task_imbalance_ntk.py
SHAPLEY=$SCRIPT_DIR/task_imbalance_shapley.py
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

COMMON_BASE="--seed 2026 --dataset_name sst2 --num_train_dp 2000 --val_sample_num 872 --tmc_iter 500"

for RATIO in 0.7 0.9; do
    echo "==============================================================="
    echo "[imbalance-sst2] pos_ratio=$RATIO   computing NTK ..."
    echo "==============================================================="
    python $NTK $COMMON_BASE --pos_ratio $RATIO

    for METHOD in lrfshap a1; do
        echo "---------------------------------------------------------------"
        echo "[imbalance-sst2] ratio=$RATIO method=$METHOD inv baseline"
        echo "---------------------------------------------------------------"
        python $SHAPLEY $COMMON_BASE --pos_ratio $RATIO --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6
        python $SELECT  $COMMON_BASE --pos_ratio $RATIO --method $METHOD \
            --approximate inv --inv_lambda_ 1e-6

        for R in 1 5 10 15 20 25 30; do
            echo "---------------------------------------------------------------"
            echo "[imbalance-sst2] ratio=$RATIO method=$METHOD eigen r=$R%"
            echo "---------------------------------------------------------------"
            python $SHAPLEY $COMMON_BASE --pos_ratio $RATIO --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
            python $SELECT  $COMMON_BASE --pos_ratio $RATIO --method $METHOD \
                --approximate eigen --eigen_rank $R \
                --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
        done
    done
done

echo "[run_imbalance_sst2.sh] all done"

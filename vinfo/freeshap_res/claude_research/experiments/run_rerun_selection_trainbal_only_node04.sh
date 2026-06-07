#!/bin/sh
# node04 — selection rerun ONLY for the 4 trainbal valimb ratios.
# Adds *_balanced sidecar keys (and predictions.txt lines).
# valbal ratios are skipped (naive==balanced when val is balanced — redundant).

SCRIPT_DIR=freeshap_res/claude_research/experiments
SELECT=$SCRIPT_DIR/task_imbalance_data_selection.py

run_one_inv () {
    DS=$1; POS=$2; N=$3; V=$4; VPR=$5
    python $SELECT --seed 2026 --dataset_name $DS --num_train_dp $N --val_sample_num $V \
        --tmc_iter 500 --val_pos_ratio $VPR --pos_ratio $POS \
        --method lrfshap --approximate inv --inv_lambda_ 1e-6
}
run_one_eig () {
    DS=$1; POS=$2; N=$3; V=$4; VPR=$5; METHOD=$6; R=$7
    python $SELECT --seed 2026 --dataset_name $DS --num_train_dp $N --val_sample_num $V \
        --tmc_iter 500 --val_pos_ratio $VPR --pos_ratio $POS \
        --method $METHOD --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2
}

run_block () {
    DS=$1; POS=$2; N=$3; V=$4; VPR=$5
    echo "==============================================================="
    echo "  RERUN: $DS pos=$POS val=$V val_pos=$VPR"
    echo "==============================================================="
    run_one_inv $DS $POS $N $V $VPR
    for METHOD in lrfshap a1; do
        for R in 1 5 10 15 20 25 30; do
            echo "--- $DS val_pos=$VPR $METHOD r=$R% ---"
            run_one_eig $DS $POS $N $V $VPR $METHOD $R
        done
    done
}

# 4 trainbal valimb ratios
run_block mr   0.5 4500 500  0.7
run_block mr   0.5 4500 500  0.9
run_block sst2 0.5 5000 400  0.7
run_block sst2 0.5 5000 400  0.9

echo "[run_rerun_selection_trainbal_only_node04.sh] all done"

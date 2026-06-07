#!/bin/bash
# launch_imbalance.sh — single-command launcher for the SST-2 / MR imbalance
# sweep.  Wraps nohup + background + PID-file so noteb shutdown is safe.
#
# Usage (from vinfo/):
#   bash freeshap_res/claude_research/experiments/launch_imbalance.sh <DS> [GPU]
#     DS  : sst2 | mr
#     GPU : 0, 1, ...   (default 0)

DS=${1:?"usage: launch_imbalance.sh <sst2|mr> [GPU]"}
GPU=${2:-0}
DIR=freeshap_res/claude_research/experiments

case "$DS" in
    sst2)   SCRIPT=$DIR/run_imbalance_sst2.sh ;;
    mr)     SCRIPT=$DIR/run_imbalance_mr.sh ;;
    qqp)    SCRIPT=$DIR/run_imbalance_qqp.sh ;;
    node04) SCRIPT=$DIR/run_imbalance_node04.sh ;;   # MNLI + AG News
    node03) SCRIPT=$DIR/run_imbalance_node03_rte_mrpc.sh ;;  # RTE + MRPC
    agnews_cls85) SCRIPT=$DIR/run_imbalance_ag_news_cls85_node01.sh ;;  # offloaded from node04
    qqp_pos30) SCRIPT=$DIR/run_imbalance_qqp_pos30_node04.sh ;;  # QQP 70/30
    mrpc_pos70) SCRIPT=$DIR/run_imbalance_mrpc_pos70_node03.sh ;;  # MRPC 70/30
    rte_pos50) SCRIPT=$DIR/run_imbalance_rte_pos50_node01.sh ;;  # RTE balanced
    mrpc_n2300_valbal) SCRIPT=$DIR/run_imbalance_mrpc_n2300_valbal_node01.sh ;;  # MRPC n=2300 val balanced
    mrpc_n2300_valbal_resume) SCRIPT=$DIR/run_imbalance_mrpc_n2300_valbal_node01_resume.sh ;;  # resume: pos70 a1 eigen + pos90 (no a1 INV)
    qqp_pos30_n5000_valbal) SCRIPT=$DIR/run_imbalance_qqp_pos30_n5000_valbal_node01.sh ;;  # node01 offload — QQP pos30
    qqp_pos10_n5000_valbal) SCRIPT=$DIR/run_imbalance_qqp_pos10_n5000_valbal_node04.sh ;;  # node04 offload — QQP pos10
    rte_n1300_valbal) SCRIPT=$DIR/run_imbalance_rte_n1300_valbal_node04.sh ;;  # node04 — RTE n=1300 val balanced 262 (3 ratios)
    mnli_cls33_lrfshap) SCRIPT=$DIR/run_imbalance_mnli_cls33_lrfshap_node01.sh ;;  # node01 — MNLI cls33 balanced, lrfshap only (INV + eigen × 7)
    mnli_cls33_lrfshap_eigen) SCRIPT=$DIR/run_imbalance_mnli_cls33_lrfshap_eigen_node01.sh ;;  # node01 — MNLI cls33 lrfshap eigen × 7 ONLY (no INV)
    mnli_cls33_a1) SCRIPT=$DIR/run_imbalance_mnli_cls33_a1_node03.sh ;;  # node03 — MNLI cls33 balanced, a1 eigen × 7 only
    mnli_cls33_a1_inv) SCRIPT=$DIR/run_imbalance_mnli_cls33_a1_inv_node03.sh ;;  # node03 — MNLI cls33 a1 INV (full rank) only
    mnli_cls33_a1_plus_inv) SCRIPT=$DIR/run_imbalance_mnli_cls33_a1_plus_inv_node03.sh ;;  # node03 — MNLI cls33 a1 eigen × 7 + lrfshap INV (8 sub-runs)
    mnli_cls90_lrfshap_eigen) SCRIPT=$DIR/run_imbalance_mnli_cls90_lrfshap_eigen_node01.sh ;;  # node01 — MNLI cls90 lrfshap eigen × 7
    mnli_cls90_a1_eigen) SCRIPT=$DIR/run_imbalance_mnli_cls90_a1_eigen_node04.sh ;;  # node04 — MNLI cls90 a1 eigen × 7
    mnli_cls90_inv) SCRIPT=$DIR/run_imbalance_mnli_cls90_inv_node05.sh ;;  # node05 — MNLI cls90 lrfshap INV (full kernel)
    mr_valimb_pos70) SCRIPT=$DIR/run_imbalance_mr_pos50_valimb_pos70_node04.sh ;;  # node04 — MR train 50/50 + val pos70 (label shift symmetry)
    mr_valimb_pos90) SCRIPT=$DIR/run_imbalance_mr_pos50_valimb_pos90_node05.sh ;;  # node05 — MR train 50/50 + val pos90
    sst2_valimb_pos70) SCRIPT=$DIR/run_imbalance_sst2_pos50_valimb_pos70_node01.sh ;;  # node01 — SST-2 train 50/50 + val pos70
    sst2_valimb_pos90) SCRIPT=$DIR/run_imbalance_sst2_pos50_valimb_pos90_node03.sh ;;  # node03 — SST-2 train 50/50 + val pos90
    qqp_valimb_pos10) SCRIPT=$DIR/run_imbalance_qqp_pos50_valimb_pos10_node03.sh ;;  # node03 — QQP train 50/50 + val pos10 (label0 maj 90%)
    mrpc_valimb_pos70) SCRIPT=$DIR/run_imbalance_mrpc_pos50_valimb_pos70_node04.sh ;;  # node04 — MRPC train 50/50 + val pos70 (label1 maj 70%)
    mrpc_valimb_pos90) SCRIPT=$DIR/run_imbalance_mrpc_pos50_valimb_pos90_node05.sh ;;  # node05 — MRPC train 50/50 + val pos90 (label1 maj 90%)
    qqp_valimb_pos30) SCRIPT=$DIR/run_imbalance_qqp_pos50_valimb_pos30_node01.sh ;;  # node01 — QQP train 50/50 + val pos30 (label0 maj 70%)
    agnews_n5000_valbal) SCRIPT=$DIR/run_imbalance_agnews_n5000_valbal_node03.sh ;;  # AG News n=5000 val balanced
    sst2_n5000_valbal) SCRIPT=$DIR/run_imbalance_sst2_n5000_valbal_node04.sh ;;  # SST-2 n=5000 val balanced
    mnli_qqp_n5000_valbal) SCRIPT=$DIR/run_imbalance_mnli_qqp_n5000_valbal_node05.sh ;;  # MNLI + QQP n=5000 val balanced
    mr_qqp_valbal) SCRIPT=$DIR/run_imbalance_mr_qqp_valbal_node05.sh ;;  # MR n=4500 + QQP n=5000 val balanced (replaces mnli)
    agnews_trainbal_valimb) SCRIPT=$DIR/run_imbalance_agnews_trainbal_valimb_node05.sh ;;  # node05 — AG News train cls25 + val cls55 / cls85
    mnli_trainbal_valimb_pos60) SCRIPT=$DIR/run_imbalance_mnli_trainbal_valimb_pos60_node04.sh ;;  # node04 — MNLI train cls33 + val cls60_20_20
    rte_trainbal_valimb) SCRIPT=$DIR/run_imbalance_rte_trainbal_valimb_node01.sh ;;  # node01 — RTE train pos50 + val pos70 / pos90
    mnli_trainbal_valimb_cls90_lrfshap_inv) SCRIPT=$DIR/run_imbalance_mnli_trainbal_valimb_cls90_lrfshap_inv_node03.sh ;;  # node03 — MNLI val cls90 NTK + INV + lrfshap eigen x7
    mnli_trainbal_valimb_cls90_a1) SCRIPT=$DIR/run_imbalance_mnli_trainbal_valimb_cls90_a1_node01.sh ;;  # node01 — MNLI val cls90 a1 eigen x7 (NTK by node03)
    mnli_trainbal_valimb_cls90_lrfshap_eigen) SCRIPT=$DIR/run_imbalance_mnli_trainbal_valimb_cls90_lrfshap_eigen_node05.sh ;;  # node05 — MNLI val cls90 lrfshap eigen x7 only (parallel to node03 INV)
    *) echo "unknown dataset: $DS"; exit 1 ;;
esac

LOG=$DIR/run_imbalance_${DS}.log
PID=$DIR/run_imbalance_${DS}.pid

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found.  pwd=$(pwd)"
    exit 1
fi

if pgrep -u "$(id -un)" -f "run_imbalance_${DS}" > /dev/null; then
    echo "WARNING: a previous imbalance-${DS} run is still alive."
    echo "  pkill -u \$USER -f run_imbalance_${DS}"
    echo "  pkill -u \$USER -f task_imbalance_"
    exit 1
fi

chmod +x "$SCRIPT" 2>/dev/null

CUDA_VISIBLE_DEVICES=$GPU PATH=/home/donghwan/.conda/envs/freeshap/bin:$PATH nohup bash "$SCRIPT" </dev/null > "$LOG" 2>&1 &
echo $! > "$PID"
disown

sleep 2
if ps -p "$(cat $PID)" > /dev/null; then
    echo "==================================================================="
    echo "  STARTED OK  (imbalance-${DS})"
    echo "  GPU       = $GPU"
    echo "  PID       = $(cat $PID)"
    echo "  log       = $LOG"
    echo "  pid file  = $PID"
    echo "==================================================================="
    echo "tail -f $LOG   # watch progress"
    echo "----- first log lines -----"
    head -20 "$LOG" 2>/dev/null
    echo "----- end -----"
else
    echo "ERROR: process died immediately. Log tail:"
    tail -50 "$LOG"
    exit 1
fi

#!/bin/bash
# launch_all.sh — single-command launcher for the combined AG News + SST-2
# A1 sweep. Wraps the nohup + background + PID-file dance into one script
# so the user only types one command.
#
# Usage (from vinfo/):
#   bash freeshap_res/claude_research/experiments/launch_all.sh         # GPU 0
#   bash freeshap_res/claude_research/experiments/launch_all.sh 1       # GPU 1

GPU=${1:-0}
DIR=freeshap_res/claude_research/experiments
SCRIPT=$DIR/run_agnews_sst2_label.sh
LOG=$DIR/run_agnews_sst2_label.log
PID=$DIR/run_agnews_sst2_label.pid

# Sanity: are we in vinfo/?
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found."
    echo "       Are you in /extdata1/donghwan/freeshap/vinfo ?"
    echo "       pwd = $(pwd)"
    exit 1
fi

# Sanity: any previous run alive?
if pgrep -u "$(id -un)" -f run_agnews_sst2_label > /dev/null; then
    echo "WARNING: a previous run is still alive."
    echo "  Kill it first:"
    echo "    pkill -u \$USER -f run_agnews_sst2_label"
    echo "    pkill -u \$USER -f task_label_shapley"
    echo "  Then retry."
    exit 1
fi

# Make sure all called scripts are executable.
chmod +x $DIR/run_agnews_sst2_label.sh $DIR/run_agnews_label.sh $DIR/run_sst2_label.sh 2>/dev/null

# Launch in background with nohup.
CUDA_VISIBLE_DEVICES=$GPU nohup bash "$SCRIPT" > "$LOG" 2>&1 &
echo $! > "$PID"

sleep 2

if ps -p "$(cat $PID)" > /dev/null; then
    echo "==================================================================="
    echo "  STARTED OK"
    echo "  GPU       = $GPU"
    echo "  PID       = $(cat $PID)"
    echo "  log       = $LOG"
    echo "  pid file  = $PID"
    echo "==================================================================="
    echo
    echo "Watch progress:"
    echo "  tail -f $LOG"
    echo
    echo "Check still running:"
    echo "  ps -p \$(cat $PID)"
    echo
    echo "Kill if needed:"
    echo "  kill \$(cat $PID)"
    echo
    echo "Detach tmux now:  Ctrl-b  then  d"
    echo
    echo "----- first log lines -----"
    head -20 "$LOG" 2>/dev/null
    echo "----- end -----"
else
    echo "ERROR: process died immediately. Log tail:"
    tail -50 "$LOG"
    exit 1
fi

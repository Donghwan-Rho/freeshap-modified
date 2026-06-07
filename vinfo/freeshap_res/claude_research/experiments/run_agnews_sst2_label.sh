#!/bin/sh
# Sequential wrapper: AG News -> SST-2 A1 sweep + aggregation.
# Each sub-script handles its own errors via `set -e`. We deliberately do
# NOT set -e here, so if AG News fails (e.g., one rank crashes), SST-2 still
# attempts to run.
#
# Invoke from vinfo/:
#   cd /extdata1/donghwan/freeshap/vinfo
#   bash freeshap_res/claude_research/experiments/run_agnews_sst2_label.sh

SCRIPT_DIR=freeshap_res/claude_research/experiments

t0=$(date +%s)
echo "================================================================"
echo "[$(date)] Combined AG News + SST-2 A1 run starting"
echo "================================================================"

echo
echo "[$(date)] >>> Phase 1 / 2 : AG News (n=2000, val=1000)"
bash $SCRIPT_DIR/run_agnews_label.sh
agnews_exit=$?
echo "[$(date)] <<< AG News finished  (exit code $agnews_exit)"

echo
echo "[$(date)] >>> Phase 2 / 2 : SST-2 (n=2000, val=872)"
bash $SCRIPT_DIR/run_sst2_label.sh
sst2_exit=$?
echo "[$(date)] <<< SST-2 finished  (exit code $sst2_exit)"

t1=$(date +%s)
elapsed=$((t1 - t0))
echo
echo "================================================================"
echo "[$(date)] Combined run done"
echo "  AG News exit = $agnews_exit"
echo "  SST-2 exit   = $sst2_exit"
echo "  total elapsed = ${elapsed}s  ($((elapsed/60)) min)"
echo "================================================================"

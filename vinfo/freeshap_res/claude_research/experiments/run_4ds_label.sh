#!/bin/sh
# Sequential wrapper: MNLI -> MR -> MRPC -> QQP A1 sweep + aggregation.
# Each dataset script aborts on its own error (set -e), but the wrapper does
# NOT set -e, so one dataset failing won't stop the others.

SCRIPT_DIR=freeshap_res/claude_research/experiments

t0=$(date +%s)
echo "================================================================"
echo "[$(date)] 4-dataset A1 run starting: MNLI -> MR -> MRPC -> QQP"
echo "================================================================"

for DS in mnli mr mrpc qqp; do
    echo
    echo "================================================================"
    echo "[$(date)] >>> $DS"
    echo "================================================================"
    bash $SCRIPT_DIR/run_${DS}_label.sh
    eval "${DS}_exit=$?"
    eval "echo \"[$(date)] <<< $DS finished  (exit code \$${DS}_exit)\""
done

t1=$(date +%s)
elapsed=$((t1 - t0))
echo
echo "================================================================"
echo "[$(date)] 4-dataset run done"
echo "  MNLI exit  = $mnli_exit"
echo "  MR exit    = $mr_exit"
echo "  MRPC exit  = $mrpc_exit"
echo "  QQP exit   = $qqp_exit"
echo "  total elapsed = ${elapsed}s  ($((elapsed/60)) min)"
echo "================================================================"

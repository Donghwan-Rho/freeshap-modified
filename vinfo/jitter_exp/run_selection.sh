#!/bin/sh
# ============================================================
# 사용: sh jitter_exp/run_selection.sh <dataset> [seeds...]
#   예:  sh jitter_exp/run_selection.sh qqp
#        sh jitter_exp/run_selection.sh mr 2024
#
# task_shapley 로 이미 만든 SV(pkl)를 재사용해 data_selection predictions 만 생성.
# 존재하는 모든 (lam,eps) 셀 + inv (리포트가 전체 격자를 그리므로 십자로 제한하지 않음).
# 이미 만든 predictions 는 skip (resume 가능).
# (run_removal.sh 와 동일한 구조 — task_data_selection.py 사용, REM 없음)
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
DS=${1:?"dataset 이름을 인자로 주세요 (예: qqp)"}
shift
SEEDS="${*:-2024}"
x() { echo "$1" | grep -oP "$2"; }

for S in $SEEDS; do
  # ---- eigen: 존재하는 모든 (lam,eps) 셀 (리포트가 전체 격자를 그리므로) ----
  for f in jitter_exp/res/shapley/$DS/eigen/bert_seed${S}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    out="jitter_exp/res/data_selection/$DS/eigen/predictions/${b%.pkl}_predictions.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+')
    rank=$(x "$b" '_eig\K[0-9.]+'); elam=$(x "$b" '_eiglam\K[0-9.e+-]+'); eeps=$(x "$b" '_eigeps\K[0-9.e+-]+')
    tmc=$(x "$b" '_tmc\K[0-9]+')
    echo "[run] eigen $DS seed$S eiglam$elam eigeps$eeps"
    $PY task_data_selection.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate eigen --eigen_rank $rank --eigen_lambda_ $elam --eigeps $eeps \
      --inv_lambda_ 1e-6 --tmc_iter $tmc --out_root ./jitter_exp/res
  done

  # ---- nystrom: 존재하는 모든 (lam,eps) 셀 ----
  for f in jitter_exp/res/shapley/$DS/nystrom/bert_seed${S}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    out="jitter_exp/res/data_selection/$DS/nystrom/predictions/${b%.pkl}_predictions.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+')
    nd=$(x "$b" '_nys\K[0-9.]+'); nlam=$(x "$b" '_nyslam\K[0-9.e+-]+'); neps=$(x "$b" '_nyseps\K[0-9.e+-]+')
    tmc=$(x "$b" '_tmc\K[0-9]+')
    echo "[run] nystrom $DS seed$S nyslam$nlam nyseps$neps"
    $PY task_data_selection.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate nystrom --nystrom_d $nd --nystrom_lambda_ $nlam --nyseps $neps \
      --inv_lambda_ 1e-6 --tmc_iter $tmc --out_root ./jitter_exp/res
  done

  # ---- inv (exact 곡선용, freeshap_res) ----
  for f in freeshap_res/shapley/$DS/inv/bert_seed${S}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    out="freeshap_res/data_selection/$DS/inv/predictions/${b%.pkl}_predictions.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+')
    ilam=$(x "$b" '_lam\K[0-9.e+-]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    echo "[run] inv $DS seed$S"
    $PY task_data_selection.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate inv --inv_lambda_ $ilam --tmc_iter $tmc --out_root ./freeshap_res
  done
done
echo "[done] $DS selection"

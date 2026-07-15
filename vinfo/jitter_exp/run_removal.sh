#!/bin/sh
# ============================================================
# 사용: sh jitter_exp/run_removal.sh <dataset> [seeds...]
#   예:  sh jitter_exp/run_removal.sh qqp
#        sh jitter_exp/run_removal.sh mr 2024
#
# task_shapley 로 이미 만든 SV(pkl)를 재사용해 removal predictions 만 생성.
# 존재하는 모든 (lam,eps) 셀 + inv (리포트가 전체 격자를 그리므로 십자로 제한하지 않음).
# REM 은 0~50% (그 이상은 remaining set 이 작아 kernel regression 이 깨짐 → 제외).
# 이미 만든 predictions 는 skip (resume 가능).
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
DS=${1:?"dataset 이름을 인자로 주세요 (예: qqp)"}
shift
SEEDS="${*:-2024}"
# 제거 % 격자: 0~99, 1%씩 (100점) — task_data_removal.py 기본값과 동일.
# 고-제거%에서 acc 가 chance 아래로 떨어지는 건 정상(최악 점만 남겨 학습) — "bad kernel regression"은
# test_loss>1 경고일 뿐 acc 는 실제값(garbage 아님).
REM="$(seq -s' ' 0 99)"
x() { echo "$1" | grep -oP "$2"; }

for S in $SEEDS; do
  # ---- eigen: 존재하는 모든 (lam,eps) 셀 (리포트가 전체 격자를 그리므로) ----
  for f in jitter_exp/res/shapley/$DS/eigen/*_seed${S}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    out="jitter_exp/res/data_removing/$DS/eigen/predictions/${b%.pkl}_predictions.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+')
    rank=$(x "$b" '_eig\K[0-9.]+'); elam=$(x "$b" '_eiglam\K[0-9.e+-]+'); eeps=$(x "$b" '_eigeps\K[0-9.e+-]+')
    tmc=$(x "$b" '_tmc\K[0-9]+')
    echo "[run] eigen $DS seed$S eiglam$elam eigeps$eeps"
    $PY task_data_removal.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate eigen --eigen_rank $rank --eigen_lambda_ $elam --eigeps $eeps \
      --inv_lambda_ 1e-6 --tmc_iter $tmc --out_root ./jitter_exp/res --num_train_removed_list $REM
  done

  # ---- nystrom: 존재하는 모든 (lam,eps) 셀 ----
  for f in jitter_exp/res/shapley/$DS/nystrom/*_seed${S}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    out="jitter_exp/res/data_removing/$DS/nystrom/predictions/${b%.pkl}_predictions.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+')
    nd=$(x "$b" '_nys\K[0-9.]+'); nlam=$(x "$b" '_nyslam\K[0-9.e+-]+'); neps=$(x "$b" '_nyseps\K[0-9.e+-]+')
    tmc=$(x "$b" '_tmc\K[0-9]+')
    echo "[run] nystrom $DS seed$S nyslam$nlam nyseps$neps"
    $PY task_data_removal.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate nystrom --nystrom_d $nd --nystrom_lambda_ $nlam --nyseps $neps \
      --inv_lambda_ 1e-6 --tmc_iter $tmc --out_root ./jitter_exp/res --num_train_removed_list $REM
  done

  # ---- inv (exact 곡선용, freeshap_res) ----
  for f in freeshap_res/shapley/$DS/inv/*_seed${S}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    out="freeshap_res/data_removing/$DS/inv/predictions/${b%.pkl}_predictions.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+')
    ilam=$(x "$b" '_lam\K[0-9.e+-]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    echo "[run] inv $DS seed$S"
    $PY task_data_removal.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate inv --inv_lambda_ $ilam --tmc_iter $tmc --out_root ./freeshap_res --num_train_removed_list $REM
  done
done
echo "[done] $DS removal"

#!/bin/sh
# ============================================================
# 사용: sh jitter_exp/run_downstream_pinv_lev.sh <dataset> [seeds...]
#   예:  sh jitter_exp/run_downstream_pinv_lev.sh qqp
#        sh jitter_exp/run_downstream_pinv_lev.sh rte 2024
# 끝나면 python jitter_exp/build_eigen_pinv_lev_report.py --dataset qqp
#
# 이미 만든 eigen / nystrom_pinv / nystrom_lev SV pkl(rank sweep, lam=1e-2 eps=1e-8)을
# 재사용해 빠진 data_selection / data_removing predictions 만 생성.
#   eigen 은 freeshap_res(eiglam1e-02_eigeps1e-8), pinv/lev 는 각자 res 폴더.
#   - SV pkl 있으면 TMC skip(로드) -> 예측만 계산 (빠름)
#   - 이미 있는 predictions txt 는 skip (resume 안전)
# report_eigen_pinv_lev 의 AUC 그래프(p2)를 채우기 위한 러너.
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
DS=${1:?"dataset 이름을 인자로 주세요 (예: qqp)"}
shift
SEEDS="${*:-2024 2025 2026}"
REM="$(seq -s' ' 0 99)"
TAG="nyslam1e-02_nyseps1e-8"   # rank-sweep 고정 세팅만 대상
x() { echo "$1" | grep -oP "$2"; }

for S in $SEEDS; do
  # ---------- eigen (freeshap_res, rank sweep) ----------
  for f in freeshap_res/shapley/$DS/eigen/bert_seed${S}_num2000_*_eiglam1e-02_eigeps1e-8_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    er=$(x "$b" '_eig\K[0-9.]+')
    sel="freeshap_res/data_selection/$DS/eigen/predictions/${b%.pkl}_predictions.txt"
    rem="freeshap_res/data_removing/$DS/eigen/predictions/${b%.pkl}_predictions.txt"
    if [ ! -e "$sel" ]; then
      echo "[sel] eigen $DS seed$S rank=${er}%"
      $PY task_data_selection.py --config ntk_prompt --dataset_name $DS --seed $S \
        --num_train_dp $num --val_sample_num $val --approximate eigen --eigen_rank $er \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter $tmc \
        --out_root ./freeshap_res
    fi
    if [ ! -e "$rem" ]; then
      echo "[rem] eigen $DS seed$S rank=${er}%"
      $PY task_data_removal.py --config ntk_prompt --dataset_name $DS --seed $S \
        --num_train_dp $num --val_sample_num $val --approximate eigen --eigen_rank $er \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter $tmc \
        --out_root ./freeshap_res --num_train_removed_list $REM
    fi
  done

  # ---------- nystrom_pinv ----------
  for f in jitter_exp/nys_pinv_res/shapley/$DS/nystrom_pinv/bert_seed${S}_*_${TAG}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    nd=$(x "$b" '_nyspinv\K[0-9.]+')
    sel="jitter_exp/nys_pinv_res/data_selection/$DS/nystrom_pinv/predictions/${b%.pkl}_predictions.txt"
    rem="jitter_exp/nys_pinv_res/data_removing/$DS/nystrom_pinv/predictions/${b%.pkl}_predictions.txt"
    if [ ! -e "$sel" ]; then
      echo "[sel] pinv $DS seed$S rank=${nd}%"
      $PY task_data_selection.py --config ntk_prompt --dataset_name $DS --seed $S \
        --num_train_dp $num --val_sample_num $val --approximate nystrom_pinv --nystrom_d $nd \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter $tmc \
        --out_root ./jitter_exp/nys_pinv_res
    fi
    if [ ! -e "$rem" ]; then
      echo "[rem] pinv $DS seed$S rank=${nd}%"
      $PY task_data_removal.py --config ntk_prompt --dataset_name $DS --seed $S \
        --num_train_dp $num --val_sample_num $val --approximate nystrom_pinv --nystrom_d $nd \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter $tmc \
        --out_root ./jitter_exp/nys_pinv_res --num_train_removed_list $REM
    fi
  done

  # ---------- nystrom_lev ----------
  for f in jitter_exp/nys_lev_res/shapley/$DS/nystrom_lev/bert_seed${S}_*_${TAG}_*.pkl; do
    [ -e "$f" ] || continue
    case "$f" in *poison*) continue;; esac
    b=$(basename "$f")
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    nd=$(x "$b" '_nyslev\K[0-9.]+')
    sel="jitter_exp/nys_lev_res/data_selection/$DS/nystrom_lev/predictions/${b%.pkl}_predictions.txt"
    rem="jitter_exp/nys_lev_res/data_removing/$DS/nystrom_lev/predictions/${b%.pkl}_predictions.txt"
    if [ ! -e "$sel" ]; then
      echo "[sel] lev $DS seed$S rank=${nd}%"
      $PY task_data_selection.py --config ntk_prompt --dataset_name $DS --seed $S \
        --num_train_dp $num --val_sample_num $val --approximate nystrom_lev --nystrom_d $nd \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter $tmc \
        --out_root ./jitter_exp/nys_lev_res
    fi
    if [ ! -e "$rem" ]; then
      echo "[rem] lev $DS seed$S rank=${nd}%"
      $PY task_data_removal.py --config ntk_prompt --dataset_name $DS --seed $S \
        --num_train_dp $num --val_sample_num $val --approximate nystrom_lev --nystrom_d $nd \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter $tmc \
        --out_root ./jitter_exp/nys_lev_res --num_train_removed_list $REM
    fi
  done
done
echo "[done] $DS eigen/pinv/lev downstream (seeds:$SEEDS)"

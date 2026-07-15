#!/bin/sh
# ============================================================
# 전체 grid search: dataset × seed × {inv, nystrom 4x4, eigen 4x4} × {shapley, selection, removal} + wld
#
#   nystrom : lam {1e-4,1e-3,1e-2,1e-1} x nyseps {1e-1,1,1e+1,1e+2}    (최적 lam=1e-2, eps=1e+1 내부)
#   eigen   : lam {1e-3,1e-2,1e-1,1}    x eigeps {1e-8,1e-6,1e-4,1e-2}  (배포값 1e-8 포함, 6 decade)
#   wld     : 위와 동일한 격자 + inv, poison 10%
#
# out_root 규약 (리포트가 찾는 위치와 반드시 일치):
#   inv  의 shapley/selection/removal  -> ./freeshap_res
#   nys/eig 의 shapley/selection/removal -> ./jitter_exp/res
#   wld 는 inv 포함 전부              -> ./jitter_exp/res
#
# 모든 task 가 기존 결과 있으면 로드/skip → 재실행 안전(증분).
# 사용:  sh jitter_exp/grid_search.sh
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python

SEEDS="2024 2025 2026"
DATASETS="qqp mr mrpc"
NYS_LAMS="1e-4 1e-3 1e-2 1e-1"; NYS_EPSS="1e-1 1 1e+1 1e+2"
EIG_LAMS="1e-3 1e-2 1e-1 1";    EIG_EPSS="1e-8 1e-6 1e-4 1e-2"
REM="$(seq -s' ' 0 99)"          # removal 제거 % 격자: 0~99 1%씩 (run_removal.sh 와 동일해야 함)
POISON=10

for S in $SEEDS; do
  for D in $DATASETS; do

    # ---- dataset 별 val_sample_num ----
    case "$D" in
      sst2) V=872  ;;
      mrpc) V=408  ;;
      rte)  V=277  ;;
      *)    V=1000 ;;
    esac
    echo "################ $D (val=$V)  seed=$S ################"

    # ========== 1) inv (exact 기준선) — out_root=./freeshap_res ==========
    echo "[inv] $D seed$S"
    $PY task_shapley.py --config ntk_prompt --seed $S --dataset_name $D \
      --num_train_dp 2000 --val_sample_num $V --approximate inv \
      --inv_lambda_ 1e-6 --tmc_iter 500 --out_root ./freeshap_res
    $PY task_data_selection.py --config ntk_prompt --seed $S --dataset_name $D \
      --num_train_dp 2000 --val_sample_num $V --approximate inv \
      --inv_lambda_ 1e-6 --tmc_iter 500 --out_root ./freeshap_res
    $PY task_data_removal.py --config ntk_prompt --seed $S --dataset_name $D \
      --num_train_dp 2000 --val_sample_num $V --approximate inv \
      --inv_lambda_ 1e-6 --tmc_iter 500 --out_root ./freeshap_res \
      --num_train_removed_list $REM

    # ========== 2) nystrom 4x4 — out_root=./jitter_exp/res ==========
    for L in $NYS_LAMS; do
      for E in $NYS_EPSS; do
        echo "[nys] $D seed$S lam=$L eps=$E"
        $PY task_shapley.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom --nystrom_d 20 \
          --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root ./jitter_exp/res
        $PY task_data_selection.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom --nystrom_d 20 \
          --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root ./jitter_exp/res
        $PY task_data_removal.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom --nystrom_d 20 \
          --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root ./jitter_exp/res \
          --num_train_removed_list $REM
      done
    done

    # ========== 3) eigen 4x4 — out_root=./jitter_exp/res ==========
    for L in $EIG_LAMS; do
      for E in $EIG_EPSS; do
        echo "[eig] $D seed$S lam=$L eps=$E"
        $PY task_shapley.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate eigen --eigen_rank 20 \
          --inv_lambda_ 1e-6 --eigen_lambda_ $L --eigeps $E --tmc_iter 500 --out_root ./jitter_exp/res
        $PY task_data_selection.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate eigen --eigen_rank 20 \
          --inv_lambda_ 1e-6 --eigen_lambda_ $L --eigeps $E --tmc_iter 500 --out_root ./jitter_exp/res
        $PY task_data_removal.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate eigen --eigen_rank 20 \
          --inv_lambda_ 1e-6 --eigen_lambda_ $L --eigeps $E --tmc_iter 500 --out_root ./jitter_exp/res \
          --num_train_removed_list $REM
      done
    done

  done
done
echo "[done] grid search (all datasets, all tasks)"

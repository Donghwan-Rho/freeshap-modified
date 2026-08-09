#!/bin/sh
# ============================================================
# pseudoinverse-Nystrom (nystrom_pinv) FULL grid.
#   grid_search.sh 의 nystrom eps/lambda grid 와 동일:
#     nystrom_lambda_ {1e-4,1e-3,1e-2,1e-1}  x  nyseps {1e-1,1,1e+1,1e+2}
#   seed 3개 x dataset 7개, for 문 구조도 grid_search.sh 와 동일.
#
# nystrom_pinv 만 실행 (shapley/selection/removal). inv/eigen 제외
#   (inv 기준선은 grid_search 가 만든 ./freeshap_res 것을 리포트가 재사용).
# 출력: ./jitter_exp/nys_pinv_res  (method_dir=nystrom_pinv, 파일명 _nyspinv…)
#
# 모든 task 는 기존 결과 있으면 로드/skip → 재실행 안전(증분).
# 사용:  sh jitter_exp/nys_pinv.sh
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python

SEEDS="2024 2025 2026"
DATASETS="qqp mr rte sst2 mnli ag_news mrpc"
NYS_LAMS="1e-3 1e-2 1e-1 1"; NYS_EPSS="1e-8 1e-6 1e-4 1e-2"
REM="$(seq -s' ' 0 99)"          # removal 제거 % 격자: 0~99 (grid_search.sh 와 동일)
OUT=./jitter_exp/nys_pinv_res

for S in $SEEDS; do
  for D in $DATASETS; do

    # ---- dataset 별 val_sample_num (grid_search.sh 와 동일) ----
    case "$D" in
      sst2) V=872  ;;
      mrpc) V=408  ;;
      rte)  V=277  ;;
      *)    V=1000 ;;
    esac
    echo "################ nystrom_pinv  $D (val=$V)  seed=$S ################"

    for L in $NYS_LAMS; do
      for E in $NYS_EPSS; do
        echo "[nys_pinv] $D seed$S lam=$L eps=$E"
        $PY task_shapley.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d 20 \
          --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root $OUT
        $PY task_data_selection.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d 20 \
          --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root $OUT
        $PY task_data_removal.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d 20 \
          --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root $OUT \
          --num_train_removed_list $REM
      done
    done

  done
done
echo "[done] nystrom_pinv full grid (datasets: $DATASETS, seeds: $SEEDS)"

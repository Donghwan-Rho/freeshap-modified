#!/bin/sh
# ============================================================
# grid_search.sh 의 노드 분할 버전.
#   sel 세트 (task_shapley → data_selection → data_removal) : $SEL_DATASETS
#   wld 세트 (task_wrong_label_detection)                   : $WLD_DATASETS
# 진행 순서: seed 단위로 sel → wld 를 번갈아 수행
#   (seed2024: sel → wld) → (seed2025: sel → wld) → (seed2026: sel → wld)
#
# out_root 규약:
#   inv 의 shapley/selection/removal     -> ./freeshap_res
#   nys/eig 의 shapley/selection/removal -> ./jitter_exp/res
#   wld 는 inv 포함 전부                  -> ./jitter_exp/res
# 모든 task 는 기존 결과 있으면 로드/skip → 재실행 안전(증분).
#
# 사용:  sh jitter_exp/private_ai.sh
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python

# ───── 이 노드가 맡을 작업 (노드마다 여기만 바꾸면 됨) ─────
SEL_DATASETS="rte"      # shapley + data_selection + data_removal 돌릴 데이터셋
WLD_DATASETS="qqp"      # wrong_label_detection 돌릴 데이터셋
SEEDS="2024 2025 2026"
# ──────────────────────────────────────────────────────

NYS_LAMS="1e-4 1e-3 1e-2 1e-1"; NYS_EPSS="1e-1 1 1e+1 1e+2"
EIG_LAMS="1e-3 1e-2 1e-1 1";    EIG_EPSS="1e-8 1e-6 1e-4 1e-2"
REM="$(seq -s' ' 0 99)"          # removal 제거 % 격자: 0~99 1%씩
POISON=10

# dataset -> val_sample_num
getval() {
  case "$1" in
    sst2) echo 872  ;;
    mrpc) echo 408  ;;
    rte)  echo 277  ;;
    *)    echo 1000 ;;
  esac
}

for S in $SEEDS; do

  # ======================================================
  # 1) sel 세트: shapley → data_selection → data_removal
  # ======================================================
  for D in $SEL_DATASETS; do
    V=$(getval "$D")
    echo "########## [sel] $D (val=$V)  seed=$S ##########"

    # ---- inv (exact 기준선) — out_root=./freeshap_res ----
    echo "[sel-inv] $D seed$S"
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

    # ---- nystrom 4x4 — out_root=./jitter_exp/res ----
    for L in $NYS_LAMS; do
      for E in $NYS_EPSS; do
        echo "[sel-nys] $D seed$S lam=$L eps=$E"
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

    # ---- eigen 4x4 — out_root=./jitter_exp/res ----
    for L in $EIG_LAMS; do
      for E in $EIG_EPSS; do
        echo "[sel-eig] $D seed$S lam=$L eps=$E"
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

  # ======================================================
  # 2) wld 세트: task_wrong_label_detection (inv → nys 4x4 → eig 4x4)
  # ======================================================
  for D in $WLD_DATASETS; do
    V=$(getval "$D")
    echo "########## [wld] $D (val=$V)  seed=$S ##########"

    echo "[wld-inv] $D seed$S"
    $PY task_wrong_label_detection.py --dataset_name $D --seed $S \
      --num_train_dp 2000 --val_sample_num $V --approximate inv \
      --inv_lambda_ 1e-6 --poison_pct $POISON --tmc_iter 500 --out_root ./jitter_exp/res

    for L in $NYS_LAMS; do
      for E in $NYS_EPSS; do
        echo "[wld-nys] $D seed$S lam=$L eps=$E"
        $PY task_wrong_label_detection.py --dataset_name $D --seed $S \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom --nystrom_d 20 \
          --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E \
          --poison_pct $POISON --tmc_iter 500 --out_root ./jitter_exp/res
      done
    done

    for L in $EIG_LAMS; do
      for E in $EIG_EPSS; do
        echo "[wld-eig] $D seed$S lam=$L eps=$E"
        $PY task_wrong_label_detection.py --dataset_name $D --seed $S \
          --num_train_dp 2000 --val_sample_num $V --approximate eigen --eigen_rank 20 \
          --inv_lambda_ 1e-6 --eigen_lambda_ $L --eigeps $E \
          --poison_pct $POISON --tmc_iter 500 --out_root ./jitter_exp/res
      done
    done
  done

done

echo "[done] private_ai: sel=[$SEL_DATASETS] wld=[$WLD_DATASETS] seeds=[$SEEDS]"

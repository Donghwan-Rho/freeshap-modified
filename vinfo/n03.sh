#!/bin/sh
# ============================================================
# CIFAR-10 vision n=5000 — inv + eigen 의 shapley + selection 재측정 러너.
#   목적: 기존 n=5000 자료가 RTX 2080 Ti 에서 측정돼 있어 speedup 비교의 GPU 를
#         3090 으로 통일하기 위해 "타이밍이 들어가는" shapley/selection 만 다시 만든다.
#         inv 는 λ 를 1e-2 -> 1e-6 로 통일하므로 removal/wld 도 함께 새로 만든다.
#
# 세팅: num_train=$N (기본 5000), val=1000, tmc=500,
#       inv λ=$INV_LAM (기본 1e-6 — 다른 캠페인과 통일), eigen λ=1e-2, eigeps=1e-8.
#   ※ 옛 inv(λ=1e-2) 산출물은 전부 백업으로 옮겨지고, removal 은 저비용이지만
#     wld 는 poison TMC 라 seed 당 ~3h 추가로 든다.
#
# 사용: 인자에서 숫자 -> seed, 문자 -> dataset (기본 seed 2024 2025 2026, cifar10).
#   CUDA_VISIBLE_DEVICES=<3090> sh n03.sh          # 전체 (기존 것은 자동 백업 후 재측정)
#   CUDA_VISIBLE_DEVICES=<3090> sh n03.sh 2024     # seed 하나만 (GPU 나눠 병렬 가능)
#   DO_SELECTION=0 sh n03.sh                       # shapley 만
#
# ※ 기존 2080 Ti 산출물(inv λ=1e-2 전부 + eigen shapley/selection/removal)은
#   backup_2080ti/ 로 이미 옮겨둠 (129개). 이 스크립트는 실행만 한다.
# 재실행 순서: n03.sh (shapley+selection) -> vision/wld_removal_vision.sh (removal+wld)
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
N=${N:-5000}
V=${V:-1000}
INV_LAM=${INV_LAM:-1e-6}
EIG_LAM=${EIG_LAM:-1e-2}
EIGEPS=${EIGEPS:-1e-8}
DO_SHAPLEY=${DO_SHAPLEY:-1}
DO_SELECTION=${DO_SELECTION:-1}
RANKS=${RANKS:-"1 5 10 15 20 25 30"}

# 파일명 태그 (python 의 f"{lam:.0e}" 표기와 일치)
case "$INV_LAM" in
  1e-6) INV_TAG=1e-06 ;;
  1e-2) INV_TAG=1e-02 ;;
  *)    INV_TAG=$INV_LAM ;;
esac
case "$EIG_LAM" in
  1e-2) EIG_TAG=1e-02 ;;
  1e-6) EIG_TAG=1e-06 ;;
  *)    EIG_TAG=$EIG_LAM ;;
esac

# ---- 인자 분류: 숫자=seed, 그 외=dataset ----
SEEDS=""; DATASETS=""
for a in "$@"; do
  case "$a" in
    [0-9]*) SEEDS="$SEEDS $a" ;;
    *)      DATASETS="$DATASETS $a" ;;
  esac
done
SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-cifar10}"
echo "[cfg] seeds:$SEEDS datasets:$DATASETS num=$N val=$V inv_lam=$INV_LAM eig_lam=$EIG_LAM eigeps=$EIGEPS"

for S in $SEEDS; do
  for D in $DATASETS; do
    echo "################ vision shapley+selection  $D seed=$S (num=$N) ################"

    # ---- NTK 캐시 (라벨/λ 무관 — 있으면 그대로 사용) ----
    NTK=./freeshap_res/ntk/$D/resnet_seed${S}_num${N}_val${V}_signFalse.pkl
    if [ -f "$NTK" ]; then
      echo "[skip] ntk ($NTK)"
    else
      $PY vision/task_ntk_vision.py --config ntk_vision --seed $S --dataset_name $D \
        --num_train_dp $N --val_sample_num $V
    fi

    # ============ inv ============
    STEM="resnet_seed${S}_num${N}_val${V}_lam${INV_TAG}_signFalse_earlystopTrue_tmc500"
    SV_PKL=./freeshap_res/shapley/$D/inv/${STEM}.pkl
    SEL_TXT=./freeshap_res/data_selection/$D/inv/indices/${STEM}_indices.txt
    SEL_PRED=./freeshap_res/data_selection/$D/inv/predictions/${STEM}_predictions.txt
    if [ "$DO_SHAPLEY" = "1" ]; then
      if [ -f "$SV_PKL" ]; then echo "  [skip] inv shapley"
      else
        $PY vision/task_shapley_vision.py --config ntk_vision --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ $INV_LAM \
          --tmc_iter 500
      fi
    fi
    if [ "$DO_SELECTION" = "1" ]; then
      if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] inv selection"
      elif [ -f "$SEL_TXT" ] && [ -f "$SEL_PRED" ]; then echo "  [skip] inv selection"
      else
        $PY vision/task_data_selection_vision.py --config ntk_vision --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ $INV_LAM \
          --tmc_iter 500
      fi
    fi

    # ============ eigen rank sweep ============
    for R in $RANKS; do
      echo "[eigen] $D seed$S rank=${R}%"
      STEM="resnet_seed${S}_num${N}_val${V}_eig${R}.0_eiglam${EIG_TAG}_eigeps${EIGEPS}_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      SV_PKL=./freeshap_res/shapley/$D/eigen/${STEM}.pkl
      SEL_TXT=./freeshap_res/data_selection/$D/eigen/indices/${STEM}_indices.txt
      SEL_PRED=./freeshap_res/data_selection/$D/eigen/predictions/${STEM}_predictions.txt
      if [ "$DO_SHAPLEY" = "1" ]; then
        if [ -f "$SV_PKL" ]; then echo "  [skip] shapley"
        else
          $PY vision/task_shapley_vision.py --config ntk_vision --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ $EIG_LAM --eigeps $EIGEPS --tmc_iter 500
        fi
      fi
      if [ "$DO_SELECTION" = "1" ]; then
        if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] selection"
        elif [ -f "$SEL_TXT" ] && [ -f "$SEL_PRED" ]; then echo "  [skip] selection"
        else
          $PY vision/task_data_selection_vision.py --config ntk_vision --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ $EIG_LAM --eigeps $EIGEPS --tmc_iter 500
        fi
      fi
    done

  done
done
echo "[done] vision inv+eigen shapley+selection (seeds:$SEEDS / num=$N / inv_lam=$INV_LAM)"

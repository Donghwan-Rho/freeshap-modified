#!/bin/sh
# ============================================================
# CIFAR-10 vision — removal + wrong-label detection 일괄 러너 (inv + eigen 전용).
#   방법: inv + eigen (rank sweep 1~30%).  ※ nystrom 계열은 n=5000 에 불필요해 제거함
#        (pinv/lev 는 n=2000 리포트용으로만 쓰며 이미 완비).
#   removal : 기존 SV pkl 재사용 (TMC 없음) -> top/bottom/random 제거 곡선
#   wld     : poison(라벨 10% flip) 데이터로 SV 를 새로 계산 (TMC 필요 — 느림)
#             ntk_vision_poison.yaml + task_wrong_label_detection_vision.py
#             NTK 는 라벨 무관이라 기존 캐시 재사용.
#
# 세팅: num_train=$N (기본 5000), val=1000, tmc=500, eigen/nys lam=1e-2, eps=1e-8.
#   n=5000 캠페인의 inv SV 는 lam1e-02 로 만들어져 있어 INV_LAM 기본값도 1e-2
#   (n=2000 이면 1e-6 이므로 아래 case 에서 자동 전환).
#
# 사용: 인자에서 숫자 -> seed, 문자 -> dataset (기본 seed 2024 2025 2026, cifar10).
#   CUDA_VISIBLE_DEVICES=<GPU> sh vision/wld_removal_vision.sh              # 전체
#   CUDA_VISIBLE_DEVICES=<GPU> sh vision/wld_removal_vision.sh 2024         # seed 하나
#   N=2000 CUDA_VISIBLE_DEVICES=<GPU> sh vision/wld_removal_vision.sh       # n 바꾸기
#   DO_WLD=0 sh vision/wld_removal_vision.sh                                # removal 만
#   DO_REMOVAL=0 sh vision/wld_removal_vision.sh                            # wld 만
# 결과 파일 있으면 스킵 (증분 안전).
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
REM="$(seq -s' ' 0 99)"
N=${N:-5000}
V=${V:-1000}
DO_REMOVAL=${DO_REMOVAL:-1}
DO_WLD=${DO_WLD:-1}
POISON=${POISON:-10}

# n 별 inv lambda (기존 SV pkl 과 맞춰야 TMC 재계산을 피함)
INV_LAM=${INV_LAM:-1e-6}   # n5000 캠페인도 1e-6 로 통일 (기존 1e-2 자료는 backup_2080ti/ 로 이동)
case "$INV_LAM" in
  1e-2) INV_TAG=1e-02 ;;
  1e-6) INV_TAG=1e-06 ;;
  *)    INV_TAG=$INV_LAM ;;
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
echo "[cfg] seeds:$SEEDS datasets:$DATASETS num=$N val=$V inv_lam=$INV_LAM removal=$DO_REMOVAL wld=$DO_WLD"

for S in $SEEDS; do
  for D in $DATASETS; do
    echo "################ vision removal+wld  $D  seed=$S  (num=$N) ################"

    # ============ inv ============
    STEM="resnet_seed${S}_num${N}_val${V}_lam${INV_TAG}_signFalse_earlystopTrue_tmc500"
    WSTEM="${STEM}_poison${POISON}_ps${S}"
    SV=./freeshap_res/shapley/$D/inv/${STEM}.pkl
    REM_TXT=./freeshap_res/data_removing/$D/inv/predictions/${STEM}_predictions.txt
    WLD_TXT=./freeshap_res/wrong_label_detection/$D/inv/predictions/${WSTEM}_detection.txt
    WLD_PKL=./freeshap_res/wrong_label_detection/$D/inv/predictions/${WSTEM}_detection.pkl

    if [ "$DO_REMOVAL" = "1" ]; then
      if [ ! -f "$SV" ]; then
        echo "  [no-sv] inv removal ($SV) — shapley 먼저 필요"
      elif [ -f "$REM_TXT" ]; then
        echo "  [skip] inv removal"
      else
        $PY vision/task_data_removal_vision.py --config ntk_vision --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ $INV_LAM \
          --tmc_iter 500 --num_train_removed_list $REM
      fi
    fi
    if [ "$DO_WLD" = "1" ]; then
      if [ -f "$WLD_TXT" ] && [ -f "$WLD_PKL" ]; then
        echo "  [skip] inv wld"
      else
        $PY vision/task_wrong_label_detection_vision.py --config ntk_vision --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ $INV_LAM \
          --poison_pct $POISON --tmc_iter 500
      fi
    fi

    # ============ rank sweep: eigen ============
    for R in 1 5 10 15 20 25 30; do
      echo "[rank ${R}%] $D seed$S"

      # ---- eigen ----
      STEM="resnet_seed${S}_num${N}_val${V}_eig${R}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      # wld 파일명은 invlam 태그가 없음 (NLP 판과 동일 규약)
      WSTEM="resnet_seed${S}_num${N}_val${V}_eig${R}.0_eiglam1e-02_eigeps1e-8_cholesky_float32_signFalse_earlystopTrue_tmc500_poison${POISON}_ps${S}"
      SV=./freeshap_res/shapley/$D/eigen/${STEM}.pkl
      REM_TXT=./freeshap_res/data_removing/$D/eigen/predictions/${STEM}_predictions.txt
      WLD_TXT=./freeshap_res/wrong_label_detection/$D/eigen/predictions/${WSTEM}_detection.txt
      WLD_PKL=./freeshap_res/wrong_label_detection/$D/eigen/predictions/${WSTEM}_detection.pkl
      if [ "$DO_REMOVAL" = "1" ]; then
        if [ ! -f "$SV" ]; then echo "  [no-sv] eigen r$R removal"
        elif [ -f "$REM_TXT" ]; then echo "  [skip] eigen r$R removal"
        else
          $PY vision/task_data_removal_vision.py --config ntk_vision --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500 \
            --num_train_removed_list $REM
        fi
      fi
      if [ "$DO_WLD" = "1" ]; then
        if [ -f "$WLD_TXT" ] && [ -f "$WLD_PKL" ]; then echo "  [skip] eigen r$R wld"
        else
          $PY vision/task_wrong_label_detection_vision.py --config ntk_vision --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct $POISON --tmc_iter 500
        fi
      fi

    done

  done
done
echo "[done] vision removal+wld (seeds:$SEEDS / datasets:$DATASETS / num=$N)"

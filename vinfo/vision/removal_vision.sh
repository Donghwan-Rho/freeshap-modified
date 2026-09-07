#!/bin/sh
# ============================================================
# CIFAR-10 vision data removal 일괄 러너.
#   기존 SV pkl(task_shapley_vision 결과)을 재사용해 top/bottom/random 제거 곡선 생성.
#   방법: inv + eigen/nystrom/nystrom_pinv/nystrom_lev (rank sweep 1~30%).
#   세팅: num_train=2000, val=1000, lam=1e-2(eigen/nys)/1e-6(inv), eps=1e-8, tmc=500.
#   출력: ./freeshap_res/data_removing/cifar10/{method}/predictions/..._predictions.txt
#
# 스킵 논리:
#   - SV pkl 없으면 [no-sv] 출력 후 건너뜀 (shapley 먼저 돌려야 함)
#   - removal predictions txt 이미 있으면 [skip]
#
# 사용: 인자에서 숫자 -> seed, 문자 -> dataset (기본 seed 2024 2025 2026, cifar10).
#   CUDA_VISIBLE_DEVICES=<GPU> sh vision/removal_vision.sh
#   CUDA_VISIBLE_DEVICES=<GPU> sh vision/removal_vision.sh 2024
# 실행 위치 무관 (스크립트가 vinfo 로 cd).
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
REM="$(seq -s' ' 0 99)"

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
N=2000; V=1000
echo "[cfg] seeds:$SEEDS  datasets:$DATASETS  (num=$N val=$V)"

run_removal() {
  # $1=dataset $2=seed $3=method_dir $4=stem $5...=task args
  _D=$1; _S=$2; _MD=$3; _STEM=$4; shift 4
  SV=./freeshap_res/shapley/$_D/$_MD/${_STEM}.pkl
  OUT=./freeshap_res/data_removing/$_D/$_MD/predictions/${_STEM}_predictions.txt
  if [ ! -f "$SV" ]; then
    echo "  [no-sv] $_MD ($SV) — shapley 먼저 필요"
  elif [ -f "$OUT" ]; then
    echo "  [skip] $_MD removal ($OUT)"
  else
    $PY vision/task_data_removal_vision.py --config ntk_vision --seed $_S --dataset_name $_D \
      --num_train_dp $N --val_sample_num $V --tmc_iter 500 \
      --num_train_removed_list $REM "$@"
  fi
}

for S in $SEEDS; do
  for D in $DATASETS; do
    echo "################ removal vision  $D seed=$S ################"

    # ---- inv ----
    STEM="resnet_seed${S}_num${N}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500"
    run_removal $D $S inv "$STEM" --approximate inv --inv_lambda_ 1e-6

    for R in 1 5 10 15 20 25 30; do
      echo "[removal] $D seed$S rank=${R}%"

      # ---- eigen ----
      STEM="resnet_seed${S}_num${N}_val${V}_eig${R}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      run_removal $D $S eigen "$STEM" --approximate eigen --eigen_rank $R \
        --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8

      # ---- nystrom (uniform) ----
      STEM="resnet_seed${S}_num${N}_val${V}_nys${R}.0_nyslam1e-02_nyseps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      run_removal $D $S nystrom "$STEM" --approximate nystrom --nystrom_d $R \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8

      # ---- nystrom_pinv ----
      STEM="resnet_seed${S}_num${N}_val${V}_nyspinv${R}.0_nyslam1e-02_nyseps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      run_removal $D $S nystrom_pinv "$STEM" --approximate nystrom_pinv --nystrom_d $R \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8

      # ---- nystrom_lev ----
      STEM="resnet_seed${S}_num${N}_val${V}_nyslev${R}.0_nyslam1e-02_nyseps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      run_removal $D $S nystrom_lev "$STEM" --approximate nystrom_lev --nystrom_d $R \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8
    done

  done
done
echo "[done] removal vision (seeds:$SEEDS / datasets:$DATASETS)"

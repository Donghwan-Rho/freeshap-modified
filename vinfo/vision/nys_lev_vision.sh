#!/bin/sh
# ============================================================
# leverage-score Nystrom (nystrom_lev) — CIFAR-10 vision 버전, rank sweep.
#   n_vision.sh 세팅 준용: num_train=2000, val=1000, lam=1e-2, tmc=500, config ntk_vision.
#   eps: --nyseps 1e-8 (파일명 태그 nyseps1e-8; eigen 쪽은 --eigeps, 기본 1e-8).
# 출력: ./freeshap_res (vision task 는 out_root 인자 없음; method_dir=nystrom_lev 로 격리).
#   task: NTK(캐시 없으면 생성) -> shapley -> data_selection. (vision 은 removal/wld 없음)
#
# 사용: 인자에서 숫자 -> seed, 문자 -> dataset 으로 자동 분류 (순서 무관).
#   CUDA_VISIBLE_DEVICES=<GPU> sh vision/nys_lev_vision.sh              # seed 2024 2025 2026, cifar10
#   CUDA_VISIBLE_DEVICES=<GPU> sh vision/nys_lev_vision.sh 2024        # 특정 seed 만
# 실행 위치 무관 (스크립트가 vinfo 로 cd).
# 모든 task 는 결과 파일 있으면 셸에서 스킵.
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python

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

for S in $SEEDS; do
  for D in $DATASETS; do

    # ---- NTK 캐시 (없으면 생성) ----
    NTK=./freeshap_res/ntk/$D/resnet_seed${S}_num${N}_val${V}_signFalse.pkl
    if [ -f "$NTK" ]; then
      echo "[skip] ntk ($NTK)"
    else
      $PY vision/task_ntk_vision.py --config ntk_vision --seed $S --dataset_name $D \
        --num_train_dp $N --val_sample_num $V
    fi

    for R in 1 5 10 15 20 25 30; do
      echo "[lev] $D seed$S rank=${R}%"

      # ---- 결과 파일 경로 (이미 있으면 해당 task 스킵) ----
      STEM="resnet_seed${S}_num${N}_val${V}_nyslev${R}.0_nyslam1e-02_nyseps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      SV_PKL=./freeshap_res/shapley/$D/nystrom_lev/${STEM}.pkl
      SEL_TXT=./freeshap_res/data_selection/$D/nystrom_lev/indices/${STEM}_indices.txt
      SEL_PRED=./freeshap_res/data_selection/$D/nystrom_lev/predictions/${STEM}_predictions.txt

      if [ -f "$SV_PKL" ]; then
        echo "  [skip] shapley  ($SV_PKL)"
      else
        $PY vision/task_shapley_vision.py --config ntk_vision --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate nystrom_lev --nystrom_d $R \
          --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500
      fi
      if [ -f "$SEL_TXT" ] && [ -f "$SEL_PRED" ]; then
        echo "  [skip] selection ($SEL_TXT)"   # indices+predictions 둘 다 있을 때만
      else
        $PY vision/task_data_selection_vision.py --config ntk_vision --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate nystrom_lev --nystrom_d $R \
          --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500
      fi
    done

  done
done
echo "[done] nystrom_lev vision (seeds:$SEEDS / datasets:$DATASETS)"

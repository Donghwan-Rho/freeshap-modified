#!/bin/sh
# CIFAR-10 vision 실행 스크립트 (n05_0.sh 형식)
# 실행 위치: cd /extdata1/donghwan/freeshap/vinfo  ← 여기서 실행해야 상대경로가 맞음
#   CUDA_VISIBLE_DEVICES=<빈GPU> ./vision/n_vision.sh
# seed 2024 2025 2026 전부 실행. 결과 파일이 이미 있으면 해당 task 스킵.
#   (shapley 는 pkl, selection 은 indices+predictions txt 둘 다 있을 때만 스킵)

D=cifar10; N=2000; V=1000

for S in 2024 2025 2026; do
  echo "################ vision $D seed=$S (num=$N val=$V) ################"

  # ===== 1) NTK =====
  NTK=./freeshap_res/ntk/$D/resnet_seed${S}_num${N}_val${V}_signFalse.pkl
  if [ -f "$NTK" ]; then
    echo "[skip] ntk ($NTK)"
  else
    python vision/task_ntk_vision.py --config ntk_vision --seed $S --dataset_name $D --num_train_dp $N --val_sample_num $V
  fi

  # ===== 2) Shapley + Data selection: inv =====
  STEM="resnet_seed${S}_num${N}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500"
  SV=./freeshap_res/shapley/$D/inv/${STEM}.pkl
  SEL=./freeshap_res/data_selection/$D/inv/indices/${STEM}_indices.txt
  SELP=./freeshap_res/data_selection/$D/inv/predictions/${STEM}_predictions.txt
  if [ -f "$SV" ]; then
    echo "[skip] shapley inv seed$S"
  else
    python vision/task_shapley_vision.py        --config ntk_vision --seed $S --dataset_name $D --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
  fi
  if [ -f "$SEL" ] && [ -f "$SELP" ]; then
    echo "[skip] selection inv seed$S"
  else
    python vision/task_data_selection_vision.py --config ntk_vision --seed $S --dataset_name $D --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
  fi

  # ===== 3) Shapley + Data selection: eigen (rank 여러 개) =====
  for r in 1 5 10 15 20 25 30; do
    STEM="resnet_seed${S}_num${N}_val${V}_eig${r}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
    SV=./freeshap_res/shapley/$D/eigen/${STEM}.pkl
    SEL=./freeshap_res/data_selection/$D/eigen/indices/${STEM}_indices.txt
    SELP=./freeshap_res/data_selection/$D/eigen/predictions/${STEM}_predictions.txt
    if [ -f "$SV" ]; then
      echo "[skip] shapley eigen r$r seed$S"
    else
      python vision/task_shapley_vision.py        --config ntk_vision --seed $S --dataset_name $D --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $r --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500
    fi
    if [ -f "$SEL" ] && [ -f "$SELP" ]; then
      echo "[skip] selection eigen r$r seed$S"
    else
      python vision/task_data_selection_vision.py --config ntk_vision --seed $S --dataset_name $D --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $r --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500
    fi
  done

  # ===== 4) Shapley + Data selection: nystrom (d 여러 개) =====
  for d in 1 5 10 15 20 25 30; do
    STEM="resnet_seed${S}_num${N}_val${V}_nys${d}.0_nyslam1e-02_nyseps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
    SV=./freeshap_res/shapley/$D/nystrom/${STEM}.pkl
    SEL=./freeshap_res/data_selection/$D/nystrom/indices/${STEM}_indices.txt
    SELP=./freeshap_res/data_selection/$D/nystrom/predictions/${STEM}_predictions.txt
    if [ -f "$SV" ]; then
      echo "[skip] shapley nystrom d$d seed$S"
    else
      python vision/task_shapley_vision.py        --config ntk_vision --seed $S --dataset_name $D --num_train_dp $N --val_sample_num $V --approximate nystrom --nystrom_d $d --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500
    fi
    if [ -f "$SEL" ] && [ -f "$SELP" ]; then
      echo "[skip] selection nystrom d$d seed$S"
    else
      python vision/task_data_selection_vision.py --config ntk_vision --seed $S --dataset_name $D --num_train_dp $N --val_sample_num $V --approximate nystrom --nystrom_d $d --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500
    fi
  done

done
echo "[done] n_vision (seeds: 2024 2025 2026)"

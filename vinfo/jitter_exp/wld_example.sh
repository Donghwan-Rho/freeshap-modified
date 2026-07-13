#!/bin/sh

# pdf 생성 명령어: python jitter_exp/build_wld_report.py --dataset qqp --seeds 2024 2025 2026 --num_train 2000 --tmc 500 --poison 10

cd /extdata1/donghwan/freeshap/vinfo

# ============================================================
# qqp mislabel(wrong-label) detection, poison 10%
# 핵심 질문: nystrom을 "fidelity 좋은 jitter"로 튜닝하면
#            eigen만큼 detection을 잘하나? (공정성 재실험)
#   - nystrom: nyseps sweep {1e-8(나쁜 기존값), 1e0, 1e1, 1e2}, d=20 고정
#   - eigen  : eigeps=1e-8 (eigen의 좋은 구간, 참조), rank=20 고정
#   - inv    : exact 참조
# 결과: ./jitter_exp/res/wrong_label_detection/qqp/...
# NTK 캐시는 ./freeshap_res/ntk (공유) 자동 사용
# ============================================================

# wld
for S in 2024 2025 2026; do

  # 블록1: nystrom nyseps sweep (nystrom_d=20, lam=1e-2)
  for L in 1e-3 1e-2 1e-1; do
    for E in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 1e+1; do
      python task_wrong_label_detection.py --dataset_name qqp --seed $S \
        --num_train_dp 2000 --val_sample_num 1000 --approximate nystrom \
        --nystrom_d 20 --nystrom_lambda_ $L --nyseps $E \
        --poison_pct 10 --tmc_iter 500 --out_root ./jitter_exp/res
    done
  done

  # 블록2: eigen 참조 (eigen_rank=20, eigeps=1e-8, lam=1e-2)
  for L in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1; do
    for E in 1e-8 1e-7 1e-6 1e-5 1e-4; do
      python task_wrong_label_detection.py --dataset_name qqp --seed $S \
        --num_train_dp 2000 --val_sample_num 1000 --approximate eigen \
        --eigen_rank 20 --eigen_lambda_ $L --eigeps $E \
        --poison_pct 10 --tmc_iter 500 --out_root ./jitter_exp/res
    done
  done

  # 블록3: inv (exact 참조)
  python task_wrong_label_detection.py --dataset_name qqp --seed $S \
    --num_train_dp 2000 --val_sample_num 1000 --approximate inv \
    --inv_lambda_ 1e-6 \
    --poison_pct 10 --tmc_iter 500 --out_root ./jitter_exp/res

done


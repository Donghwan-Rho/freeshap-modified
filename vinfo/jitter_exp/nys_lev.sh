#!/bin/sh
# ============================================================
# leverage-score Nystrom (nystrom_lev) probe — rank sweep.
#   pinv 와 동일한 pseudoinverse 구성, landmark 만 uniform -> ridge-leverage 표집.
#   ridge leverage: l_i = [K(K + lam*n I)^{-1}]_ii  (lam = --nystrom_lambda_)
# 완전 격리: out_root=./jitter_exp/nys_lev_res, method_dir=nystrom_lev, 태그 _nyslev.
#   기존 eigen/nystrom/nystrom_pinv 결과·경로에 영향 없음. NTK 는 ./freeshap_res/ntk 공유.
# KPS(rel_error) rank 비교용 세팅: eigen/pinv 와 동일 (lam=1e-2, eps=1e-8).
#
# 사용: 인자에서 숫자 -> seed, 문자 -> dataset 으로 자동 분류 (순서 무관).
#   sh jitter_exp/nys_lev.sh                     # seed 2024 2025 2026 x dataset 7개 전부
#   sh jitter_exp/nys_lev.sh 2024                # seed 2024 만, dataset 전부
#   sh jitter_exp/nys_lev.sh qqp rte             # 전 seed, qqp rte 만
#   sh jitter_exp/nys_lev.sh 2025 2026 qqp mr    # seed 2025 2026 x qqp mr
# 루프 구조: seed(바깥) -> dataset(안) -> rank.
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
DATASETS="${DATASETS:-qqp mr rte sst2 mnli ag_news mrpc}"
echo "[cfg] seeds:$SEEDS  datasets:$DATASETS"

for S in $SEEDS; do
  for D in $DATASETS; do

    # ---- dataset 별 val_sample_num (grid_search.sh 와 동일) ----
    case "$D" in
      sst2) V=872  ;;
      mrpc) V=408  ;;
      rte)  V=277  ;;
      *)    V=1000 ;;
    esac

    for R in 1 5 10 15 20 25 30; do
      echo "[lev] $D (val=$V) seed$S rank=${R}%"
      $PY task_shapley.py --config ntk_prompt --seed $S --dataset_name $D \
        --num_train_dp 2000 --val_sample_num $V --approximate nystrom_lev --nystrom_d $R \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
        --out_root ./jitter_exp/nys_lev_res
      $PY task_data_selection.py --config ntk_prompt --seed $S --dataset_name $D \
        --num_train_dp 2000 --val_sample_num $V --approximate nystrom_lev --nystrom_d $R \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
        --out_root ./jitter_exp/nys_lev_res
      $PY task_data_removal.py --config ntk_prompt --seed $S --dataset_name $D \
        --num_train_dp 2000 --val_sample_num $V --approximate nystrom_lev --nystrom_d $R \
        --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
        --out_root ./jitter_exp/nys_lev_res --num_train_removed_list $REM
    done

  done
done
echo "[done] nystrom_lev (seeds:$SEEDS / datasets:$DATASETS)"

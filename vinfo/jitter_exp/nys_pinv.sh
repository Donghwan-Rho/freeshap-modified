#!/bin/sh
# ============================================================
# pseudoinverse-Nystrom (nystrom_pinv) probe — rank sweep, bert (ntk_prompt) 버전.
#   landmark 는 uniform 표집, 회귀는 pseudoinverse 구성 (nyseps floor).
# 완전 격리: out_root=./jitter_exp/nys_pinv_res, method_dir=nystrom_pinv, 태그 _nyspinv.
#   기존 eigen/nystrom/nystrom_lev 결과·경로에 영향 없음. NTK 는 ./freeshap_res/ntk 공유.
# rank 비교용 세팅: eigen/lev 와 동일 (lam=1e-2, eps=1e-8 고정, grid search 없음).
#   num_train=2000 고정, rank(nystrom_d) {1,5,10,15,20,25,30}% sweep.
# wrong-label 은 {config}_poison.yaml (ntk_prompt_poison) 로 poison 데이터 로드.
#
# 사용: 인자에서 숫자 -> seed, 문자 -> dataset 으로 자동 분류 (순서 무관).
#   sh jitter_exp/nys_pinv.sh                     # seed 2024 2025 2026 x dataset 7개 전부
#   sh jitter_exp/nys_pinv.sh 2024                # seed 2024 만, dataset 전부
#   sh jitter_exp/nys_pinv.sh qqp rte             # 전 seed, qqp rte 만
#   sh jitter_exp/nys_pinv.sh 2025 2026 qqp mr    # seed 2025 2026 x qqp mr
# 루프 구조: seed(바깥) -> dataset(안) -> rank.
# 모든 task 는 결과 파일 있으면 셸에서 스킵 (+python 내부 증분 로직 이중 안전망).
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
REM="$(seq -s' ' 0 99)"
OUT=./jitter_exp/nys_pinv_res

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
      echo "[pinv] $D (val=$V) seed$S rank=${R}%"

      # ---- 결과 파일 경로 (이미 있으면 해당 task 스킵) ----
      # shapley/selection/removal 파일명: _nyspinv{R}.0_nyslam..._invlam... (신형)
      # wrong-label 파일명: _nyspinv{R}_lam..._poison10_ps{seed} (wld 고유 형식)
      STEM="bert_seed${S}_num2000_val${V}_nyspinv${R}.0_nyslam1e-02_nyseps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      WSTEM="bert_seed${S}_num2000_val${V}_nyspinv${R}_lam1e-02_nyseps1e-8_cholesky_float32_signFalse_earlystopTrue_tmc500_poison10_ps${S}"
      SV_PKL=$OUT/shapley/$D/nystrom_pinv/${STEM}.pkl
      SEL_TXT=$OUT/data_selection/$D/nystrom_pinv/indices/${STEM}_indices.txt
      REM_TXT=$OUT/data_removing/$D/nystrom_pinv/predictions/${STEM}_predictions.txt
      WLD_TXT=$OUT/wrong_label_detection/$D/nystrom_pinv/predictions/${WSTEM}_detection.txt
      WLD_PKL=$OUT/wrong_label_detection/$D/nystrom_pinv/predictions/${WSTEM}_detection.pkl

      if [ -f "$SV_PKL" ]; then
        echo "  [skip] shapley  ($SV_PKL)"
      else
        $PY task_shapley.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d $R \
          --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
          --out_root $OUT
      fi
      if [ -f "$SEL_TXT" ]; then
        echo "  [skip] selection ($SEL_TXT)"
      else
        $PY task_data_selection.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d $R \
          --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
          --out_root $OUT
      fi
      if [ -f "$REM_TXT" ]; then
        echo "  [skip] removal  ($REM_TXT)"
      else
        $PY task_data_removal.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d $R \
          --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
          --out_root $OUT --num_train_removed_list $REM
      fi
      if [ -f "$WLD_TXT" ] && [ -f "$WLD_PKL" ]; then
        echo "  [skip] wrong-label ($WLD_TXT)"
      else
        $PY task_wrong_label_detection.py --config ntk_prompt --seed $S --dataset_name $D \
          --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d $R \
          --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --poison_pct 10 --tmc_iter 500 \
          --out_root $OUT
      fi
    done

  done
done
echo "[done] nystrom_pinv rank sweep (seeds:$SEEDS / datasets:$DATASETS)"

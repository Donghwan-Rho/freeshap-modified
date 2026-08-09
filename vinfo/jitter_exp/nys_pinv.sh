#!/bin/sh
# ============================================================
# pseudoinverse-Nystrom (nystrom_pinv) FULL grid — llama (ntk_llama) 버전.
#   grid_search.sh 의 nystrom eps/lambda grid 와 동일:
#     nystrom_lambda_ {1e-3,1e-2,1e-1,1}  x  nyseps {1e-8,1e-6,1e-4,1e-2}
#   seed 3개 x dataset 7개, for 문 구조도 grid_search.sh 와 동일. num_train=2000 고정.
#
# nystrom_pinv 만 실행 (shapley/selection/removal/wrong-label). inv/eigen 제외
#   (inv 기준선은 grid_search 가 만든 ./freeshap_res 것을 리포트가 재사용).
# 출력: ./jitter_exp/nys_pinv_res  (method_dir=nystrom_pinv, 파일명 _nyspinv…)
#   wrong-label 은 {config}_poison.yaml (ntk_llama_poison) 로 poison 데이터 로드.
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
    echo "################ nystrom_pinv llama  $D (val=$V)  seed=$S ################"

    for L in $NYS_LAMS; do
      for E in $NYS_EPSS; do
        echo "[nys_pinv] $D seed$S lam=$L eps=$E"

        # ---- 파일명 λ 태그: python 의 f"{lam:.0e}" 표기와 일치시킴 ----
        case "$L" in
          1e-3) LTAG="1e-03" ;;
          1e-2) LTAG="1e-02" ;;
          1e-1) LTAG="1e-01" ;;
          1)    LTAG="1e+00" ;;
          *)    LTAG="$L"    ;;
        esac

        # ---- 결과 파일 경로 (이미 있으면 해당 task 스킵) ----
        # shapley/selection/removal 파일명: _nyspinv20.0_nyslam..._invlam... (신형)
        # wrong-label 파일명: _nyspinv20_lam..._poison10_ps{seed} (wld 고유 형식)
        STEM="llama_seed${S}_num2000_val${V}_nyspinv20.0_nyslam${LTAG}_nyseps${E}_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
        WSTEM="llama_seed${S}_num2000_val${V}_nyspinv20_lam${LTAG}_nyseps${E}_cholesky_float32_signFalse_earlystopTrue_tmc500_poison10_ps${S}"
        SV_PKL=$OUT/shapley/$D/nystrom_pinv/${STEM}.pkl
        SEL_TXT=$OUT/data_selection/$D/nystrom_pinv/indices/${STEM}_indices.txt
        REM_TXT=$OUT/data_removing/$D/nystrom_pinv/predictions/${STEM}_predictions.txt
        WLD_TXT=$OUT/wrong_label_detection/$D/nystrom_pinv/predictions/${WSTEM}_detection.txt
        WLD_PKL=$OUT/wrong_label_detection/$D/nystrom_pinv/predictions/${WSTEM}_detection.pkl

        if [ -f "$SV_PKL" ]; then
          echo "  [skip] shapley  ($SV_PKL)"
        else
          $PY task_shapley.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d 20 \
            --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root $OUT
        fi
        if [ -f "$SEL_TXT" ]; then
          echo "  [skip] selection ($SEL_TXT)"
        else
          $PY task_data_selection.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d 20 \
            --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root $OUT
        fi
        if [ -f "$REM_TXT" ]; then
          echo "  [skip] removal  ($REM_TXT)"
        else
          $PY task_data_removal.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d 20 \
            --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --tmc_iter 500 --out_root $OUT \
            --num_train_removed_list $REM
        fi
        if [ -f "$WLD_TXT" ] && [ -f "$WLD_PKL" ]; then
          echo "  [skip] wrong-label ($WLD_TXT)"
        else
          $PY task_wrong_label_detection.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp 2000 --val_sample_num $V --approximate nystrom_pinv --nystrom_d 20 \
            --inv_lambda_ 1e-6 --nystrom_lambda_ $L --nyseps $E --poison_pct 10 --tmc_iter 500 \
            --out_root $OUT
        fi
      done
    done

  done
done
echo "[done] nystrom_pinv llama full grid (datasets: $DATASETS, seeds: $SEEDS)"

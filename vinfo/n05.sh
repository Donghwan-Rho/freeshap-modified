#!/bin/sh
# ============================================================
# llama (ntk_llama) inv + eigen rank sweep 러너.
#   각 (seed, dataset) 에서 inv 1셀 + eigen rank {1,5,10,15,20,25,30}% 셀을
#   shapley -> selection -> removal -> wrong-label 순서로 실행.
#   세팅: num_train=$N (기본 2000), eigen lam=1e-2 / eigeps=1e-8, inv lam=1e-6, tmc=500.
#   출력: ./freeshap_res (기본 out_root).
#
# 사용: 인자에서 숫자>=2024 -> seed, 그 외 문자 -> dataset (순서 무관).
#   sh n05.sh                           # n=2000, seed 2024 2025 2026 x 7개 데이터셋
#   sh n05.sh 2024 qqp mr               # seed 2024 x {qqp mr}
#
#   ★ n=1000 스케일 포인트 (speedup_er 의 n 별 열 재료 — selection 곡선만 필요):
#       N=1000 DO_REMOVAL=0 DO_WLD=0 sh n05.sh
#     NTK(num1000)는 7개 데이터셋 x 3 seed 전부 확보돼 있어 바로 돌릴 수 있다.
#     removal/wld 까지 원하면 DO_* 를 빼면 된다 (wld 는 poison TMC 라 훨씬 느림).
#
#   그 외 토글: DO_SHAPLEY / DO_SELECTION / DO_REMOVAL / DO_WLD (기본 전부 1)
#              RANKS="1 5 10"  처럼 rank 목록도 바꿀 수 있음
#
# 루프 구조: seed(바깥) -> dataset(안) -> [inv] -> rank.
# 모든 task 는 결과 파일 있으면 셸에서 스킵 (+python 내부 증분 로직 이중 안전망).
#   shapley: pkl / selection·removal: predictions txt / wld: detection txt+pkl.
# ============================================================
REM="$(seq -s' ' 0 99)"
N=${N:-2000}
RANKS=${RANKS:-"1 5 10 15 20 25 30"}
DO_SHAPLEY=${DO_SHAPLEY:-1}
DO_SELECTION=${DO_SELECTION:-1}
DO_REMOVAL=${DO_REMOVAL:-1}
DO_WLD=${DO_WLD:-1}

# ---- 인자 분류: 2024 이상 숫자=seed, 그 외=dataset ----
SEEDS=""; DATASETS=""
for a in "$@"; do
  case "$a" in
    [0-9]*) SEEDS="$SEEDS $a" ;;
    *)      DATASETS="$DATASETS $a" ;;
  esac
done
SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-mnli sst2 qqp rte mrpc mr ag_news}"
echo "[cfg] num_train=$N  seeds:$SEEDS  datasets:$DATASETS"
echo "[cfg] ranks:$RANKS  shapley=$DO_SHAPLEY selection=$DO_SELECTION removal=$DO_REMOVAL wld=$DO_WLD"

for S in $SEEDS; do
  for D in $DATASETS; do

    # ---- dataset 별 val_sample_num (grid_search.sh 와 동일) ----
    case "$D" in
      sst2) V=872  ;;
      mrpc) V=408  ;;
      rte)  V=277  ;;
      *)    V=1000 ;;
    esac
    echo "################ llama inv+eigen  $D (n=$N, val=$V)  seed=$S ################"

    # ---- NTK 캐시 확인 (없으면 task_shapley 가 llama 를 로드해 새로 만든다 — 느림) ----
    NTK=./freeshap_res/ntk/$D/llama_seed${S}_num${N}_val${V}_signFalse.pkl
    [ -f "$NTK" ] || echo "  [note] NTK 캐시 없음 ($NTK) — shapley 단계에서 새로 생성됨"

    # ============ inv ============
    STEM="llama_seed${S}_num${N}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500"
    WSTEM="llama_seed${S}_num${N}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500_poison10_ps${S}"
    SV_PKL=./freeshap_res/shapley/$D/inv/${STEM}.pkl
    SEL_TXT=./freeshap_res/data_selection/$D/inv/predictions/${STEM}_predictions.txt
    REM_TXT=./freeshap_res/data_removing/$D/inv/predictions/${STEM}_predictions.txt
    WLD_TXT=./freeshap_res/wrong_label_detection/$D/inv/predictions/${WSTEM}_detection.txt
    WLD_PKL=./freeshap_res/wrong_label_detection/$D/inv/predictions/${WSTEM}_detection.pkl

    if [ "$DO_SHAPLEY" = "1" ]; then
      if [ -f "$SV_PKL" ]; then echo "  [skip] inv shapley  ($SV_PKL)"
      else
        python task_shapley.py --config ntk_llama --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
      fi
    fi
    if [ "$DO_SELECTION" = "1" ]; then
      if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] inv selection"
      elif [ -f "$SEL_TXT" ]; then echo "  [skip] inv selection ($SEL_TXT)"
      else
        python task_data_selection.py --config ntk_llama --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
      fi
    fi
    if [ "$DO_REMOVAL" = "1" ]; then
      if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] inv removal"
      elif [ -f "$REM_TXT" ]; then echo "  [skip] inv removal  ($REM_TXT)"
      else
        python task_data_removal.py --config ntk_llama --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500 \
          --num_train_removed_list $REM
      fi
    fi
    if [ "$DO_WLD" = "1" ]; then
      if [ -f "$WLD_TXT" ] && [ -f "$WLD_PKL" ]; then echo "  [skip] inv wrong-label ($WLD_TXT)"
      else
        python task_wrong_label_detection.py --config ntk_llama --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 \
          --poison_pct 10 --tmc_iter 500
      fi
    fi

    # ============ eigen rank sweep ============
    for R in $RANKS; do
      echo "[eigen] $D seed$S rank=${R}%"

      # shapley/selection/removal 파일명: _eig{R}.0_eiglam..._eigeps..._invlam... (신형)
      # wrong-label 파일명: _eig{R}_lam..._eigeps..._poison10_ps{seed} (wld 고유 형식)
      STEM="llama_seed${S}_num${N}_val${V}_eig${R}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      WSTEM="llama_seed${S}_num${N}_val${V}_eig${R}_lam1e-02_eigeps1e-8_cholesky_float32_signFalse_earlystopTrue_tmc500_poison10_ps${S}"
      SV_PKL=./freeshap_res/shapley/$D/eigen/${STEM}.pkl
      SEL_TXT=./freeshap_res/data_selection/$D/eigen/predictions/${STEM}_predictions.txt
      REM_TXT=./freeshap_res/data_removing/$D/eigen/predictions/${STEM}_predictions.txt
      WLD_TXT=./freeshap_res/wrong_label_detection/$D/eigen/predictions/${WSTEM}_detection.txt
      WLD_PKL=./freeshap_res/wrong_label_detection/$D/eigen/predictions/${WSTEM}_detection.pkl

      if [ "$DO_SHAPLEY" = "1" ]; then
        if [ -f "$SV_PKL" ]; then echo "  [skip] shapley  ($SV_PKL)"
        else
          python task_shapley.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500
        fi
      fi
      if [ "$DO_SELECTION" = "1" ]; then
        if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] selection"
        elif [ -f "$SEL_TXT" ]; then echo "  [skip] selection ($SEL_TXT)"
        else
          python task_data_selection.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500
        fi
      fi
      if [ "$DO_REMOVAL" = "1" ]; then
        if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] removal"
        elif [ -f "$REM_TXT" ]; then echo "  [skip] removal  ($REM_TXT)"
        else
          python task_data_removal.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500 \
            --num_train_removed_list $REM
        fi
      fi
      if [ "$DO_WLD" = "1" ]; then
        if [ -f "$WLD_TXT" ] && [ -f "$WLD_PKL" ]; then echo "  [skip] wrong-label ($WLD_TXT)"
        else
          python task_wrong_label_detection.py --config ntk_llama --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct 10 --tmc_iter 500
        fi
      fi
    done

  done
done
echo "[done] llama inv+eigen (n=$N / seeds:$SEEDS / datasets:$DATASETS)"

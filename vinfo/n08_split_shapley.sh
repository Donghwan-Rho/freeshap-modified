#!/bin/sh
# ============================================================
# B안 0단계 (RTE / MRPC 전용): train 을 쪼갠 새 n 으로 NTK + Shapley 를 만든다.
#
#   RTE/MRPC 는 train split 전체(2490 / 3668)를 쓰고 있어 held-out 을 뗄 자리가 없었다.
#   heldout_common.FIXED_SPLIT 대로 held-out 을 seed 42 로 **먼저** 고정하고 train = 나머지 전부:
#       rte  : train 1500 + held-out 990   (val 277 그대로)
#       mrpc : train 3000 + held-out 668   (val 408 그대로)
#   train 도 seed 와 무관하게 고정되고 seed 는 TMC 순열만 바꾼다 (예전 RTE/MRPC 와 같은 구조).
#   n 이 바뀌므로 NTK(train+val) 와 Shapley(inv + eigen rank sweep) 를 새로 계산한다.
#   기존 num2490 / num3668 결과는 그대로 둔다 (in-sample 그림·removal·wld 가 쓴다).
#
#   순서:  n08 (여기: NTK + Shapley)  ->  n07 (held-out 블록)  ->  n06 (selection + a0)
#
# 사용: 인자에서 숫자 -> seed, bert/llama -> 모델, 그 외 -> dataset (기본 rte mrpc).
#   CUDA_VISIBLE_DEVICES=<GPU> sh n08_split_shapley.sh              # bert, rte+mrpc, seed 3개
#   CUDA_VISIBLE_DEVICES=<GPU> sh n08_split_shapley.sh llama rte    # llama, rte 만
#   CUDA_VISIBLE_DEVICES=<GPU> sh n08_split_shapley.sh 2024         # seed 하나만
#   DRY=1 sh n08_split_shapley.sh                                    # 명령만 출력
#   토글: DO_NTK / DO_INV / DO_EIGEN (기본 1), RANKS="1 5 10" 로 rank 변경
#
# 결과 파일이 있으면 스킵 (resume 안전).
# ============================================================
MODEL=${MODEL:-bert}
SEEDS=""; DATASETS=""
for a in "$@"; do
  case "$a" in
    [0-9]*)             SEEDS="$SEEDS $a" ;;
    bert|llama|resnet)  MODEL="$a" ;;
    *)                  DATASETS="$DATASETS $a" ;;
  esac
done
SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-rte mrpc}"
RANKS=${RANKS:-"1 5 10 15 20 25 30"}
DO_NTK=${DO_NTK:-1}; DO_INV=${DO_INV:-1}; DO_EIGEN=${DO_EIGEN:-1}

case "$MODEL" in
  bert)  CFG=ntk_prompt ;;
  llama) CFG=ntk_llama ;;
  *) echo "[error] n08 은 bert / llama 만 (받은 값: $MODEL)"; exit 1 ;;
esac
run() { if [ "${DRY:-0}" = "1" ]; then echo "  \$ $*"; else "$@"; fi; }

echo "[cfg] MODEL=$MODEL ($CFG)  seeds:$SEEDS  datasets:$DATASETS  ranks:$RANKS"

for S in $SEEDS; do
  for D in $DATASETS; do
    case "$D" in
      rte)  N=1500 ; V=277 ;;
      mrpc) N=3000 ; V=408 ;;
      *) echo "  [skip] $D: FIXED_SPLIT 데이터셋(rte/mrpc)만 대상"; continue ;;
    esac
    echo "################ split NTK+Shapley: $MODEL $D (train=$N, val=$V) seed=$S ################"

    # ---- 1) NTK (train+val), held-out 을 뺀 후보에서 ----
    NTK=./freeshap_res/ntk/$D/${MODEL}_seed${S}_num${N}_val${V}_signFalse.pkl
    if [ "$DO_NTK" = "1" ]; then
      if [ -f "$NTK" ]; then echo "  [skip] ntk ($NTK)"
      else
        echo "  [run] ntk (--exclude_heldout)"
        run python task_ntk.py --config $CFG --dataset_name $D --seed $S \
          --num_train_dp $N --val_sample_num $V --exclude_heldout
      fi
    fi
    if [ ! -f "$NTK" ] && [ "${DRY:-0}" != "1" ]; then echo "  [no-ntk] $NTK 없음 -> shapley 건너뜀"; continue; fi

    # ---- 2) Shapley inv ----
    STEM="${MODEL}_seed${S}_num${N}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500"
    if [ "$DO_INV" = "1" ]; then
      if [ -f "./freeshap_res/shapley/$D/inv/${STEM}.pkl" ]; then echo "  [skip] inv shapley"
      else
        echo "  [run] inv shapley"
        run python task_shapley.py --config $CFG --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 --tmc_iter 500
      fi
    fi

    # ---- 3) Shapley eigen rank sweep ----
    if [ "$DO_EIGEN" = "1" ]; then
      for R in $RANKS; do
        ESTEM="${MODEL}_seed${S}_num${N}_val${V}_eig${R}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
        if [ -f "./freeshap_res/shapley/$D/eigen/${ESTEM}.pkl" ]; then echo "  [skip] eigen rank=${R}%"
        else
          echo "  [run] eigen shapley rank=${R}%"
          run python task_shapley.py --config $CFG --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500
        fi
      done
    fi
  done
done
echo "[done] split NTK+Shapley ($MODEL / seeds:$SEEDS / datasets:$DATASETS)"
echo "다음: sh n07_heldout_ntk.sh $MODEL $DATASETS  ->  sh n06_selection_eval.sh $MODEL $DATASETS"

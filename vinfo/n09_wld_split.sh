#!/bin/sh
# ============================================================
# RTE / MRPC 의 새 train 분할(rte 1500, mrpc 3000)로 wrong-label detection 을 다시 돌린다.
#
#   held-out 프로토콜에서 RTE/MRPC 는 train 을 쪼개 썼으므로(heldout_common.FIXED_SPLIT),
#   3x7 그림의 wld 행도 같은 train 으로 맞추려면 그 n 에서 wld 를 새로 돌려야 한다.
#   wld 는 train 라벨 10% 를 오염시켜 TMC 를 다시 돌리는 실험이라 seed 당 시간이 꽤 든다.
#
#   출력: $OUT_ROOT (기본 ./freeshap_res_wld)
#     $OUT_ROOT/wrong_label_detection/{ds}/{inv|eigen}/predictions/*_detection.{txt,pkl}
#     $OUT_ROOT/shapley/{ds}/{inv|eigen}/*_poison10_ps{seed}.pkl        (오염 라벨의 SV)
#   NTK 캐시는 task_wrong_label_detection.py 가 항상 ./freeshap_res/ntk 에서 읽는다.
#     -> 없으면 여기서 task_ntk.py --exclude_heldout 로 먼저 만든다 (held-out 을 뺀 고정 train).
#        커널은 라벨과 무관하므로 selection 과 같은 캐시를 그대로 공유한다.
#
# 사용: 인자에서 숫자 -> seed, bert/llama -> 모델, 그 외 -> dataset (기본 rte mrpc).
#   CUDA_VISIBLE_DEVICES=<GPU> sh n09_wld_split.sh                 # bert, rte+mrpc, seed 3개
#   CUDA_VISIBLE_DEVICES=<GPU> sh n09_wld_split.sh llama mrpc      # llama, mrpc 만
#   CUDA_VISIBLE_DEVICES=<GPU> sh n09_wld_split.sh 2024            # seed 하나만
#   OUT_ROOT=./freeshap_res sh n09_wld_split.sh                     # 출력 폴더 바꾸기
#   DRY=1 sh n09_wld_split.sh                                       # 명령만 출력
#   토글: DO_INV / DO_EIGEN (기본 1), RANKS="1 5 10" 로 rank 변경
#
# 결과 파일(txt+pkl)이 있으면 스킵 (resume 안전).
# ============================================================
MODEL=${MODEL:-bert}
OUT_ROOT=${OUT_ROOT:-./freeshap_res}
RANKS=${RANKS:-"1 5 10 15 20 25 30"}
DO_INV=${DO_INV:-1}; DO_EIGEN=${DO_EIGEN:-1}
POISON=10

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

case "$MODEL" in
  bert)  CFG=ntk_prompt ;;
  llama) CFG=ntk_llama ;;
  *) echo "[error] n09 는 bert / llama 만 (받은 값: $MODEL)"; exit 1 ;;
esac
run() { if [ "${DRY:-0}" = "1" ]; then echo "  \$ $*"; else "$@"; fi; }

mkdir -p "$OUT_ROOT"
echo "[cfg] MODEL=$MODEL ($CFG)  seeds:$SEEDS  datasets:$DATASETS  ranks:$RANKS  poison=${POISON}%  -> $OUT_ROOT"

for S in $SEEDS; do
  for D in $DATASETS; do
    case "$D" in
      rte)  N=1500 ; V=277 ;;
      mrpc) N=3000 ; V=408 ;;
      *) echo "  [skip] $D: FIXED_SPLIT 데이터셋(rte/mrpc)만 대상"; continue ;;
    esac
    echo "################ wld (split): $MODEL $D (train=$N, val=$V) seed=$S ################"

    # ---- 0) NTK 캐시 (train+val), held-out 을 뺀 고정 train ----
    NTK=./freeshap_res/ntk/$D/${MODEL}_seed${S}_num${N}_val${V}_signFalse.pkl
    if [ ! -f "$NTK" ]; then
      echo "  [run] ntk (--exclude_heldout): $NTK"
      run python task_ntk.py --config $CFG --dataset_name $D --seed $S \
        --num_train_dp $N --val_sample_num $V --exclude_heldout
      if [ ! -f "$NTK" ] && [ "${DRY:-0}" != "1" ]; then echo "  [fail] ntk 생성 실패 -> 건너뜀"; continue; fi
    else
      echo "  [skip] ntk ($NTK)"
    fi

    # ---- 1) inv ----
    if [ "$DO_INV" = "1" ]; then
      WSTEM="${MODEL}_seed${S}_num${N}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500_poison${POISON}_ps${S}"
      TXT=$OUT_ROOT/wrong_label_detection/$D/inv/predictions/${WSTEM}_detection.txt
      PKL=$OUT_ROOT/wrong_label_detection/$D/inv/predictions/${WSTEM}_detection.pkl
      if [ -f "$TXT" ] && [ -f "$PKL" ]; then echo "  [skip] inv wld ($TXT)"
      else
        echo "  [run] inv wld"
        run python task_wrong_label_detection.py --config $CFG --seed $S --dataset_name $D \
          --num_train_dp $N --val_sample_num $V --approximate inv --inv_lambda_ 1e-6 \
          --poison_pct $POISON --tmc_iter 500 --out_root $OUT_ROOT
      fi
    fi

    # ---- 2) eigen rank sweep ----
    if [ "$DO_EIGEN" = "1" ]; then
      for R in $RANKS; do
        WSTEM="${MODEL}_seed${S}_num${N}_val${V}_eig${R}_lam1e-02_eigeps1e-8_cholesky_float32_signFalse_earlystopTrue_tmc500_poison${POISON}_ps${S}"
        TXT=$OUT_ROOT/wrong_label_detection/$D/eigen/predictions/${WSTEM}_detection.txt
        PKL=$OUT_ROOT/wrong_label_detection/$D/eigen/predictions/${WSTEM}_detection.pkl
        if [ -f "$TXT" ] && [ -f "$PKL" ]; then echo "  [skip] eigen wld rank=${R}%"
        else
          echo "  [run] eigen wld rank=${R}%"
          run python task_wrong_label_detection.py --config $CFG --seed $S --dataset_name $D \
            --num_train_dp $N --val_sample_num $V --approximate eigen --eigen_rank $R \
            --eigen_lambda_ 1e-2 --eigeps 1e-8 --poison_pct $POISON --tmc_iter 500 --out_root $OUT_ROOT
        fi
      done
    fi
  done
done
echo "[done] wld split ($MODEL / seeds:$SEEDS / datasets:$DATASETS) -> $OUT_ROOT"

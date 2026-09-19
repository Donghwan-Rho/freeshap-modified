#!/bin/sh
# ============================================================
# B안 2단계: data selection 을 held-out 평가 집합에서 다시 측정.
#   val 로 점수를 매기고(= Shapley 그대로 재사용), train/test 에서 뗀 고정 집합(seed 42)에서 평가한다.
#   FreeShap 논문 App. F.2 의 data selection 프로토콜과 같은 구성이다.
#   기존 in-sample 결과는 freeshap_res/data_selection_insample/ 에 보관돼 있다.
#
#   전제: freeshap_res/ntk_heldout/ 에 그 (모델,데이터셋,seed) 블록이 있어야 한다
#         -> 1단계  sh n07_heldout_ntk.sh  를 먼저 돌릴 것.
#   RTE/MRPC 는 train 을 전부 써서 held-out 을 뗄 수 없다 -> 자동 스킵.
#
#   Shapley pkl 은 재사용한다 (TMC 재계산 없음). 돌리는 것은 selection 예측과 a0 뿐이다.
#   removal / wrong-label 은 프로토콜이 그대로라 손대지 않는다.
#
# 사용: 인자에서 숫자 -> seed, 그 외 문자 -> dataset (순서 무관).
#   CUDA_VISIBLE_DEVICES=<빈GPU> sh n06_selection_eval.sh                  # bert 5개 데이터셋 x 3 seed
#   CUDA_VISIBLE_DEVICES=<빈GPU> sh n06_selection_eval.sh 2024 sst2        # 하나만
#   MODEL=resnet CUDA_VISIBLE_DEVICES=<빈GPU> sh n06_selection_eval.sh     # CIFAR-10
#   MODEL=llama  CUDA_VISIBLE_DEVICES=<빈GPU> sh n06_selection_eval.sh     # (나중에)
#   DRY=1 sh n06_selection_eval.sh                                          # 명령만 출력
#
#   그 외 토글: DO_SELECTION / DO_A0 / DO_REMOVAL (기본 1), RANKS="1 5 10" 로 rank 목록 변경
#   removal 도 같은 SV·같은 held-out 블록으로 돈다 (결과 -> freeshap_res/data_removing/,
#   예전 in-sample 결과는 data_removing_insample/ 에 보관).
#
# 결과 -> freeshap_res/data_selection/ . 이미 있으면 스킵 (resume 안전).
# 그림은 SELECTION_DIR 로 폴더를 고른다 (기본 data_selection).
MODEL=${MODEL:-bert}
# ---- 인자 분류: 숫자=seed, bert/llama/resnet=모델, 그 외=dataset ----
#   (MODEL=llama 환경변수 대신 'sh n06_selection_eval.sh llama' 처럼 써도 되게)
SEEDS=""; DATASETS=""
for a in "$@"; do
  case "$a" in
    [0-9]*)             SEEDS="$SEEDS $a" ;;
    bert|llama|resnet)  MODEL="$a" ;;
    *)                  DATASETS="$DATASETS $a" ;;
  esac
done
N=${N:-5000}
RANKS=${RANKS:-"1 5 10 15 20 25 30"}
DO_SELECTION=${DO_SELECTION:-1}
DO_A0=${DO_A0:-1}
DO_REMOVAL=${DO_REMOVAL:-1}
REM="$(seq -s' ' 0 99)"          # removal 비율 목록 (0=제거 없음 baseline ~ 99)

case "$MODEL" in
  bert)   CFG=ntk_prompt ; SCRIPT=task_data_selection.py               ; RSCRIPT=task_data_removal.py               ; DEF_DS="sst2 mnli ag_news mr qqp rte mrpc" ;;
  llama)  CFG=ntk_llama  ; SCRIPT=task_data_selection.py               ; RSCRIPT=task_data_removal.py               ; DEF_DS="sst2 mnli ag_news mr qqp rte mrpc" ;;
  resnet) CFG=ntk_vision ; SCRIPT=vision/task_data_selection_vision.py ; RSCRIPT=vision/task_data_removal_vision.py ; DEF_DS="cifar10" ;;
  *) echo "[error] MODEL 은 bert / llama / resnet (받은 값: $MODEL)"; exit 1 ;;
esac
FLAG="--heldout"
OUTBASE=./freeshap_res/data_selection
REMBASE=./freeshap_res/data_removing

SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-$DEF_DS}"

run() { if [ "${DRY:-0}" = "1" ]; then echo "  \$ $*"; else "$@"; fi; }

echo "[cfg] MODEL=$MODEL ($CFG)  num_train=$N  seeds:$SEEDS  datasets:$DATASETS"
echo "[cfg] ranks:$RANKS  selection=$DO_SELECTION a0=$DO_A0  -> $OUTBASE"

for S in $SEEDS; do
  for D in $DATASETS; do

    # ---- dataset 별 val_sample_num / num_train (full-size 규약) ----
    case "$D" in
      sst2) V=872  ; ND=$N    ;;
      mrpc) V=408  ; ND=3000  ;;   # FIXED_SPLIT: train 3000 + held-out 668 (예전 in-sample 은 3668)
      rte)  V=277  ; ND=1500  ;;   # FIXED_SPLIT: train 1500 + held-out 990 (예전 in-sample 은 2490)
      *)    V=1000 ; ND=$N    ;;
    esac

    # ---- held-out NTK 블록이 없으면 이 (dataset, seed) 통째로 건너뛴다 ----
    #   (없는 채로 돌리면 작업마다 모델을 로드한 뒤에야 FileNotFoundError 로 죽어 시간을 버린다)
    case "$D" in
      mr)   HO_TAG="ho1066ts42" ;;    # MR 은 공식 test split 1066 개
      rte)  HO_TAG="ho990s42"   ;;
      mrpc) HO_TAG="ho668s42"   ;;
      *)    HO_TAG="ho1000s42"  ;;
    esac
    HO_NTK=./freeshap_res/ntk_heldout/$D/${MODEL}_seed${S}_num${ND}_${HO_TAG}_signFalse.pkl
    if [ ! -f "$HO_NTK" ]; then
      echo "  [no-ntk] $MODEL $D seed$S: held-out 블록 없음 -> 먼저  sh n07_heldout_ntk.sh $MODEL $D $S"
      continue
    fi

    echo "################ $MODEL $D (n=$ND, val=$V) seed=$S  [held-out] ################"

    # ============ inv (FreeShap) ============
    STEM="${MODEL}_seed${S}_num${ND}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500"
    SV_PKL=./freeshap_res/shapley/$D/inv/${STEM}.pkl
    SEL_TXT=$OUTBASE/$D/inv/predictions/${STEM}_predictions.txt
    A0_TXT=$OUTBASE/$D/inv/base_accuracy/${STEM}_base.txt

    if [ "$DO_SELECTION" = "1" ]; then
      if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] inv selection ($SV_PKL)"
      elif [ -f "$SEL_TXT" ]; then echo "  [skip] inv selection ($SEL_TXT)"
      else
        echo "  [run] inv selection"
        run python $SCRIPT $FLAG --config $CFG --seed $S --dataset_name $D \
          --num_train_dp $ND --val_sample_num $V --approximate inv \
          --inv_lambda_ 1e-6 --tmc_iter 500 --out_root ./freeshap_res
      fi
    fi

    # removal: 같은 SV·같은 held-out 블록으로 상위 k% 제거 곡선 (0~99%)
    if [ "$DO_REMOVAL" = "1" ]; then
      REM_TXT=$REMBASE/$D/inv/predictions/${STEM}_predictions.txt
      if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] inv removal"
      elif [ -f "$REM_TXT" ]; then echo "  [skip] inv removal ($REM_TXT)"
      else
        echo "  [run] inv removal"
        run python $RSCRIPT $FLAG --config $CFG --seed $S --dataset_name $D \
          --num_train_dp $ND --val_sample_num $V --approximate inv \
          --inv_lambda_ 1e-6 --tmc_iter 500 --out_root ./freeshap_res --num_train_removed_list $REM
      fi
    fi

    # a0 (0% 선택 기준값) — vision 은 selection 안에서 같이 저장되므로 NLP 만.
    if [ "$DO_A0" = "1" ] && [ "$MODEL" != "resnet" ]; then
      if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] inv a0"
      elif [ -f "$A0_TXT" ]; then echo "  [skip] inv a0 ($A0_TXT)"
      else
        echo "  [run] inv base accuracy"
        run python task_base_accuracy.py $FLAG --config $CFG --seed $S --dataset_name $D \
          --num_train_dp $ND --val_sample_num $V --tmc_iter 500 --inv_lambda_ 1e-6
      fi
    fi

    # ============ eigen rank sweep (selection + removal) ============
    for R in $RANKS; do
      STEM="${MODEL}_seed${S}_num${ND}_val${V}_eig${R}.0_eiglam1e-02_eigeps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
      SV_PKL=./freeshap_res/shapley/$D/eigen/${STEM}.pkl
      SEL_TXT=$OUTBASE/$D/eigen/predictions/${STEM}_predictions.txt
      REM_TXT=$REMBASE/$D/eigen/predictions/${STEM}_predictions.txt
      if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] eigen rank=${R}% ($SV_PKL)"; continue; fi
      if [ "$DO_SELECTION" = "1" ]; then
        if [ -f "$SEL_TXT" ]; then echo "  [skip] eigen selection rank=${R}%"
        else
          echo "  [run] eigen selection rank=${R}%"
          run python $SCRIPT $FLAG --config $CFG --seed $S --dataset_name $D \
            --num_train_dp $ND --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500 \
            --out_root ./freeshap_res
        fi
      fi
      if [ "$DO_REMOVAL" = "1" ]; then
        if [ -f "$REM_TXT" ]; then echo "  [skip] eigen removal rank=${R}%"
        else
          echo "  [run] eigen removal rank=${R}%"
          run python $RSCRIPT $FLAG --config $CFG --seed $S --dataset_name $D \
            --num_train_dp $ND --val_sample_num $V --approximate eigen --eigen_rank $R \
            --inv_lambda_ 1e-6 --eigen_lambda_ 1e-2 --eigeps 1e-8 --tmc_iter 500 \
            --out_root ./freeshap_res --num_train_removed_list $REM
        fi
      fi
    done

  done
done
echo "[done] held-out $MODEL (n=$N / seeds:$SEEDS / datasets:$DATASETS)"

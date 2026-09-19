#!/bin/sh
# ============================================================
# B안 1단계: held-out 평가 집합의 eNTK 블록 계산.
#
#   기존  freeshap_res/ntk/{ds}/{model}_seed{S}_num{N}_val{V}_signFalse.pkl        (N+V, N)
#   생성  freeshap_res/ntk_heldout/{ds}/{model}_seed{S}_num{N}_ho{H}s42_signFalse.pkl (N+H, N)
#
#   held-out 은 seed 와 무관하게 고정한다 (heldout_seed=42).
#     - 3개 seed 의 train subset 합집합을 제외하고 뽑으므로 어느 seed 에서도 학습에 안 쓰인 점들이다.
#     - MR 만 예외: train 풀(8530)에 자리가 없어 공식 test split(1066) 을 쓴다.
#     - RTE/MRPC 는 train 을 전부 써서 불가 -> 자동 스킵 (train 을 쪼개려면 SV 재계산 필요).
#   앞 N행(train x train)은 기존 캐시 값으로 덮어써 Shapley 계산에 쓴 커널과 일치시킨다.
#
#   ※ 비용: compute_ntk 가 train+heldout 의 gradient 를 모두 구하므로
#     (모델, 데이터셋, seed) 당 6000행 짜리 NTK 작업 1회 ≈ 기존 task_ntk 1회.
#     Shapley(TMC)는 재계산하지 않는다.
#
# 사용: 인자에서 숫자 -> seed, 그 외 문자 -> dataset (순서 무관).
#   CUDA_VISIBLE_DEVICES=<빈GPU> sh n07_heldout_ntk.sh              # bert 5개 데이터셋 x 3 seed
#   CUDA_VISIBLE_DEVICES=<빈GPU> sh n07_heldout_ntk.sh 2024 sst2    # 하나만
#   MODEL=resnet CUDA_VISIBLE_DEVICES=<빈GPU> sh n07_heldout_ntk.sh # CIFAR-10
#   MODEL=llama  CUDA_VISIBLE_DEVICES=<빈GPU> sh n07_heldout_ntk.sh # (나중에)
#   DRY=1 sh n07_heldout_ntk.sh                                      # 명령만 출력
#
# 끝나면:  rte/mrpc 는 n08_split_shapley.sh 로 Shapley 를 만든 뒤  ->  n06_selection_eval.sh
# ============================================================
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

case "$MODEL" in
  bert)   CFG=ntk_prompt ; SCRIPT=task_ntk_heldout.py               ; DEF_DS="sst2 mnli ag_news mr qqp rte mrpc" ;;
  llama)  CFG=ntk_llama  ; SCRIPT=task_ntk_heldout.py               ; DEF_DS="sst2 mnli ag_news mr qqp rte mrpc" ;;
  resnet) CFG=ntk_vision ; SCRIPT=vision/task_ntk_heldout_vision.py ; DEF_DS="cifar10" ;;
  *) echo "[error] MODEL 은 bert / llama / resnet (받은 값: $MODEL)"; exit 1 ;;
esac

SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-$DEF_DS}"

run() { if [ "${DRY:-0}" = "1" ]; then echo "  \$ $*"; else "$@"; fi; }

echo "[cfg] MODEL=$MODEL ($CFG)  n=$N  seeds:$SEEDS  datasets:$DATASETS"

for S in $SEEDS; do
  for D in $DATASETS; do
    # ---- 데이터셋별 val / n / held-out 태그 ----
    #   RTE/MRPC 는 train 을 전부 쓰던 데이터셋이라 held-out 을 먼저 고정하고 train = 나머지
    #   (heldout_common.FIXED_SPLIT: rte 1500+990, mrpc 3000+668). 그 n 의 NTK 캐시가
    #   task_ntk.py --exclude_heldout 로 먼저 만들어져 있어야 한다 (n08_split_shapley.sh).
    case "$D" in
      sst2) V=872  ; ND=$N   ; TAG="ho1000s42"  ;;
      mr)   V=1000 ; ND=$N   ; TAG="ho1066ts42" ;;   # 공식 test split 1066 개
      rte)  V=277  ; ND=1500 ; TAG="ho990s42"   ;;
      mrpc) V=408  ; ND=3000 ; TAG="ho668s42"   ;;
      *)    V=1000 ; ND=$N   ; TAG="ho1000s42"  ;;
    esac
    OUT=./freeshap_res/ntk_heldout/$D/${MODEL}_seed${S}_num${ND}_${TAG}_signFalse.pkl
    if [ -f "$OUT" ]; then
      echo "[skip] $MODEL $D seed$S  ($OUT)"
      continue
    fi

    # ---- 기본 NTK 캐시(train+val) 확인: held-out 블록은 이 캐시의 train subset 을 그대로 쓴다 ----
    #   rte/mrpc 는 새 n(FIXED_SPLIT) 이라 캐시가 없을 수 있다 -> 여기서 task_ntk --exclude_heldout 로 만든다.
    #   그 외 데이터셋은 원래 캠페인의 캐시가 있어야 한다 (held-out 정의에 3 seed 의 train 합집합이 필요).
    BASE_NTK=./freeshap_res/ntk/$D/${MODEL}_seed${S}_num${ND}_val${V}_signFalse.pkl
    if [ ! -f "$BASE_NTK" ]; then
      case "$D" in
        rte|mrpc)
          echo "  [run] base ntk 먼저 생성 (--exclude_heldout): $BASE_NTK"
          run python task_ntk.py --config $CFG --dataset_name $D --seed $S \
            --num_train_dp $ND --val_sample_num $V --exclude_heldout
          if [ ! -f "$BASE_NTK" ] && [ "${DRY:-0}" != "1" ]; then
            echo "  [fail] base ntk 생성 실패 -> $D seed$S 건너뜀"; continue
          fi ;;
        *)
          echo "  [no-base-ntk] $BASE_NTK 없음 -> 원래 캠페인의 task_ntk 결과(3 seed 전부)가 먼저 필요"; continue ;;
      esac
    fi
    echo "################ held-out NTK: $MODEL $D (n=$ND, val=$V) seed=$S ################"
    run python $SCRIPT --config $CFG --dataset_name $D --seed $S \
      --num_train_dp $ND --val_sample_num $V --out_root ./freeshap_res
  done
done
echo "[done] held-out NTK ($MODEL / seeds:$SEEDS / datasets:$DATASETS)"
echo "다음: sh n08_split_shapley.sh $MODEL   (rte/mrpc 의 Shapley, NTK 는 건너뜀)  ->  sh n06_selection_eval.sh $MODEL"

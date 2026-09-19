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
# 끝나면 2단계:  MODE=heldout sh n06_selection_eval.sh
# ============================================================
MODEL=${MODEL:-bert}
N=${N:-5000}

case "$MODEL" in
  bert)   CFG=ntk_prompt ; SCRIPT=task_ntk_heldout.py               ; DEF_DS="sst2 mnli ag_news mr qqp" ;;
  llama)  CFG=ntk_llama  ; SCRIPT=task_ntk_heldout.py               ; DEF_DS="sst2 mnli ag_news mr qqp" ;;
  resnet) CFG=ntk_vision ; SCRIPT=vision/task_ntk_heldout_vision.py ; DEF_DS="cifar10" ;;
  *) echo "[error] MODEL 은 bert / llama / resnet (받은 값: $MODEL)"; exit 1 ;;
esac

SEEDS=""; DATASETS=""
for a in "$@"; do
  case "$a" in
    [0-9]*) SEEDS="$SEEDS $a" ;;
    *)      DATASETS="$DATASETS $a" ;;
  esac
done
SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-$DEF_DS}"

run() { if [ "${DRY:-0}" = "1" ]; then echo "  \$ $*"; else "$@"; fi; }

echo "[cfg] MODEL=$MODEL ($CFG)  n=$N  seeds:$SEEDS  datasets:$DATASETS"

for S in $SEEDS; do
  for D in $DATASETS; do
    case "$D" in
      rte|mrpc) echo "[skip] $D: train split 전체를 써서 held-out 을 뗄 수 없음"; continue ;;
    esac
    case "$D" in
      sst2) V=872 ;;
      *)    V=1000 ;;
    esac
    # held-out 태그: MR 만 test split 1066 개, 나머지는 train 에서 1000 개
    case "$D" in
      mr) TAG="ho1066ts42" ;;
      *)  TAG="ho1000s42"  ;;
    esac
    OUT=./freeshap_res/ntk_heldout/$D/${MODEL}_seed${S}_num${N}_${TAG}_signFalse.pkl
    if [ -f "$OUT" ]; then
      echo "[skip] $MODEL $D seed$S  ($OUT)"
      continue
    fi
    echo "################ held-out NTK: $MODEL $D (n=$N, val=$V) seed=$S ################"
    run python $SCRIPT --config $CFG --dataset_name $D --seed $S \
      --num_train_dp $N --val_sample_num $V --out_root ./freeshap_res
  done
done
echo "[done] held-out NTK ($MODEL / seeds:$SEEDS / datasets:$DATASETS)"
echo "다음: MODE=heldout sh n06_selection_eval.sh   (MODEL=$MODEL)"

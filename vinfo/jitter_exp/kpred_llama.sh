#!/bin/sh
# ============================================================
# llama kernel_prediction npz 러너 — report p3/p4 (kernel/kpred 상대오차) 재료.
#   compute_kernel_pred_error.py --model llama 를 데이터셋별로 실행.
#   세팅: rank {1,5,10,15,20,25,30}%, λ=1e-2, ε=1e-8, methods eigen/pinv/lev.
#   npz 이미 있는 셀은 내부에서 자동 스킵 (증분 안전). NTK 캐시(num2000)만 필요.
#   llama 모델 로드가 (dataset, seed)마다 일어나므로 llama 있는 GPU 서버에서 실행.
#
# 출력: 각 method root 의 kernel_prediction/{ds}/... (report 가 읽는 위치 그대로)
#   -> 끝나면 jitter_exp/kernel_prediction 관련 폴더들을 메인 서버로 rsync 후 리포트 재생성.
#
# 사용: 인자에서 숫자 -> seed, 문자 -> dataset 으로 자동 분류 (순서 무관).
#   sh jitter_exp/kpred_llama.sh                # seed 2024 2025 2026 x dataset 7개
#   sh jitter_exp/kpred_llama.sh sst2 mnli      # 일부 데이터셋만
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo 2>/dev/null || cd "$(dirname "$0")/.."

# ---- 인자 분류: 숫자=seed, 그 외=dataset ----
SEEDS=""; DATASETS=""
for a in "$@"; do
  case "$a" in
    [0-9]*) SEEDS="$SEEDS $a" ;;
    *)      DATASETS="$DATASETS $a" ;;
  esac
done
SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-sst2 mnli qqp mr rte ag_news mrpc}"
echo "[cfg] seeds:$SEEDS  datasets:$DATASETS"

for D in $DATASETS; do
  echo "################ kpred llama  $D ################"
  python jitter_exp/compute_kernel_pred_error.py --dataset $D --model llama \
    --seeds $SEEDS --ranks 1 5 10 15 20 25 30 --lams 1e-2 --epss 1e-8 \
    --methods eigen nystrom_pinv nystrom_lev
done
echo "[done] kpred llama (seeds:$SEEDS / datasets:$DATASETS)"

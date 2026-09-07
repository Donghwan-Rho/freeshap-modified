#!/bin/sh
# ============================================================
# llama kernel_prediction npz 백필 — 리포트 p3/p4 (kernel/kpred 상대오차) 재료.
#   현재 비어있는 부분 (2026-08-12 조사, sst2/mnli 는 완료):
#     qqp / mr / rte / ag_news / mrpc — inv 0/3, eigen·pinv·lev 각 0/21 (전부)
#   NTK 캐시(num2000)만 필요 — SV/실험 진행과 무관하게 실행 가능.
#   npz 있는 셀은 내부에서 자동 스킵 (증분 안전, 재실행 OK).
#   llama 모델 로드가 (dataset, seed)마다 일어나므로 llama 있는 GPU 서버에서:
#     CUDA_VISIBLE_DEVICES=<GPU> sh n04_1.sh
# 끝나면 리포트 재생성 시 p3/p4 채워짐. (다른 서버에서 돌렸다면
#   jitter_exp/*_res/kernel_prediction, freeshap_res/kernel_prediction 을 rsync)
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo 2>/dev/null || cd "$(dirname "$0")"

for D in qqp mr rte ag_news mrpc; do
  echo "################ kpred llama  $D ################"
  python jitter_exp/compute_kernel_pred_error.py --dataset $D --model llama \
    --seeds 2024 2025 2026 --ranks 1 5 10 15 20 25 30 --lams 1e-2 --epss 1e-8 \
    --methods eigen nystrom_pinv nystrom_lev
done
echo "[done] kpred llama backfill (qqp mr rte ag_news mrpc)"

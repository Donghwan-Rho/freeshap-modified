#!/bin/sh
# ============================================================
# nys_pinv vs eigen 비교 리포트 — selection / removal / wld, dataset 7개.
#   nys_pinv (nys_pinv_res) vs eigen (jitter_exp/res), inv 기준선(freeshap_res / res).
#   출력: jitter_exp/report_eigen_pinv_{task}_{dataset}.pdf
#         (n2000/tmc500 은 각 리포트 첫 페이지/제목에 표시)
# 사용: sh jitter_exp/report_eigen_pinv.sh            # 7개 dataset 전부
#       sh jitter_exp/report_eigen_pinv.sh qqp rte    # 일부만
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
DATASETS="${*:-qqp mr rte sst2 mnli ag_news mrpc}"
M="--methods nystrom_pinv eigen"

for D in $DATASETS; do
  echo "################ eigen vs nys_pinv : $D ################"
  $PY jitter_exp/build_pinv_selection_report.py --dataset $D $M --acc_k30 20 \
      --out jitter_exp/report_eigen_pinv_selection_$D.pdf
  $PY jitter_exp/build_pinv_removal_report.py   --dataset $D $M --auc_k 20 \
      --out jitter_exp/report_eigen_pinv_removal_$D.pdf
  $PY jitter_exp/build_pinv_wld_report.py       --dataset $D $M --poison 10 --det_k30 20 \
      --out jitter_exp/report_eigen_pinv_wld_$D.pdf
  $PY jitter_exp/build_rel_error_report.py      --dataset $D $M \
      --out jitter_exp/report_rel_error_$D.pdf
done
echo "[done] eigen vs nys_pinv reports (datasets: $DATASETS)"

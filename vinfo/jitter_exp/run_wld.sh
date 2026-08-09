#!/bin/sh
# ============================================================
# 사용: sh jitter_exp/run_wld.sh <dataset> [seeds...]
#   예:  sh jitter_exp/run_wld.sh qqp
#        sh jitter_exp/run_wld.sh mr 2024
#
# task_wrong_label_detection.py 로 이미 만든 poison SV(pkl)를 재사용해
# wrong-label detection predictions(txt/pkl) 만 생성.
#   - shapley pkl 있으면 TMC skip(로드), detection 결과 txt 없을 때만 실행.
#   - 존재하는 모든 (lam,eps) poison 셀 + inv (전부 ./jitter_exp/res).
# 이미 만든 detection txt 는 skip (resume 가능).
# (run_selection.sh / run_removal.sh 와 동일한 구조, poison SV 대상)
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
DS=${1:?"dataset 이름을 인자로 주세요 (예: qqp)"}
shift
SEEDS="${*:-2024 2025 2026}"
ROOT=./jitter_exp/res
INVLAM=1e-6            # eigen/nystrom wld 의 inv 커널 정규화 (grid 와 동일, 파일명엔 미포함)
x() { echo "$1" | grep -oP "$2"; }

for S in $SEEDS; do
  # ---- eigen: 존재하는 모든 poison (lam,eps) 셀 ----
  for f in $ROOT/shapley/$DS/eigen/bert_seed${S}_*poison*.pkl; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    out="$ROOT/wrong_label_detection/$DS/eigen/predictions/${b%.pkl}_detection.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    rank=$(x "$b" '_eig\K[0-9]+'); elam=$(x "$b" '_lam\K[0-9.e+-]+'); eeps=$(x "$b" '_eigeps\K[0-9.e+-]+')
    pct=$(x "$b" '_poison\K[0-9]+'); ps=$(x "$b" '_ps\K[0-9]+')
    psarg=""; [ -n "$ps" ] && psarg="--poison_seed $ps"
    echo "[run] wld-eigen $DS seed$S lam$elam eigeps$eeps poison$pct"
    $PY task_wrong_label_detection.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate eigen --eigen_rank $rank --eigen_lambda_ $elam --eigeps $eeps \
      --inv_lambda_ $INVLAM --poison_pct $pct $psarg --tmc_iter $tmc --out_root $ROOT
  done

  # ---- nystrom: 존재하는 모든 poison (lam,eps) 셀 ----
  for f in $ROOT/shapley/$DS/nystrom/bert_seed${S}_*poison*.pkl; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    out="$ROOT/wrong_label_detection/$DS/nystrom/predictions/${b%.pkl}_detection.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    nd=$(x "$b" '_nys\K[0-9]+'); nlam=$(x "$b" '_lam\K[0-9.e+-]+'); neps=$(x "$b" '_nyseps\K[0-9.e+-]+')
    pct=$(x "$b" '_poison\K[0-9]+'); ps=$(x "$b" '_ps\K[0-9]+')
    psarg=""; [ -n "$ps" ] && psarg="--poison_seed $ps"
    echo "[run] wld-nystrom $DS seed$S lam$nlam nyseps$neps poison$pct"
    $PY task_wrong_label_detection.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate nystrom --nystrom_d $nd --nystrom_lambda_ $nlam --nyseps $neps \
      --inv_lambda_ $INVLAM --poison_pct $pct $psarg --tmc_iter $tmc --out_root $ROOT
  done

  # ---- inv (exact 기준선) ----
  for f in $ROOT/shapley/$DS/inv/bert_seed${S}_*poison*.pkl; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    out="$ROOT/wrong_label_detection/$DS/inv/predictions/${b%.pkl}_detection.txt"
    [ -e "$out" ] && continue
    num=$(x "$b" '_num\K[0-9]+'); val=$(x "$b" '_val\K[0-9]+'); tmc=$(x "$b" '_tmc\K[0-9]+')
    ilam=$(x "$b" '_lam\K[0-9.e+-]+'); pct=$(x "$b" '_poison\K[0-9]+'); ps=$(x "$b" '_ps\K[0-9]+')
    psarg=""; [ -n "$ps" ] && psarg="--poison_seed $ps"
    echo "[run] wld-inv $DS seed$S poison$pct"
    $PY task_wrong_label_detection.py --dataset_name $DS --seed $S --num_train_dp $num --val_sample_num $val \
      --approximate inv --inv_lambda_ $ilam --poison_pct $pct $psarg --tmc_iter $tmc --out_root $ROOT
  done
done
echo "[done] $DS wld"

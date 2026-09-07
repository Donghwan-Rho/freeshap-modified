#!/bin/sh
# ============================================================
# SV-landmark Nystrom (nystrom_q4 / nystrom_q0) probe — rank sweep, BERT (ntk_prompt).
#   pinv 와 동일한 pseudoinverse 구성. landmark 만 uniform 랜덤 대신
#   **같은 seed 의 exact(inv) Shapley value 상위/하위/중앙 d 개** 를 결정적으로 고른다.
#   SV 오름차순 정렬 후 분위 위치를 중심으로 d 개를 뽑는 5 단계
#     q0(min) < q1(25%) < q2(median) < q3(75%) < q4(max)
#   를 uniform(pinv) 과 비교해, landmark 선택이 만드는
#   품질 범위(= Nystrom 의 landmark 민감도)를 rank 축으로 측정하는 것이 목적.
#   목적: Nystrom 의 불안정성이 landmark 선택에 얼마나 좌우되는지의 상/하한 측정.
#
#   ※ oracle diagnostic 입니다 — inv SV 를 미리 알아야 하므로 실용적 '방법' 이 아닙니다.
#     또 landmark 에 들어간 점은 커널 행이 정확히 복원되어 그 점의 SV 가 맞는 건 자명하므로,
#     분석은 **landmark 가 아닌 점** 으로 한정해야 순환 논증이 아닙니다.
#
# 루프 구조: seed -> dataset -> rank -> **mode(top,bottom)** -> task
#   같은 rank 의 top 이 끝나면 바로 bottom 이 돌아, rank 단위로 곧장 비교할 수 있다.
#
# 완전 격리: out_root=./jitter_exp/nys_landmark_{top,bottom}_res,
#   method_dir=nystrom_{top,bottom}, 파일명 태그 _nysq4 / _nysq0.
#   기존 eigen/nystrom/pinv/lev 결과·경로에 영향 없음. NTK 는 ./freeshap_res/ntk 공유.
#   landmark 재료인 inv SV 는 --landmark_sv_root (기본 ./freeshap_res) 에서 읽는다.
#     -> 해당 (seed, dataset) 의 inv shapley pkl 이 없으면 그 조합을 건너뛴다.
#
# 세팅: num_train=2000 (bert/llama/resnet 공통; NUM=5000 으로 덮어쓰기 가능), nyslam=1e-2, nyseps=1e-8 고정, tmc=500.
#   (이 모드에서 --nyseps auto 는 금지 — auto 는 랜덤 subset 으로 σ_min(W) 를 추정해
#    실제 landmark 와 어긋난다.)
#
# task 토글 — 기본은 shapley + selection 만 (landmark 분석에 필요한 것만):
#   DO_SHAPLEY=1 DO_SELECTION=1 DO_REMOVAL=0 DO_WLD=0
#   ※ wld 는 poison 라벨로 TMC 를 처음부터 다시 돌아서 켜면 시간이 2배 이상 든다.
#
# 사용:
#   sh jitter_exp/nys_landmark.sh                        # bert, seed 3 x dataset 7, top+bottom
#   sh jitter_exp/nys_landmark.sh 2024 ag_news           # seed 2024 x ag_news
#   MODES=q2 sh jitter_exp/nys_landmark.sh 2024          # 하나만
#   MODES="q1 q3" sh jitter_exp/nys_landmark.sh 2024     # 분위 두 개만
#   RANKS="1 30" sh jitter_exp/nys_landmark.sh 2024 rte  # 빠른 점검
#   DO_REMOVAL=1 DO_WLD=1 sh jitter_exp/nys_landmark.sh  # 나머지 task 도
#   MODEL=llama sh jitter_exp/nys_landmark.sh            # llama 로
# 결과 파일 있으면 스킵 (증분 안전).
# ============================================================
cd /extdata1/donghwan/freeshap/vinfo
PY=/home/donghwan/.conda/envs/freeshap/bin/python
REM="$(seq -s' ' 0 99)"
RANKS=${RANKS:-"1 5 10 15 20 25 30"}
MODES=${MODES:-"q4 q3 q2 q1 q0"}   # rank 마다 이 순서(SV 높은 쪽->낮은 쪽)로 돈다
SV_ROOT=${SV_ROOT:-./freeshap_res}          # landmark 재료(inv shapley)를 읽을 루트
DO_SHAPLEY=${DO_SHAPLEY:-1}
DO_SELECTION=${DO_SELECTION:-1}
DO_REMOVAL=${DO_REMOVAL:-0}
DO_WLD=${DO_WLD:-0}

# ---- 모델 (파일명 접두 = 모델명, config 는 모델별로 다름) ----
# ---- 모델별: config / task 스크립트 접두 / 기본 dataset·N ----
#   vision(resnet) 은 별도 스크립트(vision/task_*_vision.py)를 쓴다.
MODEL=${MODEL:-bert}
case "$MODEL" in
  bert)   CFG=ntk_prompt ; PFX="."        ; SFX=""        ; DEF_DS="qqp mr rte sst2 mnli ag_news mrpc" ; DEF_N=2000 ;;
  llama)  CFG=ntk_llama  ; PFX="."        ; SFX=""        ; DEF_DS="qqp mr rte sst2 mnli ag_news mrpc" ; DEF_N=2000 ;;
  resnet) CFG=ntk_vision ; PFX="./vision" ; SFX="_vision" ; DEF_DS="cifar10"                           ; DEF_N=2000 ;;
  *) echo "[error] MODEL 은 bert / llama / resnet 이어야 합니다 (받은 값: $MODEL)"; exit 1 ;;
esac
NUM=${NUM:-$DEF_N}
for M in $MODES; do
  case "$M" in q4|q3|q2|q1|q0) ;; *) echo "[error] MODES 는 q4/q3/q2/q1/q0 만 (받은 값: $M)"; exit 1 ;; esac
done

# ---- 인자 분류: 숫자=seed, 그 외=dataset ----
SEEDS=""; DATASETS=""
for a in "$@"; do
  case "$a" in
    [0-9]*) SEEDS="$SEEDS $a" ;;
    *)      DATASETS="$DATASETS $a" ;;
  esac
done
SEEDS="${SEEDS:-2024 2025 2026}"
DATASETS="${DATASETS:-$DEF_DS}"
echo "[cfg] model:$MODEL (config=$CFG, scripts=$PFX/task_*$SFX.py, N=$NUM)  modes:$MODES  sv_root:$SV_ROOT"
echo "[cfg] seeds:$SEEDS  datasets:$DATASETS  ranks:$RANKS"
echo "[cfg] tasks: shapley=$DO_SHAPLEY selection=$DO_SELECTION removal=$DO_REMOVAL wld=$DO_WLD"

for S in $SEEDS; do
  for D in $DATASETS; do

    # ---- dataset 별 val_sample_num (grid_search.sh 와 동일) ----
    case "$D" in
      sst2)    V=872  ;;
      mrpc)   V=408  ;;
      rte)    V=277  ;;
      *)      V=1000 ;;
    esac

    # ---- landmark 재료 확인: 같은 seed 의 inv shapley 가 없으면 이 (seed,dataset) 건너뜀 ----
    INV_SV=$SV_ROOT/shapley/$D/inv/${MODEL}_seed${S}_num${NUM}_val${V}_lam1e-06_signFalse_earlystopTrue_tmc500.pkl
    if [ ! -f "$INV_SV" ]; then
      echo "[skip-ds] $D seed$S — inv SV 없음 ($INV_SV). 해당 모델의 inv 를 먼저 돌리세요."
      continue
    fi

    for R in $RANKS; do
      for MODE in $MODES; do        # ★ rank 안에서 q4 -> q0 순 (rank 단위로 바로 비교)
        TAG=nys$MODE               # q0..q4 -> nysq0..nysq4
        OUT=./jitter_exp/nys_landmark_${MODE}_res   # 모델은 파일명 접두로 구분 (bert_/llama_)
        echo "[$MODEL/$MODE] $D (val=$V) seed$S rank=${R}%"

        # ---- 결과 파일 경로 (이미 있으면 해당 task 스킵) ----
        # shapley/selection/removal 파일명: _{TAG}{R}.0_nyslam..._invlam... (신형)
        # wrong-label 파일명: _{TAG}{R}_lam..._poison10_ps{seed} (wld 고유 형식)
        STEM="${MODEL}_seed${S}_num${NUM}_val${V}_${TAG}${R}.0_nyslam1e-02_nyseps1e-8_invlam1e-06_cholesky_float32_signFalse_earlystopTrue_tmc500"
        WSTEM="${MODEL}_seed${S}_num${NUM}_val${V}_${TAG}${R}_lam1e-02_nyseps1e-8_cholesky_float32_signFalse_earlystopTrue_tmc500_poison10_ps${S}"
        SV_PKL=$OUT/shapley/$D/nystrom_${MODE}/${STEM}.pkl
        SEL_TXT=$OUT/data_selection/$D/nystrom_${MODE}/indices/${STEM}_indices.txt
        REM_TXT=$OUT/data_removing/$D/nystrom_${MODE}/predictions/${STEM}_predictions.txt
        WLD_TXT=$OUT/wrong_label_detection/$D/nystrom_${MODE}/predictions/${WSTEM}_detection.txt
        WLD_PKL=$OUT/wrong_label_detection/$D/nystrom_${MODE}/predictions/${WSTEM}_detection.pkl

        if [ "$DO_SHAPLEY" = "1" ]; then
          if [ -f "$SV_PKL" ]; then echo "  [skip] shapley  ($SV_PKL)"
          else
            $PY $PFX/task_shapley$SFX.py --config $CFG --seed $S --dataset_name $D \
              --num_train_dp $NUM --val_sample_num $V --approximate nystrom_$MODE --nystrom_d $R \
              --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
              --out_root $OUT --landmark_sv_root $SV_ROOT
          fi
        fi
        if [ "$DO_SELECTION" = "1" ]; then
          if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] selection"
          elif [ -f "$SEL_TXT" ]; then echo "  [skip] selection ($SEL_TXT)"
          else
            $PY $PFX/task_data_selection$SFX.py --config $CFG --seed $S --dataset_name $D \
              --num_train_dp $NUM --val_sample_num $V --approximate nystrom_$MODE --nystrom_d $R \
              --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
              --out_root $OUT --landmark_sv_root $SV_ROOT
          fi
        fi
        if [ "$DO_REMOVAL" = "1" ]; then
          if [ ! -f "$SV_PKL" ]; then echo "  [no-sv] removal"
          elif [ -f "$REM_TXT" ]; then echo "  [skip] removal  ($REM_TXT)"
          else
            $PY $PFX/task_data_removal$SFX.py --config $CFG --seed $S --dataset_name $D \
              --num_train_dp $NUM --val_sample_num $V --approximate nystrom_$MODE --nystrom_d $R \
              --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --tmc_iter 500 \
              --out_root $OUT --landmark_sv_root $SV_ROOT --num_train_removed_list $REM
          fi
        fi
        if [ "$DO_WLD" = "1" ]; then
          if [ -f "$WLD_TXT" ] && [ -f "$WLD_PKL" ]; then echo "  [skip] wrong-label ($WLD_TXT)"
          else
            $PY $PFX/task_wrong_label_detection$SFX.py --config $CFG --seed $S --dataset_name $D \
              --num_train_dp $NUM --val_sample_num $V --approximate nystrom_$MODE --nystrom_d $R \
              --inv_lambda_ 1e-6 --nystrom_lambda_ 1e-2 --nyseps 1e-8 --poison_pct 10 --tmc_iter 500 \
              --out_root $OUT --landmark_sv_root $SV_ROOT
          fi
        fi
      done
    done

  done
done
echo "[done] $MODEL nystrom [$MODES] (seeds:$SEEDS / datasets:$DATASETS)"

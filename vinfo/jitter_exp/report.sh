#!/bin/sh

python jitter_exp/build_selection_report.py       --dataset ag_news --acc_k30 20   # selection
python jitter_exp/build_wld_report.py   --dataset ag_news --poison 10 --det_k30 20   # wld
python jitter_exp/build_removal_report.py --dataset ag_news --auc_k 20   # removal

python jitter_exp/build_selection_report.py       --dataset mnli --acc_k30 20   # selection
python jitter_exp/build_wld_report.py   --dataset mnli --poison 10 --det_k30 20   # wld
python jitter_exp/build_removal_report.py --dataset mnli --auc_k 20   # removal

python jitter_exp/build_selection_report.py       --dataset mrpc --acc_k30 20   # selection
python jitter_exp/build_wld_report.py   --dataset mrpc --poison 10 --det_k30 20   # wld
python jitter_exp/build_removal_report.py --dataset mrpc --auc_k 20   # removal

python jitter_exp/build_selection_report.py       --dataset qqp --acc_k30 20   # selection
python jitter_exp/build_wld_report.py   --dataset qqp --poison 10 --det_k30 20   # wld
python jitter_exp/build_removal_report.py --dataset qqp --auc_k 20   # removal

python jitter_exp/build_selection_report.py       --dataset mr --acc_k30 20   # selection
python jitter_exp/build_wld_report.py   --dataset mr --poison 10 --det_k30 20   # wld
python jitter_exp/build_removal_report.py --dataset mr --auc_k 20   # removal

python jitter_exp/build_selection_report.py       --dataset rte --acc_k30 20   # selection
python jitter_exp/build_wld_report.py   --dataset rte --poison 10 --det_k30 20   # wld
python jitter_exp/build_removal_report.py --dataset rte --auc_k 20   # removal

python jitter_exp/build_selection_report.py       --dataset sst2 --acc_k30 20   # selection
python jitter_exp/build_wld_report.py   --dataset sst2 --poison 10 --det_k30 20   # wld
python jitter_exp/build_removal_report.py --dataset sst2 --auc_k 20   # removal

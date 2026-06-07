"""Compare inv-mode top-r accuracy: balanced baseline vs imbalance pos70.
For SST-2 and MR.  No A1 — only LRFShap (top-r by Shapley)."""
from datasets import load_dataset
import numpy as np
import json
import re
import glob

ROOT = "./freeshap_res"

def parse_top_from_pred(path):
    """predictions.txt: extract the most recent 'inv mode' 'top:' list (100 ints)."""
    txt = open(path).read()
    # Find last 'inv mode' block
    m = re.search(r"inv mode lambda[^\n]*\ntop:\s*\n\[([^\]]*)\]", txt, re.DOTALL)
    if not m:
        return None
    return [int(x.strip()) for x in m.group(1).split(",")]

def load_top_from_sidecar(path):
    d = json.load(open(path))
    return d["top_results_inv"]

# SST-2: baseline natural + imb pos70
sst2_base_pred = (f"{ROOT}/data_selection/sst2/inv/predictions/"
                  f"bert_seed2026_num2000_val872_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt")
sst2_imb_side  = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/pos70/lrfshap/inv/"
                  f"sidecar/bert_seed2026_num2000_val872_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500.json")

mr_base_pred = (f"{ROOT}/data_selection/mr/inv/predictions/"
                f"bert_seed2026_num2000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_predictions.txt")
mr_imb_side  = (f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/mr/pos70/lrfshap/inv/"
                f"sidecar/bert_seed2026_num2000_val1066_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500.json")

sst2_base_top = parse_top_from_pred(sst2_base_pred)
sst2_imb_top = load_top_from_sidecar(sst2_imb_side)
mr_base_top = parse_top_from_pred(mr_base_pred)
mr_imb_top = load_top_from_sidecar(mr_imb_side)

# baseline pool composition
def pool_pos_frac(idx_path):
    idx = np.array([int(l.strip()) for l in open(idx_path) if l.strip()])[:2000]
    return idx

sst2_labels = np.array(load_dataset("sst2")["train"]["label"])
mr_labels = np.array(load_dataset("rotten_tomatoes")["train"]["label"])

sst2_base_idx = pool_pos_frac(f"{ROOT}/data_selection/sst2/inv/indices/"
                              f"bert_seed2026_num2000_val872_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt")
sst2_imb_idx  = pool_pos_frac(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/pos70/lrfshap/inv/indices/"
                              f"bert_seed2026_num2000_val872_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt")
mr_base_idx = pool_pos_frac(f"{ROOT}/data_selection/mr/inv/indices/"
                            f"bert_seed2026_num2000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt")
mr_imb_idx  = pool_pos_frac(f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/mr/pos70/lrfshap/inv/indices/"
                            f"bert_seed2026_num2000_val1066_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt")

print(f"SST-2 baseline pool: pos={int((sst2_labels[sst2_base_idx]==1).sum())}/2000")
print(f"SST-2 imb pos70 pool: pos={int((sst2_labels[sst2_imb_idx]==1).sum())}/2000")
print(f"MR baseline pool:   pos={int((mr_labels[mr_base_idx]==1).sum())}/2000")
print(f"MR imb pos70 pool:  pos={int((mr_labels[mr_imb_idx]==1).sum())}/2000")
print()

# Print comparison table
print("=" * 100)
print(f"SST-2: inv-mode Shapley + top-r selection accuracy (top%) — val accuracy")
print("=" * 100)
print(f"{'sel':>4}  {'k':>5} | {'base_acc':>10} {'imb70_acc':>10} {'Δ(imb70-base)':>16} | "
      f"{'base_top_pos%':>14} {'imb_top_pos%':>14}")
for sel in [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]:
    k = int(2000 * sel / 100)
    b = sst2_base_top[sel-1] / 100
    i = sst2_imb_top[sel-1] / 100
    bp = (sst2_labels[sst2_base_idx[:k]] == 1).mean() * 100
    ip = (sst2_labels[sst2_imb_idx[:k]] == 1).mean() * 100
    print(f"{sel:>3}%  {k:>5d} |  {b:>8.2f}  {i:>9.2f}   {i-b:>+13.2f} |  {bp:>11.1f}   {ip:>11.1f}")

print()
print("=" * 100)
print(f"MR: inv-mode Shapley + top-r selection accuracy (top%) — val accuracy")
print("=" * 100)
print(f"{'sel':>4}  {'k':>5} | {'base_acc':>10} {'imb70_acc':>10} {'Δ(imb70-base)':>16} | "
      f"{'base_top_pos%':>14} {'imb_top_pos%':>14}")
for sel in [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]:
    k = int(2000 * sel / 100)
    b = mr_base_top[sel-1] / 100
    i = mr_imb_top[sel-1] / 100
    bp = (mr_labels[mr_base_idx[:k]] == 1).mean() * 100
    ip = (mr_labels[mr_imb_idx[:k]] == 1).mean() * 100
    print(f"{sel:>3}%  {k:>5d} |  {b:>8.2f}  {i:>9.2f}   {i-b:>+13.2f} |  {bp:>11.1f}   {ip:>11.1f}")

"""Compare Shapley top-r class distribution across four settings.

  (1) SST-2 baseline (natural)            n=2000  seed=2026  train_maj=positive
  (2) SST-2 imbalance pos=0.7 (enforced)  n=2000  seed=2026  train_maj=positive
  (3) MRPC   baseline (natural)           n=2000  seed=2026
  (4) QQP    baseline (natural)           n=2000  seed=2026
"""
from datasets import load_dataset
import numpy as np

ROOT = "./freeshap_res"

settings = [
    ("sst2_baseline", "sst2",
     f"{ROOT}/data_selection/sst2/inv/indices/bert_seed2026_num2000_val872_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt"),
    ("sst2_imb_pos70", "sst2",
     f"{ROOT}/claude_research/data_selection_test/imbalance/data_selection/sst2/pos70/lrfshap/inv/indices/"
     f"bert_seed2026_num2000_val872_lam1e-06_lrfshap_signFalse_earlystopTrue_tmc500_indices.txt"),
    ("mrpc_baseline", "mrpc",
     f"{ROOT}/data_selection/mrpc/inv/indices/bert_seed2026_num2000_val408_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt"),
    ("qqp_baseline", "qqp",
     f"{ROOT}/data_selection/qqp/inv/indices/bert_seed2026_num2000_val1000_lam1e-06_signFalse_earlystopTrue_tmc500_indices.txt"),
]

ds_cache = {}
def get_train_labels(ds_name):
    if ds_name in ds_cache:
        return ds_cache[ds_name]
    if ds_name == "sst2":
        ds = load_dataset("sst2")
        labels = np.array(ds["train"]["label"])
    elif ds_name == "mrpc":
        ds = load_dataset("glue", "mrpc")
        labels = np.array(ds["train"]["label"])
    elif ds_name == "qqp":
        ds = load_dataset("glue", "qqp")
        labels = np.array(ds["train"]["label"])
    else:
        raise ValueError(ds_name)
    ds_cache[ds_name] = labels
    return labels

def load_idx(p):
    return np.array([int(l.strip()) for l in open(p) if l.strip()])

# Header
print(f"{'setting':<18} {'ds':<6} {'n_pool':>7} {'label1_pool':>12} {'label0_pool':>12} {'majority_label':>15}")
pool_info = {}
for tag, ds, _ in settings:
    labs = get_train_labels(ds)
    pool_info[tag] = labs

print()
print("=" * 110)
print("Per-setting summary (using first 2000 indices from the file as the train pool)")
print("=" * 110)

for tag, ds, path in settings:
    idx_all = load_idx(path)
    # First-2000 trick: some baseline files were appended multiple runs (e.g. 6000 lines for sst2),
    # so we crop to the first 2000 as the single-run pool ordering.
    idx_pool = idx_all[:2000]
    labels = pool_info[tag]
    lab_pool = labels[idx_pool]
    n_pos = int((lab_pool == 1).sum())
    n_neg = int((lab_pool == 0).sum())
    maj = "positive (label 1)" if n_pos > n_neg else "negative (label 0)"
    print(f"\n>>> {tag} ({ds}) — pool: pos={n_pos}/{len(idx_pool)} ({n_pos/len(idx_pool):.3f}), "
          f"neg={n_neg}/{len(idx_pool)} ({n_neg/len(idx_pool):.3f}) | majority = {maj}")
    print(f"     Indices file: {path}")
    print(f"     {'sel':>4} {'k':>5}  {'top_pos':>7} {'top_neg':>7}  {'pos%':>6} {'neg%':>6}  "
          f"{'pool_pos%':>9} {'maj-in-top%':>11}")
    pool_pos_frac = n_pos / len(idx_pool)
    for sel in [1, 2, 5, 10, 15, 20, 30, 50, 100]:
        k = int(2000 * sel / 100)
        top = lab_pool[:k]
        tp = int((top == 1).sum())
        tn = int((top == 0).sum())
        # majority share in top-k
        maj_lab = 1 if n_pos > n_neg else 0
        maj_in_top = (top == maj_lab).mean()
        print(f"     {sel:>3}%  {k:>4d}  {tp:>6d}  {tn:>6d}  {tp/k*100:>5.1f}  {tn/k*100:>5.1f}  "
              f"{pool_pos_frac*100:>8.1f}  {maj_in_top*100:>10.1f}")

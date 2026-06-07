#!/usr/bin/env python
"""
build_grand_df.py
=================
Iteration 04, plan §1 — data unification step.

Reads every sidecar JSON under
  data_selection_test/imbalance/data_selection/<dataset>/<train_tag>/<method>/<approx>/sidecar/*.json
plus the matching NTK pickle under
  imbalance_ntk/<dataset>/<train_tag>/bert_seed2026_num<N>_<val_tag>_signFalse.pkl
and emits long-format unified CSVs plus an eigendecomposition cache.

Outputs (in state/iteration_04/):
  - ntk_schema.md     : sanity check + schema notes (first item per plan §1)
  - grand_df.csv      : one row per (setting × method × rank_pct × sel)
  - grand_meta.csv    : one row per (dataset, regime, train_tag, val_tag)
  - eig_cache/<setting_id>.npz : NTK eigendecomposition + label projections

This script is CPU-only.  NTK eigh for n=5000 takes ~60-90 s per setting.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Keep online — HF Hub will be hit once per dataset and then locally cached.
# (offline mode breaks for some datasets that don't ship a fully-resolved cache.)

import numpy as np
import pandas as pd
import torch

# ----- paths -----
WORK_ROOT = Path("/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research")
SIDECAR_ROOT = WORK_ROOT / "data_selection_test/imbalance/data_selection"
NTK_ROOT = WORK_ROOT / "imbalance_ntk"
OUT_DIR = WORK_ROOT / "state/iteration_04"
EIG_DIR = OUT_DIR / "eig_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EIG_DIR.mkdir(parents=True, exist_ok=True)


# ----- dataset → HF split metadata (mirrors task_imbalance_ntk.py) -----
def hf_loader_args(dataset_name: str):
    if dataset_name == "sst2":
        return ("sst2",), {}, "validation"
    if dataset_name == "mr":
        return ("rotten_tomatoes",), {}, "validation"
    if dataset_name == "qqp":
        return ("glue", "qqp"), {}, "validation"
    if dataset_name == "mnli":
        return ("glue", "mnli"), {}, "validation_matched"
    if dataset_name == "ag_news":
        return ("ag_news",), {}, "test"
    if dataset_name == "mrpc":
        return ("glue", "mrpc"), {}, "validation"
    if dataset_name == "rte":
        return ("glue", "rte"), {}, "validation"
    raise ValueError(f"unsupported dataset_name={dataset_name!r}")


_LABEL_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
LABELS_CACHE_DIR = OUT_DIR / "labels_cache"


def load_labels(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_labels, val_labels) for the dataset.

    Prefers pre-saved .npy under labels_cache/ to avoid HF Hub round-trips.
    Falls back to HF datasets.load_dataset() if cache is absent.
    """
    if dataset_name in _LABEL_CACHE:
        return _LABEL_CACHE[dataset_name]
    tr_path = LABELS_CACHE_DIR / f"{dataset_name}_train.npy"
    va_path = LABELS_CACHE_DIR / f"{dataset_name}_val.npy"
    if tr_path.exists() and va_path.exists():
        train_lbl = np.load(tr_path).astype(np.int64)
        val_lbl = np.load(va_path).astype(np.int64)
        _LABEL_CACHE[dataset_name] = (train_lbl, val_lbl)
        return train_lbl, val_lbl
    from datasets import load_dataset
    posargs, kwargs, val_split = hf_loader_args(dataset_name)
    ds_full = load_dataset(*posargs, **kwargs)
    train_lbl = np.asarray(ds_full["train"]["label"], dtype=np.int64)
    val_lbl = np.asarray(ds_full[val_split]["label"], dtype=np.int64)
    LABELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(tr_path, train_lbl)
    np.save(va_path, val_lbl)
    _LABEL_CACHE[dataset_name] = (train_lbl, val_lbl)
    return train_lbl, val_lbl


# ----- filename parsing -----
VAL_TAG_RE = re.compile(r"num(?P<n>\d+)_(?P<val>val(?:bal|imb)?\d+(?:_[a-z0-9_]+?)?)_(?:eig|lam)")
EIG_RE = re.compile(r"_eig(?P<rank>\d+(?:\.\d+)?)_")


def parse_sidecar_filename(fname: str) -> dict:
    """Return dict with num_train, val_tag, eig_rank (None for lrfshap_inv)."""
    out = {}
    m = VAL_TAG_RE.search(fname)
    if m:
        out["num_train"] = int(m.group("n"))
        out["val_tag"] = m.group("val")
    me = EIG_RE.search(fname)
    out["eig_rank_pct"] = float(me.group("rank")) if me else None
    return out


def classify_regime(train_tag: str, val_tag: str) -> str:
    """
    valbal: train imbalanced + val balanced.  Has 'valbal' in val_tag AND train != balanced.
    trainbal: train balanced + val imbalanced.  Has 'valimb' in val_tag AND train == balanced.
    single: plain raw validation slice ('valNNN' w/o bal/imb suffix), train usually imbalanced.
    """
    train_is_balanced = train_tag in {"pos50", "cls33_33_33", "cls25_25_25_25"}
    if "valbal" in val_tag:
        return "valbal"
    if "valimb" in val_tag:
        return "trainbal" if train_is_balanced else "valbalmix"
    # plain val<N> => raw HF validation
    return "single"


def imbalance_level(P_y: np.ndarray) -> str:
    return "mild" if P_y.max() <= 0.7 + 1e-9 else "extreme"


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) safe."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ----- NTK pickle discovery -----
def find_ntk_pickle(dataset: str, train_tag: str, num_train: int, val_tag: str) -> Optional[Path]:
    d = NTK_ROOT / dataset / train_tag
    if not d.exists():
        return None
    # candidate naming: bert_seed2026_num<N>_<val_tag>_signFalse.pkl
    cand = d / f"bert_seed2026_num{num_train}_{val_tag}_signFalse.pkl"
    if cand.exists():
        return cand
    # Some val_tag in sidecars may differ slightly; loose match
    for p in d.glob(f"bert_seed2026_num{num_train}_{val_tag}*signFalse.pkl"):
        return p
    return None


# ----- NTK eigendecomposition + label projection cache -----
def build_eig_cache(setting_id: str, pkl_path: Path,
                    P_train_y: np.ndarray, P_val_y: np.ndarray,
                    train_labels_local: np.ndarray, val_labels_local: np.ndarray,
                    k_top: int = 500) -> dict:
    """
    Compute and cache:
      - eigvals (n,)               sorted descending
      - eigvecs (n, k_top)         top-k_top columns
      - Y_tilde (n, C)             one-hot − P_train(c)
      - Y_val_tilde (n_val, C)     one-hot − P_val(c)
      - K_train_val (n, n_val)     K[train, val] block
      - eig_diag (n,)              full eigvals (re-stored for later use)
    """
    cache_path = EIG_DIR / f"{setting_id}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    ntk = bundle["ntk"]
    if isinstance(ntk, torch.Tensor):
        ntk = ntk.detach().cpu().numpy()
    if ntk.ndim == 3:
        ntk = ntk[0]
    n_train = len(bundle["sampled_idx"])
    K_train = ntk[:n_train, :n_train].astype(np.float64, copy=False)
    K_val_train = ntk[n_train:, :n_train].astype(np.float64, copy=False)
    K_train_val = K_val_train.T  # (n, n_val)

    # symmetrize for numerical stability
    K_train_sym = 0.5 * (K_train + K_train.T)
    # tiny jitter to avoid negative eigenvalues from float32 rounding
    jitter = 1e-6 * np.trace(K_train_sym) / n_train
    K_train_sym = K_train_sym + jitter * np.eye(n_train)

    t0 = time.time()
    eigvals, eigvecs = np.linalg.eigh(K_train_sym)  # ascending
    # reverse to descending
    eigvals = eigvals[::-1].astype(np.float64)
    eigvecs = eigvecs[:, ::-1].astype(np.float64)
    t1 = time.time()
    print(f"  [eig] n={n_train}, eigh took {t1-t0:.1f}s, "
          f"top λ = {eigvals[0]:.3e}, min λ = {eigvals[-1]:.3e}", flush=True)

    C = len(P_train_y)
    Y = np.zeros((n_train, C), dtype=np.float64)
    Y[np.arange(n_train), train_labels_local] = 1.0
    Y_tilde = Y - P_train_y[None, :]  # mean-centered with train marginal

    n_val = len(val_labels_local)
    Yv = np.zeros((n_val, C), dtype=np.float64)
    Yv[np.arange(n_val), val_labels_local] = 1.0
    Y_val_tilde = Yv - P_val_y[None, :]

    k = min(k_top, n_train)
    np.savez_compressed(
        cache_path,
        eigvals=eigvals.astype(np.float64),
        eigvecs=eigvecs[:, :k].astype(np.float32),
        Y_tilde=Y_tilde.astype(np.float32),
        Y_val_tilde=Y_val_tilde.astype(np.float32),
        K_train_val=K_train_val.astype(np.float32),
        n_train=np.int64(n_train),
        n_val=np.int64(n_val),
        C=np.int64(C),
        jitter=np.float64(jitter),
    )
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs[:, :k],
        "Y_tilde": Y_tilde,
        "Y_val_tilde": Y_val_tilde,
        "K_train_val": K_train_val,
        "n_train": n_train,
        "n_val": n_val,
        "C": C,
        "jitter": jitter,
    }


# ----- sidecar walking + grand_df assembly -----
def walk_sidecars():
    """Yield (json_path, dataset, train_tag, method, approx) for every sidecar."""
    for ds_dir in sorted(SIDECAR_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for tt_dir in sorted(ds_dir.iterdir()):
            if not tt_dir.is_dir():
                continue
            train_tag = tt_dir.name
            for m_dir in sorted(tt_dir.iterdir()):
                if not m_dir.is_dir() or m_dir.name not in {"a1", "lrfshap"}:
                    continue
                method = m_dir.name
                for ap_dir in sorted(m_dir.iterdir()):
                    if not ap_dir.is_dir() or ap_dir.name not in {"eigen", "inv"}:
                        continue
                    approx = ap_dir.name
                    sc_dir = ap_dir / "sidecar"
                    if not sc_dir.exists():
                        continue
                    for p in sorted(sc_dir.glob("*.json")):
                        yield p, dataset, train_tag, method, approx


def main():
    print("=" * 78)
    print("build_grand_df.py — iteration 04 §1 data unification")
    print("=" * 78)

    # === Step 1: write ntk_schema.md from a representative pickle ===
    schema_lines = []
    schema_pkl = NTK_ROOT / "sst2/pos70/bert_seed2026_num5000_valbal856_signFalse.pkl"
    schema_lines.append("# NTK pickle schema (auto-generated by build_grand_df.py)\n")
    schema_lines.append(f"Sampled pickle: `{schema_pkl.relative_to(WORK_ROOT)}`\n")
    with open(schema_pkl, "rb") as f:
        b = pickle.load(f)
    ntk = b["ntk"]
    if isinstance(ntk, torch.Tensor):
        ntk_shape = tuple(ntk.shape)
        ntk_dtype = str(ntk.dtype)
    else:
        ntk_shape = tuple(ntk.shape)
        ntk_dtype = str(ntk.dtype)
    schema_lines.append("## bundle.keys()\n")
    for k, v in b.items():
        if hasattr(v, "shape"):
            schema_lines.append(f"- `{k}`: shape={tuple(v.shape)}, dtype={v.dtype}\n")
        elif isinstance(v, list):
            schema_lines.append(f"- `{k}`: list len={len(v)}, first 3 = {v[:3]}\n")
        elif isinstance(v, dict):
            schema_lines.append(f"- `{k}`: dict = {v}\n")
        else:
            schema_lines.append(f"- `{k}`: {type(v).__name__} = {v}\n")
    n_train = len(b["sampled_idx"])
    n_val = len(b["sampled_val_idx"])
    schema_lines.append("\n## Layout decision\n")
    schema_lines.append(
        f"ntk.shape = {ntk_shape} = (1, n+n_val, n) = (1, {n_train}+{n_val}, {n_train}).\n"
        f"즉 leading axis 1 의 squeeze 후, K_train = ntk[:{n_train}, :{n_train}],\n"
        f"K_{{val,train}} = ntk[{n_train}:, :{n_train}] (shape {n_val}×{n_train}),\n"
        f"K_train_val = transpose of K_{{val,train}}.\n"
        f"**K_{{val,val}} block 은 pickle 에 없다.** lens 2 의 Nyström val-projection 은\n"
        f"K_train_val + eigendecomposition (K_train) 만으로 가능 (plan §3.2(2) 식이\n"
        f"정확히 이 두 quantity 만 요구함). 다만 KTA_full(K_val, Ỹ_val) 는 K_val_val\n"
        f"부재로 산출 불가 — lens 2 에서 KTA_train_with_val_label (train-side projection,\n"
        f"plan §3.2(4) fallback definition) 만 사용.\n"
    )
    schema_lines.append(
        "\n**fallback 영향**: plan §1 의 *train-only NTK fallback* 는 발동되지 않음 "
        "(K_train_val 가 있으므로). 그러나 *full val-side KTA* (K_val_val ⊗ Y_val Y_val⊤) "
        "는 산출 불가하여 H3/H4 의 KTA gap 정의를 'train-Ỹ KTA' vs 'train-Ỹ_via_val 의 KTA' "
        "로 사용 (plan §3.2(4) fallback). 추가로 Nyström-projected cᵢ²_val 는 plan 식 그대로.\n"
    )
    schema_md = OUT_DIR / "ntk_schema.md"
    schema_md.write_text("".join(schema_lines))
    print(f"[step 1] wrote {schema_md}")

    # === Step 2: walk sidecars, build records ===
    sidecar_records = []
    meta_records = {}  # keyed by setting_id
    rho_set = set()
    n_skipped_no_ntk = 0

    sidecars = list(walk_sidecars())
    print(f"[step 2] found {len(sidecars)} sidecar JSONs")

    for json_path, dataset, train_tag, method, approx in sidecars:
        try:
            j = json.load(open(json_path))
        except Exception as e:
            print(f"  [warn] parse fail {json_path}: {e}")
            continue
        fparse = parse_sidecar_filename(json_path.name)
        num_train = j["num_train_dp"]
        val_sample_num = j["val_sample_num"]
        val_tag = fparse.get("val_tag")
        if val_tag is None:
            # try alternative parsing — older naming "_lam1e-06_" for lrfshap inv
            # value of val_tag should be in filename between numNNN_ and _lam
            m = re.search(r"num\d+_(?P<v>val(?:bal|imb)?\d+(?:_[a-z0-9_]+?)?)_(?:lam|eig)", json_path.name)
            if m:
                val_tag = m.group("v")
        regime = classify_regime(train_tag, val_tag or "")
        eigen_rank_pct = j.get("eigen_rank_pct")
        rho_used = j.get("eigen_lambda_")
        if rho_used is not None:
            rho_set.add(rho_used)

        # class composition — newer sidecars store `class_ratios` (list); older
        # binary sidecars only have `pos_ratio` (float, label-1 fraction).
        if "class_ratios" in j and j["class_ratios"] is not None:
            class_ratios = np.asarray(j["class_ratios"], dtype=np.float64)
        elif "pos_ratio" in j and j["pos_ratio"] is not None:
            p1 = float(j["pos_ratio"])
            class_ratios = np.array([1.0 - p1, p1], dtype=np.float64)
        else:
            print(f"  [warn] no class_ratios / pos_ratio in {json_path.name}; skip")
            continue
        P_train_y = class_ratios / class_ratios.sum()
        C = len(P_train_y)

        # method label
        if method == "a1":
            m_label = "A1"
        elif method == "lrfshap" and approx == "inv":
            m_label = "FreeShap"
        elif method == "lrfshap" and approx == "eigen":
            m_label = "LR"
        else:
            m_label = method

        # find ntk pickle
        ntk_pkl = find_ntk_pickle(dataset, train_tag, num_train, val_tag) if val_tag else None
        if ntk_pkl is None:
            n_skipped_no_ntk += 1

        # setting_id
        setting_id = f"{dataset}_{train_tag}_{val_tag}_n{num_train}_nv{val_sample_num}"

        # P_val_y: if we have ntk pickle, load val labels via dataset
        P_val_y = None
        n_val_actual = val_sample_num
        per_class_val = None
        if ntk_pkl is not None and setting_id not in meta_records:
            try:
                with open(ntk_pkl, "rb") as f:
                    bundle = pickle.load(f)
                sampled_val_idx = np.asarray(bundle["sampled_val_idx"], dtype=np.int64)
                sampled_idx = np.asarray(bundle["sampled_idx"], dtype=np.int64)
                train_lbl_full, val_lbl_full = load_labels(dataset)
                val_lbl = val_lbl_full[sampled_val_idx]
                train_lbl = train_lbl_full[sampled_idx]
                # P_val_y empirical
                cnts_val = np.bincount(val_lbl, minlength=C).astype(np.float64)
                P_val_y = cnts_val / cnts_val.sum()
                per_class_val = cnts_val.astype(int).tolist()
                # P_train_y verification — should match class_ratios
                cnts_train = np.bincount(train_lbl, minlength=C).astype(np.float64)
                P_train_emp = cnts_train / cnts_train.sum()
                # NB: meta uses empirical
                P_train_y = P_train_emp
                per_class_train = cnts_train.astype(int).tolist()
                imb_level = imbalance_level(P_train_y)
                meta_records[setting_id] = {
                    "setting_id": setting_id,
                    "dataset": dataset,
                    "regime": regime,
                    "train_ratio_tag": train_tag,
                    "val_ratio_tag": val_tag,
                    "imbalance_level": imb_level,
                    "num_train": num_train,
                    "num_val": int(len(val_lbl)),
                    "C": int(C),
                    "ntk_path": str(ntk_pkl.relative_to(WORK_ROOT)),
                    "ntk_shape": str(tuple(bundle["ntk"].shape)),
                    "train_class_counts": per_class_train,
                    "val_class_counts": per_class_val,
                    "P_train_y": P_train_y.tolist(),
                    "P_val_y": P_val_y.tolist(),
                    "top_eigs_path": str((EIG_DIR / f"{setting_id}.npz").relative_to(WORK_ROOT)),
                }
            except Exception as e:
                print(f"  [warn] meta build failed for {setting_id}: {e}")

        # acc 4-tuple per cell — convert from {0..10000} int to fraction.
        top_results = j.get(f"top_results_{approx}") or []
        random_results = j.get(f"random_results_{approx}") or []
        top_balanced = j.get(f"top_results_{approx}_balanced") or []
        random_balanced = j.get(f"random_results_{approx}_balanced") or []
        # Fallback: some sidecars (multi-class, older format) lack the *_balanced
        # field.  When val is balanced (regime=valbal), naive acc IS balanced acc
        # by definition; we substitute naive as a *fallback* and tag the column
        # later with `balanced_is_naive_fallback` so analyses can be aware.
        balanced_is_fallback = False
        if not top_balanced:
            top_balanced = top_results
            balanced_is_fallback = True
        if not random_balanced:
            random_balanced = random_results
            balanced_is_fallback = True
        sel_list = j.get("num_train_selected_list") or []
        acc_at_f0 = j.get("acc_at_f0")

        # normalisation factor (some sidecars are 0..10000, some may differ)
        def norm(v):
            return float(v) / 10000.0

        af0 = norm(acc_at_f0) if acc_at_f0 is not None else None

        for i, sel in enumerate(sel_list):
            if i >= len(top_results):
                continue
            a_top = norm(top_results[i])
            a_rand = norm(random_results[i]) if i < len(random_results) else np.nan
            a_top_b = norm(top_balanced[i]) if i < len(top_balanced) else np.nan
            a_rand_b = norm(random_balanced[i]) if i < len(random_balanced) else np.nan
            sidecar_records.append({
                "setting_id": setting_id,
                "dataset": dataset,
                "regime": regime,
                "train_ratio_tag": train_tag,
                "val_ratio_tag": val_tag,
                "imbalance_level": imbalance_level(P_train_y),
                "num_train": int(num_train),
                "num_val": int(val_sample_num),
                "method": m_label,
                "approx": approx,
                "rank_pct": eigen_rank_pct,
                "sel": int(sel),
                "acc_top_naive": a_top,
                "acc_random_naive": a_rand,
                "acc_top_balanced": a_top_b,
                "acc_random_balanced": a_rand_b,
                "acc_at_f0": af0,
                "gap_top_random_naive": a_top - a_rand,
                "gap_top_random_balanced": a_top_b - a_rand_b,
                "rho_used": rho_used,
                "balanced_is_naive_fallback": balanced_is_fallback,
            })

    df = pd.DataFrame(sidecar_records)
    grand_df_path = OUT_DIR / "grand_df.csv"
    df.to_csv(grand_df_path, index=False)
    print(f"[step 2] wrote {grand_df_path}: {len(df):,} rows, "
          f"{df['setting_id'].nunique()} unique settings, {df['method'].unique().tolist()} methods")
    print(f"  [info] sidecars with no matching NTK pickle: {n_skipped_no_ntk}")
    print(f"  [info] rho_used set across sidecars: {sorted(rho_set)}")

    # === Step 3: meta DF + eigen cache ===
    meta_df = pd.DataFrame.from_records(list(meta_records.values()))
    # Stringify list columns for CSV
    for col in ["P_train_y", "P_val_y", "train_class_counts", "val_class_counts"]:
        if col in meta_df.columns:
            meta_df[col] = meta_df[col].apply(lambda v: json.dumps(v) if v is not None else None)
    meta_df_path = OUT_DIR / "grand_meta.csv"
    meta_df.to_csv(meta_df_path, index=False)
    print(f"[step 3a] wrote {meta_df_path}: {len(meta_df)} settings")

    # eigen cache build
    print(f"[step 3b] building eig_cache for {len(meta_records)} settings ...")
    n_done = 0
    n_skip = 0
    t_eig_start = time.time()
    for setting_id, m in meta_records.items():
        cache_path = EIG_DIR / f"{setting_id}.npz"
        if cache_path.exists():
            n_skip += 1
            continue
        pkl_path = WORK_ROOT / m["ntk_path"]
        if not pkl_path.exists():
            print(f"  [skip] missing ntk pickle {pkl_path}")
            continue
        try:
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
            sampled_idx = np.asarray(bundle["sampled_idx"], dtype=np.int64)
            sampled_val_idx = np.asarray(bundle["sampled_val_idx"], dtype=np.int64)
            train_lbl_full, val_lbl_full = load_labels(m["dataset"])
            train_labels_local = train_lbl_full[sampled_idx]
            val_labels_local = val_lbl_full[sampled_val_idx]
            P_train_y = np.asarray(json.loads(m["P_train_y"] if isinstance(m["P_train_y"], str) else json.dumps(m["P_train_y"])))
            P_val_y = np.asarray(json.loads(m["P_val_y"] if isinstance(m["P_val_y"], str) else json.dumps(m["P_val_y"])))
            print(f"  [eig {n_done+1}/{len(meta_records)-n_skip}] {setting_id}", flush=True)
            build_eig_cache(setting_id, pkl_path, P_train_y, P_val_y,
                            train_labels_local, val_labels_local)
            n_done += 1
        except Exception as e:
            print(f"  [error] eig build for {setting_id}: {e}")
    print(f"[step 3b] eig_cache: {n_done} new, {n_skip} cached. "
          f"total {time.time()-t_eig_start:.1f}s")

    # === Step 4: sanity checks ===
    print("[step 4] sanity checks")
    # (i) acc_at_f0 same across methods within a setting
    sanity = []
    for sid, g in df.groupby("setting_id"):
        af = g["acc_at_f0"].dropna().unique()
        sanity.append({"setting_id": sid, "n_distinct_acc_at_f0": len(af),
                       "acc_at_f0_vals": af.tolist()})
    sanity_df = pd.DataFrame(sanity)
    n_bad = (sanity_df["n_distinct_acc_at_f0"] > 1).sum()
    print(f"  (i) settings with >1 distinct acc_at_f0: {n_bad}/{len(sanity_df)}")
    # (ii) random_results moderate consistency across methods
    rand_check = []
    for sid, g in df.groupby("setting_id"):
        methods = g["method"].unique().tolist()
        if len(methods) < 2:
            continue
        # take eigen approx + rank 10 if available, else any rank
        sub = g[(g["approx"] == "eigen") & (g["rank_pct"] == 10.0)]
        if sub["method"].nunique() < 2:
            continue
        pivot = sub.pivot_table(index="sel", columns="method", values="acc_random_balanced")
        if pivot.shape[1] < 2:
            continue
        # MAE across methods
        ref = pivot.iloc[:, 0]
        maes = (pivot.subtract(ref, axis=0).abs().mean(axis=0)).max()
        rand_check.append({"setting_id": sid, "max_mae": maes})
    rc_df = pd.DataFrame(rand_check)
    if not rc_df.empty:
        moderate_fail = (rc_df["max_mae"] > 0.01).sum()
        print(f"  (ii) settings with random-method MAE > 0.01: {moderate_fail}/{len(rc_df)}")
    # (iii) acc_top_naive monotone non-decreasing in sel for FreeShap (sanity)
    sub = df[(df["method"] == "FreeShap")].copy()
    if not sub.empty:
        viol = 0
        n_settings = 0
        for sid, g in sub.groupby("setting_id"):
            g = g.sort_values("sel")
            seq = g["acc_top_naive"].values
            # allow small noise; count strict drops > 0.05
            drops = np.diff(seq)
            n_settings += 1
            if (drops < -0.05).any():
                viol += 1
        print(f"  (iii) FreeShap settings with large monotone violations: {viol}/{n_settings}")

    # === head + counts ===
    print()
    print("[head] grand_df.csv (first 5 rows):")
    print(df.head().to_string())
    print()
    print("[head] grand_meta.csv (first 5 rows):")
    print(meta_df.head().to_string())
    print()
    print(f"[done] grand_df rows: {len(df):,}")
    print(f"[done] grand_meta rows: {len(meta_df):,}")
    print(f"[done] eig_cache files: {len(list(EIG_DIR.glob('*.npz')))}")


if __name__ == "__main__":
    main()

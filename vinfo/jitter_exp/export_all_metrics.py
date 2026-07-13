# -*- coding: utf-8 -*-
"""jitter_exp 의 모든 결과를 통합 CSV 로 export.

산출:
  metrics_all.csv — 모든 (dataset, method, param) 조합의 metric들
  columns: dataset, method, param, N, V, D_PCT, tmc,
           spearman, sv_L2_rel, top5_ov, bot5_ov,
           top10_ov, bot10_ov, kendall_tau,
           sel_1pct, sel_5pct, sel_10pct, sel_20pct, sel_30pct, sel_50pct, sel_70pct, sel_100pct,
           rand_1pct, ..., rand_100pct,
           sigma_min_W, kappa_W,     # W = K[S,S]
           sv_time_sec, prep_time_sec
"""
import os, glob, json, pickle, re
import numpy as np
import torch
from scipy.stats import kendalltau

VINFO = "/extdata1/donghwan/freeshap/vinfo"
RESULTS = f"{VINFO}/jitter_exp/results"
NTK_DIR = f"{VINFO}/freeshap_res/ntk"
INV_DIR = f"{VINFO}/freeshap_res/shapley"
OUT = f"{VINFO}/jitter_exp/metrics_all.csv"

INSPECT_PCT = [1, 5, 10, 20, 30, 50, 70, 100]

def _r(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
def spearman(a, b):
    return float(np.corrcoef(_r(a), _r(b))[0, 1])
def _load_sv(p):
    d = pickle.load(open(p, 'rb'))
    dv = np.array(d['dv_result']); si = np.array(d['sampled_idx'])
    sv = dv[:, 1, :].sum(axis=1) if dv.ndim == 3 else dv[:, 1]
    return sv, si, d
def _align(a, ia, b, ib):
    da = {int(i): v for i, v in zip(ia.tolist(), a)}
    db = {int(i): v for i, v in zip(ib.tolist(), b)}
    c = sorted(set(ia.tolist()) & set(ib.tolist()))
    return (np.array([da[i] for i in c]), np.array([db[i] for i in c]))
def _overlap(a, b, k):
    return len(set(np.argsort(a)[-k:].tolist()) & set(np.argsort(b)[-k:].tolist())) / k
def _overlap_bot(a, b, k):
    return len(set(np.argsort(a)[:k].tolist()) & set(np.argsort(b)[:k].tolist())) / k

def _sigma_min_kappa(dataset, N, V, d_int, landmark_seed):
    """W = K[S,S], S = np.random.RandomState(landmark_seed).choice(N,d,replace=False)"""
    p = f"{NTK_DIR}/{dataset}/bert_seed2024_num{N}_val{V}_signFalse.pkl"
    if not os.path.exists(p): return None, None
    with open(p, 'rb') as f: nb = pickle.load(f)
    K = nb['ntk']
    K = (K[0] if K.ndim == 3 else K).to('cpu', dtype=torch.float64).numpy()[:N, :N]
    K = 0.5 * (K + K.T)
    rng = np.random.RandomState(int(landmark_seed))
    S = np.sort(rng.choice(N, size=d_int, replace=False))
    W = K[np.ix_(S, S)]
    ev = np.linalg.eigvalsh(W)
    return float(ev[0]), float(ev[-1] / ev[0])

def _inv_sv(dataset, N, V):
    p = f"{INV_DIR}/{dataset}/inv/bert_seed2024_num{N}_val{V}_lam1e-06_signFalse_earlystopTrue_tmc500.pkl"
    if not os.path.exists(p): return None, None
    return _load_sv(p)[:2]


rows = []
for dataset in ("qqp", "mr", "rte"):
    d_dir = f"{RESULTS}/{dataset}"
    if not os.path.isdir(d_dir): continue
    # inv reference (한 dataset 당 한 번)
    _V = 277 if dataset == "rte" else 1000
    inv_sv, inv_si = _inv_sv(dataset, 2000, _V)
    if inv_sv is None:
        print(f"[skip] {dataset}: inv missing")
        continue

    for p in sorted(glob.glob(f"{d_dir}/*.pkl")):
        if 'backup_before_repro' in p: continue
        fn = os.path.basename(p)
        # method / param 파싱
        if fn.startswith('eigen_'):
            method = 'eigen'
            m = re.search(r'eig(\d+)_', fn)
            if not m: continue
            param = int(m.group(1))
            param_str = f"r={param}%"
        elif fn.startswith('nys_'):
            method = 'nys'
            m = re.search(r'jitter([0-9.e+-]+)_', fn)
            if not m: continue
            param = float(m.group(1))
            param_str = f"ε={param:.0e}"
        else:
            continue

        # N/V/d/tmc 파싱
        m_n = re.search(r'num(\d+)_', fn); N = int(m_n.group(1)) if m_n else -1
        m_v = re.search(r'val(\d+)_', fn); V = int(m_v.group(1)) if m_v else -1
        m_d = re.search(r'_(?:nys|eig)(\d+)(?:\.0)?_', fn); D_PCT = int(m_d.group(1)) if m_d else -1
        m_tmc = re.search(r'tmc(\d+)', fn); tmc = int(m_tmc.group(1)) if m_tmc else -1

        # SV metrics
        try:
            asv, asi, pkl_data = _load_sv(p)
            a, b = _align(asv, asi, inv_sv, inv_si)
            if len(a) < 100:
                print(f"[skip] {fn}: too few common samples")
                continue
            sp = spearman(a, b)
            l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            k5 = max(1, int(len(a) * 0.05))
            k10 = max(1, int(len(a) * 0.10))
            top5 = _overlap(a, b, k5); bot5 = _overlap_bot(a, b, k5)
            top10 = _overlap(a, b, k10); bot10 = _overlap_bot(a, b, k10)
            kt, _ = kendalltau(a, b); kt = float(kt)
        except Exception as e:
            print(f"[error] {fn}: {e}")
            continue

        # W metrics
        d_int = int(N * D_PCT / 100)
        sigma_min_W, kappa_W = _sigma_min_kappa(dataset, N, V, d_int, landmark_seed=2024)

        # Selection JSON
        if method == 'eigen':
            sel_p = f"{d_dir}/selection_eigen_r{param}.json"
        else:
            sel_p = f"{d_dir}/selection_nys_{param:.0e}.json"
        sel_top = [None] * len(INSPECT_PCT)
        sel_rand = [None] * len(INSPECT_PCT)
        if os.path.exists(sel_p):
            sj = json.load(open(sel_p))
            for i, pct in enumerate(INSPECT_PCT):
                if pct in sj.get('inspect_pct', []):
                    idx = sj['inspect_pct'].index(pct)
                    sel_top[i] = sj['top_acc'][idx]
                    sel_rand[i] = sj['random_acc'][idx]

        row = dict(
            dataset=dataset, method=method, param=param, param_str=param_str,
            N=N, V=V, D_PCT=D_PCT, tmc=tmc,
            spearman=round(sp, 5), sv_L2_rel=round(l2, 5),
            top5_ov=round(top5, 5), bot5_ov=round(bot5, 5),
            top10_ov=round(top10, 5), bot10_ov=round(bot10, 5),
            kendall_tau=round(kt, 5),
            sigma_min_W=round(sigma_min_W, 3) if sigma_min_W else None,
            kappa_W=round(kappa_W, 1) if kappa_W else None,
            sv_time_sec=round(pkl_data.get('sv_time', 0), 2),
            prep_time_sec=round(pkl_data.get('prep_time', 0), 3),
            pkl_file=os.path.basename(p),
        )
        for i, pct in enumerate(INSPECT_PCT):
            row[f'sel_{pct}pct'] = sel_top[i]
            row[f'rand_{pct}pct'] = sel_rand[i]
        rows.append(row)

# CSV 저장
if rows:
    keys = list(rows[0].keys())
    with open(OUT, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    print(f"\n[write] {OUT}  ({len(rows)} rows)")
    print(f"columns: {keys}")
else:
    print("no rows")

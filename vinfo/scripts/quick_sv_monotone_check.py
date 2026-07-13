# SV Spearman/L2 error vs rank/d 표만 뽑기 (kernel 계산 없이 빠르게)
import os, pickle, numpy as np, glob

VINFO = "/extdata1/donghwan/freeshap/vinfo"
SH = f"{VINFO}/freeshap_res/shapley"
DS = [('sst2',5000,872), ('mnli',5000,1000), ('ag_news',5000,1000),
      ('mr',5000,1000), ('qqp',5000,1000), ('rte',2490,277), ('mrpc',3668,408)]
PCTS = [1, 5, 10, 15, 20, 25, 30]
LAM = '1e-02'
TAIL = 'signFalse_earlystopTrue_tmc500.pkl'

def load_sv(p):
    if not os.path.exists(p): return None,None
    d = pickle.load(open(p,'rb'))
    return np.array(d['dv_result'])[:,1,:].sum(axis=1), np.array(d['sampled_idx'])
def align(a,ia,b,ib):
    da={i:v for i,v in zip(ia,a)}; db={i:v for i,v in zip(ib,b)}
    c=sorted(set(ia.tolist())&set(ib.tolist()))
    return np.array([da[i] for i in c]), np.array([db[i] for i in c])
def _rank(x):
    r=np.empty(len(x)); r[np.argsort(x)]=np.arange(len(x)); return r
def spearman(a,b): return np.corrcoef(_rank(a),_rank(b))[0,1] if len(a)>1 else np.nan

print(f"SV Spearman (approx vs inv reference) at λ={LAM}")
print(f"{'':<10s} {'method':<6s}" + "".join([f'{p:>7d}%' for p in PCTS]))
print("-"*72)
for ds,N,V in DS:
    inv = f"{SH}/{ds}/inv/bert_seed2024_num{N}_val{V}_lam1e-06_{TAIL}"
    isv,isi = load_sv(inv)
    if isv is None: print(f"{ds}: inv missing"); continue
    for method, prefix in [('eigen','eig'), ('nys','nys')]:
        row = []
        for pct in PCTS:
            p = f"{SH}/{ds}/{'nystrom' if method=='nys' else 'eigen'}/bert_seed2024_num{N}_val{V}_{prefix}{pct}.0_{prefix.replace('eig','eig').replace('nys','nys')}lam{LAM}_invlam1e-06_cholesky_float32_{TAIL}"
            # fix suffix: eiglam vs nyslam
            if method=='eigen':
                p = f"{SH}/{ds}/eigen/bert_seed2024_num{N}_val{V}_eig{pct}.0_eiglam{LAM}_invlam1e-06_cholesky_float32_{TAIL}"
            else:
                p = f"{SH}/{ds}/nystrom/bert_seed2024_num{N}_val{V}_nys{pct}.0_nyslam{LAM}_invlam1e-06_cholesky_float32_{TAIL}"
            asv,asi = load_sv(p)
            if asv is None: row.append(np.nan); continue
            a,b = align(asv,asi,isv,isi)
            row.append(spearman(a,b))
        print(f"{ds:<10s} {method:<6s}" + "".join([f' {v:>7.3f}' if not np.isnan(v) else '     ---' for v in row]))
    print()

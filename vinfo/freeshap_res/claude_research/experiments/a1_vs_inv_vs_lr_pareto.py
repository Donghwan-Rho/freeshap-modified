"""Analyze (A1 vs INV) and the LR-rank-Pareto question."""
import os, sys
sys.path.insert(0, "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/experiments")
os.chdir("/extdata1/donghwan/freeshap/vinfo")
from make_lrfshap_vs_a1_pdf import SETTINGS, lr_eig_inv, a1_eig_inv, fs_inv

RANKS = [1, 5, 10, 15, 20, 25, 30]
SELS  = [1, 2, 3, 4, 5, 10, 20]

# ========== 1) A1 vs INV (cell-by-cell over 21 settings) ==========
print("\n=== [A1 vs INV] win count per (rank, sel)  — A1 > INV / equal / A1 < INV ===")
print(f"{'':>10}  " + "  ".join(f"sel{s:>2}%" for s in SELS))
total_a1_wins = total_a1_loss = total_a1_eq = total_cells = 0
for r in RANKS:
    cells = []
    for s in SELS:
        o = e = x = tot = 0
        for stg in SETTINGS:
            a1 = a1_eig_inv(stg, r); inv = fs_inv(stg)
            if a1 is None or inv is None: continue
            tot += 1
            if a1[s-1] > inv[s-1]: o += 1
            elif a1[s-1] == inv[s-1]: e += 1
            else: x += 1
        cells.append(f"{o}/{tot}")
        total_a1_wins += o; total_a1_loss += x; total_a1_eq += e; total_cells += tot
    print(f"r={r:>3}%   " + "  ".join(f"{c:>6}" for c in cells))
print(f"\nTotal cells (rank×sel×setting valid): {total_cells}")
print(f"  A1 > INV : {total_a1_wins} ({total_a1_wins/total_cells*100:.1f}%)")
print(f"  A1 = INV : {total_a1_eq} ({total_a1_eq/total_cells*100:.1f}%)")
print(f"  A1 < INV : {total_a1_loss} ({total_a1_loss/total_cells*100:.1f}%)")

# ========== 2) Pareto: does "LR-high-rank @ same sel" beat "A1-low-rank @ same sel"? ==========
print("\n=== [Pareto check] LR at higher rank vs A1 at lower rank, same sel% ===")
print("For each setting & sel%, take A1 at r_a=5% and compare to LR at r_l ∈ {10, 20, 30}%.\n"
      "Question: does LR's higher-rank curve dominate A1's low-rank choice?")
SEL_PARETO = [1, 2, 5, 10]
A1_RANK = 5
LR_RANKS_CMP = [10, 20, 30]
print(f"\nA1@r={A1_RANK}% vs LR@r∈{LR_RANKS_CMP}%  (counts: A1 better / equal / LR better)")
for r_l in LR_RANKS_CMP:
    print(f"\n  -- A1@r={A1_RANK}% vs LR@r={r_l}% --")
    print(f"  {'sel':>4}  " + "   ".join(f"A1>LR / = / A1<LR (n)" for _ in [0]))
    for s in SEL_PARETO:
        o = e = x = tot = 0
        for stg in SETTINGS:
            a1 = a1_eig_inv(stg, A1_RANK); lr = lr_eig_inv(stg, r_l)
            if a1 is None or lr is None: continue
            tot += 1
            if a1[s-1] > lr[s-1]: o += 1
            elif a1[s-1] == lr[s-1]: e += 1
            else: x += 1
        print(f"  {s:>3}%   {o:>3} / {e:>2} / {x:>3}  ({tot})")

# ========== 3) Best-overall-acc per setting: which (method, rank) gives max acc at sel=1, 5? ==========
print("\n=== [Best (method, rank) per setting & sel] — who wins overall at small sel ===")
for s in [1, 5]:
    wins = {"INV": 0, "A1": 0, "LR": 0, "TIE": 0}
    print(f"\n  -- sel = {s}% --")
    for stg in SETTINGS:
        inv = fs_inv(stg)
        best_a1 = (None, 0)  # (rank, acc)
        best_lr = (None, 0)
        for r in RANKS:
            a1 = a1_eig_inv(stg, r); lr = lr_eig_inv(stg, r)
            if a1 is not None and a1[s-1] > best_a1[1]:
                best_a1 = (r, a1[s-1])
            if lr is not None and lr[s-1] > best_lr[1]:
                best_lr = (r, lr[s-1])
        inv_v = inv[s-1] if inv else 0
        a1_v = best_a1[1]; lr_v = best_lr[1]
        max_v = max(inv_v, a1_v, lr_v)
        winners = [k for k, v in [("INV", inv_v), ("A1", a1_v), ("LR", lr_v)] if v == max_v and v > 0]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            wins["TIE"] += 1
        print(f"    {stg['name']:<28}  INV={inv_v/100:5.2f}  A1*={a1_v/100:5.2f}(r={best_a1[0]})  "
              f"LR*={lr_v/100:5.2f}(r={best_lr[0]})  winner={winners}")
    print(f"  Total winners @ sel={s}%: {wins}")

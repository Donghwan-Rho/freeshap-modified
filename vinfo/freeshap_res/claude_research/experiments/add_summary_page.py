"""Append a rank x sel summary page to lrfshap_vs_a1.pdf.
Each cell = 'A1>LR' count / total over the 21 settings."""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# reuse SETTINGS / lr_eig_inv / a1_eig_inv from the generator
sys.path.insert(0, "/extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research/experiments")
os.chdir("/extdata1/donghwan/freeshap/vinfo")
from make_lrfshap_vs_a1_pdf import SETTINGS, lr_eig_inv, a1_eig_inv

RANKS = [1, 5, 10, 15, 20, 25, 30]
SELS  = [1, 2, 3, 4, 5, 10, 20]
PDF   = "./freeshap_res/claude_research/reports/lrfshap_vs_a1.pdf"

# ---- compute matrix ----
# matrix[r_idx][s_idx] = (a1>lr count, equal count, a1<lr count, total)
matrix = {}
for r in RANKS:
    matrix[r] = {}
    for s in SELS:
        o = e = x = tot = 0
        for stg in SETTINGS:
            lr = lr_eig_inv(stg, r); a1 = a1_eig_inv(stg, r)
            if lr is None or a1 is None: continue
            tot += 1
            if a1[s-1] > lr[s-1]: o += 1
            elif a1[s-1] == lr[s-1]: e += 1
            else: x += 1
        matrix[r][s] = (o, e, x, tot)

# ---- build one-page figure with the table ----
tmp_pdf = "/tmp/_summary_page.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)
ax.axis('off')

# Header row + 7 ranks
col_labels = [""] + [f"sel {s}%" for s in SELS]
rows = []
for r in RANKS:
    row = [f"rank = {r}%"]
    for s in SELS:
        o, e, x, tot = matrix[r][s]
        pct = (o / tot * 100) if tot else 0
        row.append(f"{o}/{tot}\n({pct:.0f}%)")
    rows.append(row)

tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center',
               cellLoc='center', colWidths=[0.15] + [0.115]*len(SELS))
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 2.6)
for (i, j), cell in tbl.get_celld().items():
    if i == 0:
        cell.set_facecolor("#cccccc")
        cell.set_text_props(weight='bold')
    elif j == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight='bold')

fig.suptitle("Summary — A1 > LRFShap win count over 21 settings\n"
             "(eigen mode, INV-prediction, value = wins / total settings)",
             fontsize=13, y=0.96)
fig.text(0.5, 0.05,
         "Total settings = 21 (3 settings per dataset × 7 datasets, "
         "in-progress qqp_70/30, mrpc_70/30, rte_50/50 excluded).\n"
         "Settings missing any (rank, sel) data are skipped in that cell's denominator.",
         ha='center', fontsize=9, color='#444')

with PdfPages(tmp_pdf) as pp:
    pp.savefig(fig, bbox_inches='tight')
plt.close(fig)

# ---- merge with existing PDF ----
from pypdf import PdfWriter, PdfReader
out = "/tmp/_merged.pdf"
writer = PdfWriter()
for page in PdfReader(PDF).pages:
    writer.add_page(page)
for page in PdfReader(tmp_pdf).pages:
    writer.add_page(page)
with open(out, "wb") as f:
    writer.write(f)

# overwrite original
import shutil
shutil.move(out, PDF)
os.remove(tmp_pdf)
print(f"[done] appended summary page -> {PDF}")
print(f"[done] total pages now: {len(PdfReader(PDF).pages)}")

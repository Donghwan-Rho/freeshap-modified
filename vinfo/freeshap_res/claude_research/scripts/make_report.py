#!/usr/bin/env python
"""Usage: python make_report.py <iter_num>

Full archive report for iter NN:
- includes all rounds (plan.md, critique.md, plan_v2, critique_v2, plan_v3,
  critique_v3) plus bibliography.md, next_directions.md, any *.png figures,
  and results.json if present.
- pipeline: Markdown -> HTML (pandoc) -> PDF (WeasyPrint), with a CSS that
  pins Korean to Noto Sans CJK KR, math symbols to STIX Two Math / DejaVu
  Sans, and monospace to DejaVu Sans Mono. Tables, lists, code blocks, and
  Unicode math symbols (ρ, λ, √, ≤, ∈, ⊗, ...) render cleanly.
"""
import sys, json, subprocess, shutil
from pathlib import Path


CSS_TEXT = r"""
@page {
  size: A4;
  margin: 18mm 16mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: "Noto Sans CJK KR", "DejaVu Sans", sans-serif;
    font-size: 9pt;
    color: #666;
  }
}

html {
  /* Font fallback chain: Korean -> math -> generic latin.
     Any Unicode glyph missing in the first font falls through. */
  font-family: "Noto Sans CJK KR", "STIX Two Text", "STIX Two Math",
               "DejaVu Sans", "DejaVu Serif", "Segoe UI Symbol", sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #111;
}

h1 { font-size: 20pt; margin-top: 1.2em; margin-bottom: 0.3em;
     border-bottom: 2px solid #222; padding-bottom: 0.1em; }
h2 { font-size: 15pt; margin-top: 1.0em; margin-bottom: 0.25em;
     border-bottom: 1px solid #bbb; padding-bottom: 0.08em; }
h3 { font-size: 12.5pt; margin-top: 0.8em; margin-bottom: 0.2em; }
h4 { font-size: 11pt; margin-top: 0.7em; margin-bottom: 0.15em; }

p { margin: 0.35em 0; }

ul, ol { margin: 0.35em 0 0.35em 1.4em; padding: 0; }
li { margin: 0.15em 0; }

/* Tables: ensure visible borders, proper column widths, no page overflow. */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.6em 0;
  font-size: 9.5pt;
  table-layout: auto;
}
th, td {
  border: 0.5pt solid #555;
  padding: 4pt 6pt;
  vertical-align: top;
  word-wrap: break-word;
}
th { background: #eee; font-weight: 600; }

/* Inline + block code. */
code, pre, kbd, samp {
  font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK KR", monospace;
  font-size: 9.5pt;
}
code { background: #f3f3f3; padding: 0 2pt; border-radius: 2pt; }
pre {
  background: #f7f7f7;
  border: 0.5pt solid #ccc;
  padding: 6pt 8pt;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
}
pre code { background: transparent; padding: 0; }

/* Math symbols -> STIX Two Math specifically (Unicode maths block). */
.math, span.math, .mathrm { font-family: "STIX Two Math", "DejaVu Sans",
                                         "Noto Sans CJK KR", serif; }

blockquote {
  border-left: 3pt solid #bbb;
  margin: 0.4em 0;
  padding: 0.1em 0.8em;
  color: #333;
  background: #fafafa;
}

a { color: #064; text-decoration: none; word-break: break-all; }

img { max-width: 100%; height: auto; }

hr { border: none; border-top: 0.5pt solid #999; margin: 1em 0; }

.section-title {
  page-break-before: always;
  page-break-after: avoid;
}
.section-title:first-of-type { page-break-before: avoid; }

/* Avoid orphaned headings. */
h1, h2, h3, h4 { page-break-after: avoid; break-after: avoid-page; }
table, pre, figure { page-break-inside: avoid; break-inside: avoid-page; }
"""


def md_to_html_fragment(md_text: str) -> str:
    """Run pandoc md -> html5 fragment with GFM tables etc."""
    proc = subprocess.run(
        ["pandoc",
         "--from=gfm+tex_math_dollars",
         "--to=html5",
         "--mathml",
         "--wrap=none"],
        input=md_text, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return (f"<pre>[pandoc failed: {proc.returncode}]\n"
                f"{proc.stderr}\n---raw---\n{md_text}</pre>")
    return proc.stdout


def read_md(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_full_report(iter_num: int):
    from weasyprint import HTML, CSS

    root = Path(__file__).resolve().parent.parent
    iter_dir = root / "state" / f"iteration_{iter_num:02d}"
    pdf_path = root / "reports" / f"iter_{iter_num:02d}.pdf"
    pdf_path.parent.mkdir(exist_ok=True)

    if not iter_dir.exists():
        print(f"ERROR: {iter_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    sections = []

    def add_section(title: str, md_body: str):
        if not md_body.strip():
            return
        html_body = md_to_html_fragment(md_body)
        sections.append(
            f'<section class="section-title">'
            f'<h1>{title}</h1>\n{html_body}</section>'
        )

    # Title page (skip when the iteration is a synthesis-only round —
    # i.e. only report.md exists and no plan/critique pairs).
    has_rounds = any((iter_dir / f"plan{suf}.md").exists() or
                     (iter_dir / f"critique{suf}.md").exists()
                     for suf in ("", "_v2", "_v3"))
    has_synth = (iter_dir / "report.md").exists()
    if has_rounds or not has_synth:
        sections.append(
            f'<section class="section-title"><h1>Iteration {iter_num:02d} Report '
            f'(Full Archive)</h1>'
            f'<p><em>Auto-generated by <code>scripts/make_report.py {iter_num}</code>. '
            f'Includes every round (v1/v2/v3) plus bibliography, next directions, '
            f'results, and figures.</em></p></section>'
        )

    # Rounds
    round_suffixes = [("", "v1"), ("_v2", "v2"), ("_v3", "v3")]
    section_idx = 1
    for suffix, vtag in round_suffixes:
        plan_p = iter_dir / f"plan{suffix}.md"
        crit_p = iter_dir / f"critique{suffix}.md"
        if plan_p.exists():
            add_section(f"{section_idx}. Research Plan ({vtag})", read_md(plan_p))
            section_idx += 1
        if crit_p.exists():
            add_section(f"{section_idx}. Critic Analysis ({vtag})", read_md(crit_p))
            section_idx += 1

    # Standalone synthesis report (when an iteration produces one
    # consolidated document instead of plan/critique pairs).
    report_p = iter_dir / "report.md"
    if report_p.exists():
        add_section(f"{section_idx}. Synthesis Report", read_md(report_p))
        section_idx += 1

    # Results
    results_p = iter_dir / "results.json"
    if results_p.exists():
        try:
            r = json.loads(results_p.read_text(encoding="utf-8"))
            body = "```json\n" + json.dumps(r, indent=2, ensure_ascii=False) + "\n```"
        except Exception as e:
            body = f"*(failed to parse results.json: {e})*"
        add_section(f"{section_idx}. Experiment Results", body)
        section_idx += 1

    # Figures
    pngs = sorted(iter_dir.glob("*.png"))
    if pngs:
        body_lines = []
        for img in pngs:
            body_lines.append(f"### {img.name}\n\n![{img.name}]({img.as_uri()})\n")
        add_section(f"{section_idx}. Figures", "\n".join(body_lines))
        section_idx += 1

    bib_p = iter_dir / "bibliography.md"
    if bib_p.exists():
        add_section(f"{section_idx}. Bibliography", read_md(bib_p))
        section_idx += 1

    nd_p = iter_dir / "next_directions.md"
    if nd_p.exists():
        add_section(f"{section_idx}. Proposed Next Directions", read_md(nd_p))
        section_idx += 1

    html_doc = (
        "<!DOCTYPE html>\n<html lang='ko'>\n<head>\n<meta charset='utf-8'/>\n"
        f"<title>Iteration {iter_num:02d} Report</title>\n</head>\n<body>\n"
        + "\n".join(sections)
        + "\n</body>\n</html>\n"
    )

    HTML(string=html_doc, base_url=str(iter_dir)).write_pdf(
        str(pdf_path), stylesheets=[CSS(string=CSS_TEXT)]
    )
    print(f"Saved: {pdf_path}")


def _check_deps():
    if shutil.which("pandoc") is None:
        print("ERROR: pandoc not found in PATH", file=sys.stderr)
        sys.exit(1)
    try:
        import weasyprint  # noqa: F401
    except ImportError:
        print("ERROR: weasyprint not installed. pip install weasyprint",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: make_report.py <iter_num>", file=sys.stderr)
        sys.exit(1)
    _check_deps()
    build_full_report(int(sys.argv[1]))

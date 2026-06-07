#!/usr/bin/env python
"""Usage: python make_summary_report.py <iter_num>

Compact reading copy: final-round plan + critique + next_directions +
bibliography only. Same rendering pipeline as make_report.py (pandoc + WeasyPrint).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import CSS_TEXT, md_to_html_fragment, read_md, _check_deps  # noqa: E402


def build_summary_report(iter_num: int):
    from weasyprint import HTML, CSS

    root = Path(__file__).resolve().parent.parent
    iter_dir = root / "state" / f"iteration_{iter_num:02d}"
    pdf_path = root / "reports" / f"iter_{iter_num:02d}_summary.pdf"
    pdf_path.parent.mkdir(exist_ok=True)

    if not iter_dir.exists():
        print(f"ERROR: {iter_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    def pick_final(stem: str):
        for suffix in ("_v3", "_v2", ""):
            p = iter_dir / f"{stem}{suffix}.md"
            if p.exists():
                return p
        return None

    plan_p = pick_final("plan")
    crit_p = pick_final("critique")
    has_rounds = plan_p is not None or crit_p is not None
    has_synth = (iter_dir / "report.md").exists()

    # Synthesis-only iteration: full report.md already serves as the reading
    # copy, so skip emitting a separate summary PDF.
    if has_synth and not has_rounds:
        print(f"Skipping summary: iteration {iter_num:02d} is synthesis-only; "
              f"reports/iter_{iter_num:02d}.pdf already serves as reading copy.")
        return

    sections = [
        f'<section class="section-title"><h1>Iteration {iter_num:02d} — Summary '
        f'Report</h1><p><em>Final round only. Companion to the full archive '
        f'<code>reports/iter_{iter_num:02d}.pdf</code>.</em></p></section>'
    ]

    def add_section(title: str, md_body: str):
        if not md_body.strip():
            return
        html_body = md_to_html_fragment(md_body)
        sections.append(
            f'<section class="section-title">'
            f'<h1>{title}</h1>\n{html_body}</section>'
        )

    if plan_p:
        add_section(f"1. Final Research Plan ({plan_p.name})", read_md(plan_p))
    if crit_p:
        add_section(f"2. Final Critic Analysis ({crit_p.name})", read_md(crit_p))

    nd_p = iter_dir / "next_directions.md"
    if nd_p.exists():
        add_section("3. Proposed Next Directions", read_md(nd_p))

    bib_p = iter_dir / "bibliography.md"
    if bib_p.exists():
        add_section("4. Bibliography", read_md(bib_p))

    html_doc = (
        "<!DOCTYPE html>\n<html lang='ko'>\n<head>\n<meta charset='utf-8'/>\n"
        f"<title>Iteration {iter_num:02d} Summary</title>\n</head>\n<body>\n"
        + "\n".join(sections)
        + "\n</body>\n</html>\n"
    )

    HTML(string=html_doc, base_url=str(iter_dir)).write_pdf(
        str(pdf_path), stylesheets=[CSS(string=CSS_TEXT)]
    )
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: make_summary_report.py <iter_num>", file=sys.stderr)
        sys.exit(1)
    _check_deps()
    build_summary_report(int(sys.argv[1]))

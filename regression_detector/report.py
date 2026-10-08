"""Report rendering: Markdown and HTML summaries of a suite comparison."""

from __future__ import annotations

import html
from pathlib import Path

from regression_detector.compare import CaseComparison, SuiteComparison


def _fmt(x: float, digits: int = 4) -> str:
    if x == float("inf"):
        return "inf"
    if x != x:  # NaN
        return "n/a"
    return f"{x:.{digits}f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def render_markdown(comp: SuiteComparison) -> str:
    """Render a full Markdown report for a suite comparison."""
    badge = "🔴" if comp.regression_detected else "🟢"
    lines = [
        f"# Model Regression Report: {comp.suite_name}",
        "",
        f"**Verdict: {badge} {comp.verdict}**",
        "",
        "## Summary",
        "",
        f"- Baseline: `{comp.baseline_name}` — mean score {_fmt(comp.mean_baseline)}",
        f"- Candidate: `{comp.candidate_name}` — mean score {_fmt(comp.mean_candidate)}",
        f"- Mean delta (candidate − baseline): {_fmt(comp.mean_delta)}",
        f"- Cases: {comp.n_cases} "
        f"({_plural(comp.n_regressions, 'regression')}, "
        f"{_plural(comp.n_improvements, 'improvement')}, "
        f"{_plural(comp.n_unchanged, 'unchanged case')})",
        f"- Test: {comp.test_name}",
        f"- Statistic: {_fmt(comp.statistic)}, p-value: {_fmt(comp.p_value)} "
        f"(alpha = {comp.alpha}, significant: {comp.significant})",
        f"- Effect: {comp.effect_label}",
        "",
    ]

    regressions = [c for c in comp.cases if c.regression]
    if regressions:
        lines += ["## Worst regressions", ""]
        lines += ["| Case | Baseline | Candidate | Delta |", "|---|---|---|---|"]
        for c in sorted(regressions, key=lambda r: r.delta)[:20]:
            lines.append(
                f"| `{c.case_id}` | {_fmt(c.baseline_score)} "
                f"| {_fmt(c.candidate_score)} | {_fmt(c.delta)} |"
            )
        if len(regressions) > 20:
            lines.append(f"| ... and {len(regressions) - 20} more | | | |")
        lines.append("")

    lines += ["## Per-case details", ""]
    lines += ["| Case | Baseline | Candidate | Delta | Outcome |", "|---|---|---|---|---|"]
    for c in comp.cases:
        outcome = "🔴 regression" if c.regression else ("🟢 improvement" if c.improvement else "—")
        lines.append(
            f"| `{c.case_id}` | {_fmt(c.baseline_score)} "
            f"| {_fmt(c.candidate_score)} | {_fmt(c.delta)} | {outcome} |"
        )
    lines.append("")
    lines.append(
        "_Note: this tool measures score differences between model versions; "
        "it does not diagnose root causes. Statistical power depends on suite size._"
    )
    lines.append("")
    return "\n".join(lines)


def render_html(comp: SuiteComparison) -> str:
    """Render a standalone HTML report for a suite comparison."""
    badge = "🔴" if comp.regression_detected else "🟢"
    color = "#b91c1c" if comp.regression_detected else "#15803d"

    def row(c: CaseComparison) -> str:
        cls = "reg" if c.regression else ("imp" if c.improvement else "")
        return (
            f'<tr class="{cls}"><td><code>{html.escape(c.case_id)}</code></td>'
            f"<td>{_fmt(c.baseline_score)}</td><td>{_fmt(c.candidate_score)}</td>"
            f"<td>{_fmt(c.delta)}</td></tr>"
        )

    rows = "\n".join(row(c) for c in comp.cases)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model Regression Report: {html.escape(comp.suite_name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
.verdict {{ font-size: 1.4rem; font-weight: bold; color: {color}; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #f3f4f6; }}
tr.reg {{ background: #fee2e2; }}
tr.imp {{ background: #dcfce7; }}
.summary li {{ margin: 0.25rem 0; }}
</style>
</head>
<body>
<h1>Model Regression Report: {html.escape(comp.suite_name)}</h1>
<p class="verdict">{badge} {html.escape(comp.verdict)}</p>
<h2>Summary</h2>
<ul class="summary">
<li>Baseline <code>{html.escape(comp.baseline_name)}</code>: mean score {_fmt(comp.mean_baseline)}</li>
<li>Candidate <code>{html.escape(comp.candidate_name)}</code>: mean score {_fmt(comp.mean_candidate)}</li>
<li>Mean delta: {_fmt(comp.mean_delta)} over {comp.n_cases} cases
({comp.n_regressions} regressions, {comp.n_improvements} improvements, {comp.n_unchanged} unchanged)</li>
<li>{html.escape(comp.test_name)}: statistic {_fmt(comp.statistic)},
p-value {_fmt(comp.p_value)} (alpha {comp.alpha})</li>
<li>Effect: {html.escape(comp.effect_label)}</li>
</ul>
<h2>Per-case details</h2>
<table>
<tr><th>Case</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr>
{rows}
</table>
<p><em>This tool measures score differences between model versions; it does not
diagnose root causes. Statistical power depends on suite size.</em></p>
</body>
</html>
"""


def write_report(comp: SuiteComparison, path: str | Path, format: str = "markdown") -> Path:
    """Write a report to ``path``. ``format`` is ``"markdown"`` or ``"html"``."""
    path = Path(path)
    if format == "markdown":
        text = render_markdown(comp)
    elif format == "html":
        text = render_html(comp)
    else:
        raise ValueError(f"unknown format {format!r}; use 'markdown' or 'html'")
    path.write_text(text, encoding="utf-8")
    return path

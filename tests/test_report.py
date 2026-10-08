"""Tests for Markdown/HTML report rendering."""

from pathlib import Path

import pytest

from regression_detector.compare import compare_suites
from regression_detector.report import render_html, render_markdown, write_report
from regression_detector.runners import CaseResult, SuiteResult


def _comparison():
    base = SuiteResult(
        suite_name="demo",
        client_name="v1",
        results=[CaseResult(f"c{i}", f"c{i}", "", 1.0, True) for i in range(4)],
    )
    cand = SuiteResult(
        suite_name="demo",
        client_name="v2",
        results=[
            CaseResult("c0", "c0", "", 0.0, False),
            CaseResult("c1", "c1", "", 0.0, False),
            CaseResult("c2", "c2", "", 1.0, True),
            CaseResult("c3", "c3", "", 1.0, True),
        ],
    )
    return compare_suites(base, cand)


def test_markdown_contains_verdict_and_tables():
    md = render_markdown(_comparison())
    assert "# Model Regression Report: demo" in md
    assert "## Summary" in md
    assert "## Per-case details" in md
    assert "c0" in md and "c3" in md
    assert "McNemar" in md


def test_markdown_marks_no_regression_verdict():
    md = render_markdown(_comparison())
    # 2 regressions out of 4: chi2 = 0.5, p ~ 0.48 -> not significant
    assert "NO SIGNIFICANT REGRESSION" in md


def test_markdown_worst_regressions_section():
    md = render_markdown(_comparison())
    assert "## Worst regressions" in md  # 2 per-case regressions exist


def test_html_is_standalone_document():
    html = render_html(_comparison())
    assert "<!DOCTYPE html>" in html
    assert "Model Regression Report" in html
    assert "c0" in html


def test_write_report_markdown(tmp_path):
    p = write_report(_comparison(), tmp_path / "r.md", format="markdown")
    assert Path(p).read_text(encoding="utf-8").startswith("# Model Regression Report")


def test_write_report_html(tmp_path):
    p = write_report(_comparison(), tmp_path / "r.html", format="html")
    assert "<!DOCTYPE html>" in Path(p).read_text(encoding="utf-8")


def test_write_report_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        write_report(_comparison(), tmp_path / "r.txt", format="pdf")


def test_html_escapes_case_ids():
    comp = _comparison()
    comp.cases[0].case_id = "<script>alert(1)</script>"
    html = render_html(comp)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html

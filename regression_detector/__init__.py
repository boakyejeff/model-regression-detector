"""Model Regression Detector.

An offline eval harness that runs the same eval suite against two model
versions (baseline vs candidate) and flags statistically significant
regressions before they ship.

The package never makes network calls. To use a real model, implement the
``ModelClient`` protocol (see ``regression_detector.runners``) with your own
``generate(prompt) -> str`` method.
"""

from regression_detector.compare import (
    CaseComparison,
    SuiteComparison,
    compare_suites,
    mcnemar_test,
    paired_ttest,
    wilcoxon_signed_rank,
)
from regression_detector.evals import EvalCase, EvalSuite
from regression_detector.report import render_html, render_markdown, write_report
from regression_detector.runners import (
    CaseResult,
    MockModelClient,
    ModelClient,
    SuiteResult,
    run_suite,
)
from regression_detector.scorers import (
    CustomScorer,
    LLMJudge,
    ScoreResult,
    contains,
    exact_match,
    get_scorer,
    numeric_tolerance,
    regex_match,
)

__all__ = [
    "CaseComparison",
    "CaseResult",
    "CustomScorer",
    "EvalCase",
    "EvalSuite",
    "LLMJudge",
    "MockModelClient",
    "ModelClient",
    "ScoreResult",
    "SuiteComparison",
    "SuiteResult",
    "compare_suites",
    "contains",
    "exact_match",
    "get_scorer",
    "mcnemar_test",
    "numeric_tolerance",
    "paired_ttest",
    "regex_match",
    "render_html",
    "render_markdown",
    "run_suite",
    "wilcoxon_signed_rank",
    "write_report",
]

__version__ = "0.1.0"

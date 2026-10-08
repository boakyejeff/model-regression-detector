"""Scorers: turn a model response into a numeric score.

Built-in scorers (registered by name):
    - ``exact_match``: response equals ``scorer_params["expected"]``
      (after stripping whitespace; optional ``case_sensitive``).
    - ``contains``: response contains ``scorer_params["substring"]``.
    - ``regex_match``: ``re.search(scorer_params["pattern"], response)``.
    - ``numeric_tolerance``: first number found in the response is within
      ``scorer_params["tolerance"]`` of ``scorer_params["expected"]``.

Custom scoring:
    - :class:`CustomScorer` wraps any ``fn(response, case)`` returning a
      ``ScoreResult``, a float score, or a bool.
    - :class:`LLMJudge` is a stub interface for model-as-judge scoring. You
      supply the judge function; this package never calls any external API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Union

from regression_detector.evals import EvalCase

__all__ = [
    "ScoreResult",
    "exact_match",
    "contains",
    "regex_match",
    "numeric_tolerance",
    "CustomScorer",
    "LLMJudge",
    "get_scorer",
    "BUILTIN_SCORERS",
]

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class ScoreResult:
    """Outcome of scoring one response."""

    score: float  # numeric score, conventionally in [0, 1]
    passed: bool  # binary pass/fail derived from the score
    detail: str = ""  # human-readable explanation

    def __post_init__(self) -> None:
        self.score = float(self.score)
        self.passed = bool(self.passed)


def exact_match(response: str, case: EvalCase) -> ScoreResult:
    """Score 1.0 iff the stripped response equals ``expected``."""
    expected = str(case.scorer_params.get("expected", ""))
    case_sensitive = bool(case.scorer_params.get("case_sensitive", False))
    got, want = response.strip(), expected.strip()
    if not case_sensitive:
        got, want = got.lower(), want.lower()
    passed = got == want
    return ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        detail=f"exact_match expected={expected!r} got={response.strip()!r}",
    )


def contains(response: str, case: EvalCase) -> ScoreResult:
    """Score 1.0 iff ``substring`` appears in the response."""
    substring = str(case.scorer_params.get("substring", ""))
    case_sensitive = bool(case.scorer_params.get("case_sensitive", False))
    hay, needle = response, substring
    if not case_sensitive:
        hay, needle = hay.lower(), needle.lower()
    passed = needle in hay
    return ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        detail=f"contains substring={substring!r} found={passed}",
    )


def regex_match(response: str, case: EvalCase) -> ScoreResult:
    """Score 1.0 iff ``pattern`` matches anywhere in the response."""
    pattern = str(case.scorer_params.get("pattern", ""))
    try:
        matched = re.search(pattern, response) is not None
    except re.error as exc:
        return ScoreResult(score=0.0, passed=False, detail=f"invalid regex: {exc}")
    return ScoreResult(
        score=1.0 if matched else 0.0,
        passed=matched,
        detail=f"regex_match pattern={pattern!r} matched={matched}",
    )


def numeric_tolerance(response: str, case: EvalCase) -> ScoreResult:
    """Score 1.0 iff the first number in the response is within ``tolerance``.

    Expects ``scorer_params`` ``expected`` (float) and ``tolerance`` (float,
    default 0.0). Score is 0.0 when no number can be parsed.
    """
    try:
        expected = float(case.scorer_params["expected"])
    except (KeyError, TypeError, ValueError):
        return ScoreResult(score=0.0, passed=False, detail="missing/invalid 'expected'")
    tolerance = float(case.scorer_params.get("tolerance", 0.0))
    match = _NUMBER_RE.search(response)
    if not match:
        return ScoreResult(score=0.0, passed=False, detail="no number found in response")
    value = float(match.group(0))
    passed = abs(value - expected) <= tolerance
    return ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        detail=f"numeric_tolerance expected={expected} got={value} tol={tolerance}",
    )


ScorerFn = Callable[[str, EvalCase], Union[ScoreResult, float, bool]]


class CustomScorer:
    """Wrap a user-supplied ``fn(response, case)`` as a scorer.

    ``fn`` may return a :class:`ScoreResult`, a float score in [0, 1], or a
    bool (True -> 1.0/pass, False -> 0.0/fail).
    """

    def __init__(self, fn: ScorerFn, name: str = "custom"):
        if not callable(fn):
            raise ValueError("CustomScorer requires a callable")
        self.fn = fn
        self.name = name

    def __call__(self, response: str, case: EvalCase) -> ScoreResult:
        result = self.fn(response, case)
        if isinstance(result, ScoreResult):
            return result
        if isinstance(result, bool):
            return ScoreResult(score=1.0 if result else 0.0, passed=result)
        score = float(result)
        return ScoreResult(score=score, passed=score >= 0.5, detail=f"custom {self.name}")


class LLMJudge:
    """Stub interface for model-as-judge scoring.

    You supply ``judge_fn(response, case) -> float`` (a score in [0, 1]).
    This package never calls any external API: if no judge function is
    supplied, calling the scorer raises :class:`RuntimeError` instead of
    silently phoning home.
    """

    def __init__(
        self,
        judge_fn: Callable[[str, EvalCase], float] | None = None,
        pass_threshold: float = 0.5,
    ):
        self.judge_fn = judge_fn
        self.pass_threshold = float(pass_threshold)

    def __call__(self, response: str, case: EvalCase) -> ScoreResult:
        if self.judge_fn is None:
            raise RuntimeError(
                "LLMJudge has no judge function. Supply judge_fn(response, case) -> float; "
                "this package never calls external APIs on its own."
            )
        score = float(self.judge_fn(response, case))
        return ScoreResult(
            score=score,
            passed=score >= self.pass_threshold,
            detail="llm_judge",
        )


BUILTIN_SCORERS = {
    "exact_match": exact_match,
    "contains": contains,
    "regex_match": regex_match,
    "numeric_tolerance": numeric_tolerance,
}


def get_scorer(name: str) -> Callable[[str, EvalCase], ScoreResult]:
    """Return the built-in scorer registered under ``name``."""
    try:
        return BUILTIN_SCORERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown scorer {name!r}. Built-ins: {sorted(BUILTIN_SCORERS)}"
        ) from None

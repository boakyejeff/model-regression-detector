"""Tests for built-in scorers, CustomScorer, and the LLMJudge stub."""

import pytest

from regression_detector.evals import EvalCase
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


def make_case(**params):
    return EvalCase(id="t1", prompt="p", scorer="exact_match", scorer_params=params)


def test_exact_match_pass():
    r = exact_match("Paris", make_case(expected="Paris"))
    assert r.passed and r.score == 1.0


def test_exact_match_strips_whitespace():
    r = exact_match("  Paris\n", make_case(expected="Paris"))
    assert r.passed


def test_exact_match_case_insensitive_by_default():
    r = exact_match("paris", make_case(expected="Paris"))
    assert r.passed


def test_exact_match_case_sensitive_option():
    r = exact_match("paris", make_case(expected="Paris", case_sensitive=True))
    assert not r.passed and r.score == 0.0


def test_exact_match_fail():
    r = exact_match("London", make_case(expected="Paris"))
    assert not r.passed


def test_contains():
    assert contains("I love Python programming", make_case(substring="python")).passed
    assert not contains("I love Java", make_case(substring="python")).passed


def test_contains_case_sensitive():
    assert not contains("Python", make_case(substring="python", case_sensitive=True)).passed


def test_regex_match():
    assert regex_match("call (555) 123-4567", make_case(pattern=r"\(\d{3}\) \d{3}-\d{4}")).passed
    assert not regex_match("no phone here", make_case(pattern=r"\(\d{3}\)")).passed


def test_regex_match_invalid_pattern():
    r = regex_match("anything", make_case(pattern="(["))
    assert not r.passed and "invalid regex" in r.detail


def test_numeric_tolerance_pass():
    r = numeric_tolerance("The answer is 30.2", make_case(expected=30, tolerance=0.5))
    assert r.passed


def test_numeric_tolerance_fail():
    r = numeric_tolerance("The answer is 35", make_case(expected=30, tolerance=0.5))
    assert not r.passed


def test_numeric_tolerance_no_number():
    r = numeric_tolerance("no numbers here", make_case(expected=30, tolerance=0.5))
    assert not r.passed


def test_numeric_tolerance_missing_expected():
    r = numeric_tolerance("42", make_case())
    assert not r.passed


def test_get_scorer_registry():
    assert get_scorer("exact_match") is exact_match
    assert get_scorer("contains") is contains
    assert get_scorer("regex_match") is regex_match
    assert get_scorer("numeric_tolerance") is numeric_tolerance
    with pytest.raises(KeyError):
        get_scorer("nope")


def test_custom_scorer_with_float():
    s = CustomScorer(lambda resp, case: 0.8)
    r = s("x", make_case())
    assert r.score == 0.8 and r.passed


def test_custom_scorer_with_bool():
    s = CustomScorer(lambda resp, case: resp == "yes")
    assert s("yes", make_case()).passed
    assert not s("no", make_case()).passed


def test_custom_scorer_with_score_result():
    s = CustomScorer(lambda resp, case: ScoreResult(score=0.2, passed=False, detail="d"))
    r = s("x", make_case())
    assert r.score == 0.2 and not r.passed and r.detail == "d"


def test_custom_scorer_rejects_non_callable():
    with pytest.raises(ValueError):
        CustomScorer("not-callable")


def test_llm_judge_requires_judge_fn():
    judge = LLMJudge()
    with pytest.raises(RuntimeError, match="no judge function"):
        judge("response", make_case())


def test_llm_judge_with_supplied_fn():
    judge = LLMJudge(judge_fn=lambda resp, case: 0.9, pass_threshold=0.5)
    r = judge("response", make_case())
    assert r.score == 0.9 and r.passed
    judge2 = LLMJudge(judge_fn=lambda resp, case: 0.2)
    assert not judge2("response", make_case()).passed

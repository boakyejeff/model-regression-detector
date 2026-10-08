"""Tests for runners: MockModelClient, run_suite, retries, timeouts."""

import time

import pytest

from regression_detector.evals import EvalCase, EvalSuite
from regression_detector.runners import MockModelClient, align_results, run_suite


def tiny_suite():
    return EvalSuite(
        name="tiny",
        cases=[
            EvalCase("c1", "capital of France?", "exact_match", {"expected": "Paris"}),
            EvalCase("c2", "2+2?", "numeric_tolerance", {"expected": 4, "tolerance": 0}),
        ],
    )


def test_mock_client_dict_lookup():
    client = MockModelClient({"a": "b"}, name="m")
    assert client.generate("a") == "b"
    assert client.generate("missing") == "I don't know."


def test_mock_client_callable():
    client = MockModelClient(lambda prompt: prompt.upper())
    assert client.generate("hi") == "HI"


def test_mock_client_records_calls():
    client = MockModelClient({"a": "b"})
    client.generate("a")
    assert client.calls == ["a"]


def test_run_suite_all_pass():
    client = MockModelClient(
        {"capital of France?": "Paris", "2+2?": "4"}, name="good"
    )
    res = run_suite(client, tiny_suite())
    assert res.client_name == "good"
    assert res.pass_rate == 1.0
    assert res.mean_score == 1.0
    assert res.n_errors == 0
    assert all(r.error is None for r in res.results)


def test_run_suite_scores_failures():
    client = MockModelClient({"capital of France?": "London", "2+2?": "5"})
    res = run_suite(client, tiny_suite())
    assert res.pass_rate == 0.0
    assert all(not r.passed for r in res.results)


def test_run_suite_retries_flaky_client():
    client = MockModelClient(
        {"capital of France?": "Paris", "2+2?": "4"},
        error_prompts={"capital of France?": 1},  # fails once, then succeeds
    )
    res = run_suite(client, tiny_suite(), max_retries=2)
    assert res.n_errors == 0
    by_id = {r.case_id: r for r in res.results}
    assert by_id["c1"].passed
    assert by_id["c1"].attempts == 2


def test_run_suite_records_persistent_errors():
    client = MockModelClient({}, error_prompts={"capital of France?": 99})
    res = run_suite(client, tiny_suite(), max_retries=1)
    by_id = {r.case_id: r for r in res.results}
    assert by_id["c1"].error is not None
    assert by_id["c1"].score == 0.0
    assert by_id["c1"].attempts == 2
    assert res.n_errors == 1


def test_run_suite_timeout():
    def slow(prompt):
        time.sleep(5)
        return "too late"

    client = MockModelClient(slow)
    res = run_suite(client, tiny_suite(), timeout=0.2, max_retries=0)
    assert res.n_errors == 2
    assert all("Timeout" in (r.error or "") for r in res.results)


def test_scorer_overrides():
    client = MockModelClient({"capital of France?": "definitely Paris", "2+2?": "4"})
    res = run_suite(
        client,
        tiny_suite(),
        scorer_overrides={"exact_match": lambda resp, case: __import__(
            "regression_detector.scorers", fromlist=["ScoreResult"]
        ).ScoreResult(1.0, True)},
    )
    assert res.pass_rate == 1.0


def test_align_results_pairs_by_id():
    good = MockModelClient({"capital of France?": "Paris", "2+2?": "4"}, name="a")
    bad = MockModelClient({"capital of France?": "London", "2+2?": "4"}, name="b")
    ra, rb = run_suite(good, tiny_suite()), run_suite(bad, tiny_suite())
    pairs = align_results(ra, rb)
    assert [b.case_id for b, _ in pairs] == ["c1", "c2"]
    assert pairs[0][0].passed and not pairs[0][1].passed


def test_align_results_missing_case_raises():
    good = MockModelClient({}, name="a")
    ra = run_suite(good, tiny_suite())
    rb = run_suite(good, EvalSuite(name="other", cases=[EvalCase("zzz", "p")]))
    with pytest.raises(ValueError, match="missing case id"):
        align_results(ra, rb)

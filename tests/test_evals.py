"""Tests for EvalCase / EvalSuite loading and validation."""

import json

import pytest
import yaml

from regression_detector.evals import EvalCase, EvalSuite


def test_eval_case_defaults():
    c = EvalCase(id="a", prompt="hello")
    assert c.scorer == "exact_match"
    assert c.scorer_params == {}
    assert c.metadata == {}


def test_eval_case_rejects_bad_input():
    with pytest.raises(ValueError):
        EvalCase(id="", prompt="x")
    with pytest.raises(ValueError):
        EvalCase(id="a", prompt=123)  # type: ignore[arg-type]


def test_eval_case_from_dict():
    c = EvalCase.from_dict(
        {"id": "m1", "prompt": "2+2?", "scorer": "numeric_tolerance",
         "scorer_params": {"expected": 4, "tolerance": 0.1},
         "metadata": {"cat": "math"}}
    )
    assert c.scorer == "numeric_tolerance"
    assert c.scorer_params["expected"] == 4
    assert c.metadata["cat"] == "math"


def test_eval_case_from_dict_requires_id_and_prompt():
    with pytest.raises(ValueError):
        EvalCase.from_dict({"id": "x"})


def test_suite_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate case ids"):
        EvalSuite(name="s", cases=[EvalCase("a", "p1"), EvalCase("a", "p2")])


def test_suite_from_yaml(tmp_path):
    data = {
        "name": "yaml-suite",
        "cases": [
            {"id": "c1", "prompt": "hi", "scorer": "contains",
             "scorer_params": {"substring": "hi"}},
            {"id": "c2", "prompt": "42?", "scorer": "numeric_tolerance",
             "scorer_params": {"expected": 42, "tolerance": 0}},
        ],
    }
    p = tmp_path / "suite.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    suite = EvalSuite.from_yaml(p)
    assert suite.name == "yaml-suite"
    assert len(suite) == 2
    assert suite.cases[0].scorer == "contains"
    assert suite.cases[1].scorer_params["expected"] == 42


def test_suite_from_json(tmp_path):
    data = {"name": "json-suite",
            "cases": [{"id": "c1", "prompt": "hi", "scorer": "exact_match",
                       "scorer_params": {"expected": "hi"}}]}
    p = tmp_path / "suite.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    suite = EvalSuite.from_json(p)
    assert suite.name == "json-suite"
    assert suite.cases[0].id == "c1"


def test_suite_from_dict_requires_cases():
    with pytest.raises(ValueError):
        EvalSuite.from_dict({"name": "empty"})


def test_suite_roundtrip():
    suite = EvalSuite(name="rt", cases=[EvalCase("a", "p", "contains", {"substring": "p"})])
    d = suite.to_dict()
    suite2 = EvalSuite.from_dict(d)
    assert suite2.name == "rt"
    assert suite2.cases[0].scorer_params == {"substring": "p"}

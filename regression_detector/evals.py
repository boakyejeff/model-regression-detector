"""Eval cases and suites.

An :class:`EvalCase` is a single prompt paired with the name of a scorer
(plus that scorer's parameters). An :class:`EvalSuite` is an ordered
collection of cases, loadable from YAML or JSON so evals can live in
version-controlled files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass
class EvalCase:
    """One evaluation item.

    Attributes:
        id: Unique identifier within a suite (used to align baseline and
            candidate runs for paired comparison).
        prompt: The prompt sent to the model.
        scorer: Name of a built-in scorer (see ``regression_detector.scorers``)
            or a custom scorer key resolved by the caller.
        scorer_params: Parameters forwarded to the scorer, e.g.
            ``{"expected": "Paris"}`` for ``exact_match`` or
            ``{"expected": 3.14, "tolerance": 0.01}`` for
            ``numeric_tolerance``.
        metadata: Free-form dict (category, difficulty, source, ...) carried
            through to reports.
    """

    id: str
    prompt: str
    scorer: str = "exact_match"
    scorer_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("EvalCase id must be a non-empty string")
        if not isinstance(self.prompt, str):
            raise ValueError("EvalCase prompt must be a string")
        if not isinstance(self.scorer, str):
            raise ValueError("EvalCase scorer must be a scorer name string")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalCase":
        """Build an EvalCase from a plain dict (e.g. parsed YAML/JSON)."""
        if "id" not in data or "prompt" not in data:
            raise ValueError("EvalCase dict requires 'id' and 'prompt' keys")
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            scorer=str(data.get("scorer", "exact_match")),
            scorer_params=dict(data.get("scorer_params", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "scorer": self.scorer,
            "scorer_params": dict(self.scorer_params),
            "metadata": dict(self.metadata),
        }


@dataclass
class EvalSuite:
    """An ordered collection of eval cases."""

    name: str
    cases: Sequence[EvalCase] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EvalSuite name must be non-empty")
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate case ids in suite: {dupes}")

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalSuite":
        if "cases" not in data:
            raise ValueError("EvalSuite dict requires a 'cases' list")
        return cls(
            name=str(data.get("name", "unnamed-suite")),
            cases=[EvalCase.from_dict(c) for c in data["cases"]],
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalSuite":
        """Load a suite from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalSuite":
        """Load a suite from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "cases": [c.to_dict() for c in self.cases],
        }

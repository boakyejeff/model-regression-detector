"""Suite runners and model clients.

A :class:`ModelClient` is anything with ``generate(prompt) -> str``. Plug in
a real model by implementing that one method. :class:`MockModelClient`
provides scripted, fully offline responses for demos and tests, and
:func:`run_suite` executes a suite with retries and optional per-call
timeouts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence, Union

from regression_detector.evals import EvalCase, EvalSuite
from regression_detector.scorers import ScoreResult, get_scorer


class ModelClient(Protocol):
    """Anything that can generate a response string for a prompt."""

    def generate(self, prompt: str) -> str:
        """Return the model's response text for ``prompt``."""
        ...


class MockModelClient:
    """Scripted, offline model client for demos and tests.

    Args:
        responses: Mapping of prompt -> response, or a callable taking the
            prompt and returning a response string.
        name: Label used in reports.
        default: Response used when a prompt has no scripted answer.
        error_prompts: Prompts that raise ``RuntimeError`` (simulates a
            failing model call). Pass a dict prompt -> failures-before-success
            to simulate flaky calls that succeed on retry.
    """

    def __init__(
        self,
        responses: Union[Mapping[str, str], Callable[[str], str]],
        name: str = "mock",
        default: str = "I don't know.",
        error_prompts: Mapping[str, int] | None = None,
    ):
        self.responses = responses
        self.name = name
        self.default = default
        self._remaining_failures = dict(error_prompts or {})
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        remaining = self._remaining_failures.get(prompt, 0)
        if remaining > 0:
            self._remaining_failures[prompt] = remaining - 1
            raise RuntimeError(f"MockModelClient simulated failure for prompt {prompt!r}")
        if callable(self.responses):
            return str(self.responses(prompt))
        return str(self.responses.get(prompt, self.default))


@dataclass
class CaseResult:
    """The scored outcome of one case against one client."""

    case_id: str
    prompt: str
    response: str
    score: float
    passed: bool
    detail: str = ""
    error: str | None = None
    attempts: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class SuiteResult:
    """All case results for one client on one suite."""

    suite_name: str
    client_name: str
    results: list[CaseResult] = field(default_factory=list)

    def scores_by_id(self) -> dict[str, float]:
        return {r.case_id: r.score for r in self.results}

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if r.error is not None)


def _call_with_timeout(fn: Callable[[], str], timeout: float | None) -> str:
    if timeout is None:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"generate() exceeded timeout of {timeout}s")


def run_suite(
    client: ModelClient,
    suite: EvalSuite,
    *,
    scorer_overrides: Mapping[str, Callable[[str, EvalCase], ScoreResult]] | None = None,
    max_retries: int = 2,
    timeout: float | None = None,
) -> SuiteResult:
    """Execute every case in ``suite`` against ``client``.

    Args:
        client: A :class:`ModelClient`.
        suite: The eval suite to run.
        scorer_overrides: Optional mapping of scorer name -> callable,
            replacing the built-in with that name for this run.
        max_retries: Retries after the first failed ``generate()`` call.
        timeout: Per-call timeout in seconds for ``generate()``; ``None``
            disables timeouts.

    Returns:
        A :class:`SuiteResult`. Failed calls (after retries) are recorded
        with ``error`` set and a score of 0.0 rather than raising.
    """
    overrides = dict(scorer_overrides or {})
    client_name = getattr(client, "name", type(client).__name__)
    results: list[CaseResult] = []

    for case in suite:
        scorer = overrides.get(case.scorer) or get_scorer(case.scorer)
        response, error, attempts = "", None, 0
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            try:
                response = _call_with_timeout(lambda: client.generate(case.prompt), timeout)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed silently
                error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            results.append(
                CaseResult(
                    case_id=case.id,
                    prompt=case.prompt,
                    response="",
                    score=0.0,
                    passed=False,
                    detail="generate() failed",
                    error=error,
                    attempts=attempts,
                    metadata=dict(case.metadata),
                )
            )
            continue
        scored = scorer(response, case)
        results.append(
            CaseResult(
                case_id=case.id,
                prompt=case.prompt,
                response=response,
                score=scored.score,
                passed=scored.passed,
                detail=scored.detail,
                attempts=attempts,
                metadata=dict(case.metadata),
            )
        )
    return SuiteResult(suite_name=suite.name, client_name=client_name, results=results)


def align_results(
    baseline: SuiteResult, candidate: SuiteResult
) -> list[tuple[CaseResult, CaseResult]]:
    """Pair baseline and candidate results by case id (in baseline order)."""
    cand_by_id = {r.case_id: r for r in candidate.results}
    pairs = []
    for b in baseline.results:
        c = cand_by_id.get(b.case_id)
        if c is None:
            raise ValueError(f"candidate run missing case id {b.case_id!r}")
        pairs.append((b, c))
    return pairs

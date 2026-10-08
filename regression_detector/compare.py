"""Paired statistical comparison of baseline vs candidate eval scores.

Binary (0/1) scores use McNemar's test on the discordant pairs:
    b = baseline pass / candidate fail (regressions)
    c = baseline fail / candidate pass (improvements)
Continuous scores use a paired t-test (default) or the Wilcoxon signed-rank
test. Implementations are pure Python (no hard dependency on scipy); when
scipy is importable it is used for the p-values instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

try:  # optional: used for p-values when available, never required
    from scipy import stats as _sp_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _sp_stats = None  # type: ignore[assignment]
    _HAS_SCIPY = False

from regression_detector.runners import SuiteResult, align_results


# ---------------------------------------------------------------------------
# Pure-Python distribution helpers
# ---------------------------------------------------------------------------


def _normal_sf(x: float) -> float:
    """Survival function P(Z > x) for the standard normal."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iter, eps, fpmin = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        raise ValueError("betai requires a, b > 0")
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta) / a
        return front * _betacf(a, b, x)
    front = math.exp(b * math.log(1.0 - x) + a * math.log(x) - log_beta) / b
    return 1.0 - front * _betacf(b, a, 1.0 - x)


def _mcnemar_pure(b: int, c: int) -> tuple[float, float]:
    """McNemar chi-square (continuity corrected) and two-sided p-value."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    p_value = math.erfc(math.sqrt(chi2 / 2.0))  # chi-square(1) tail
    return chi2, p_value


def _paired_ttest_pure(diffs: Sequence[float]) -> tuple[float, float]:
    """Paired t statistic and two-sided p-value (pure Python)."""
    n = len(diffs)
    if n < 2:
        raise ValueError("paired t-test needs at least 2 observations")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0.0:
        # No spread: a nonzero constant shift is a certain difference,
        # an all-zero diff is no evidence either way.
        if mean == 0.0:
            return 0.0, 1.0
        return math.copysign(math.inf, mean), 0.0
    t = mean / math.sqrt(var / n)
    nu = n - 1
    # Two-sided p for |T| > |t|: I_{nu/(nu+t^2)}(nu/2, 1/2)
    x = nu / (nu + t * t)
    p_value = _betai(nu / 2.0, 0.5, x)
    return t, min(max(p_value, 0.0), 1.0)


def _wilcoxon_pure(diffs: Sequence[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank statistic (W+) and two-sided p-value.

    Uses the normal approximation with continuity and tie corrections,
    matching ``scipy.stats.wilcoxon(..., method="approx", correction=True)``.
    """
    nz = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    order = sorted(range(n), key=lambda i: nz[i][0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and nz[order[j + 1]][0] == nz[order[i]][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(r for ( _, s), r in zip(nz, ranks) if s > 0)
    mean_w = n * (n + 1) / 4.0
    # tie correction: sum over tie groups of (t^3 - t) / 48
    tie_corr = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and nz[order[j + 1]][0] == nz[order[i]][0]:
            j += 1
        tsize = j - i + 1
        if tsize > 1:
            tie_corr += (tsize**3 - tsize) / 48.0
        i = j + 1
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var_w <= 0:
        return w_plus, 1.0
    z = (w_plus - mean_w - 0.5 * math.copysign(1.0, w_plus - mean_w)) / math.sqrt(var_w)
    p_value = 2.0 * _normal_sf(abs(z))
    return w_plus, min(max(p_value, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Public test functions (scipy when available, pure-Python fallback)
# ---------------------------------------------------------------------------


def mcnemar_test(
    baseline_scores: Sequence[float], candidate_scores: Sequence[float]
) -> tuple[float, float]:
    """McNemar's test for paired binary scores.

    Returns (chi2_statistic, p_value). ``b`` counts baseline-pass/candidate-fail
    pairs (regressions), ``c`` counts the reverse (improvements).
    """
    b = sum(1 for x, y in zip(baseline_scores, candidate_scores) if x == 1.0 and y == 0.0)
    c = sum(1 for x, y in zip(baseline_scores, candidate_scores) if x == 0.0 and y == 1.0)
    if _HAS_SCIPY:
        chi2, _ = _mcnemar_pure(b, c)
        p_value = float(_sp_stats.chi2.sf(chi2, 1))
        return chi2, p_value
    return _mcnemar_pure(b, c)


def paired_ttest(
    baseline_scores: Sequence[float], candidate_scores: Sequence[float]
) -> tuple[float, float]:
    """Paired t-test on (candidate - baseline) differences.

    Returns (t_statistic, two-sided p_value).
    """
    diffs = [y - x for x, y in zip(baseline_scores, candidate_scores)]
    if _HAS_SCIPY:
        res = _sp_stats.ttest_rel(candidate_scores, baseline_scores)
        t = float(res.statistic)
        p = float(res.pvalue)
        if math.isnan(t):
            return 0.0, 1.0
        return t, p
    return _paired_ttest_pure(diffs)


def wilcoxon_signed_rank(
    baseline_scores: Sequence[float], candidate_scores: Sequence[float]
) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired differences.

    Returns (statistic, two-sided p_value).
    """
    diffs = [y - x for x, y in zip(baseline_scores, candidate_scores)]
    if _HAS_SCIPY:
        nz = [d for d in diffs if d != 0.0]
        if not nz:
            return 0.0, 1.0
        res = _sp_stats.wilcoxon(
            [y for x, y in zip(baseline_scores, candidate_scores) if y - x != 0.0],
            [x for x, y in zip(baseline_scores, candidate_scores) if y - x != 0.0],
            correction=True,  # match the pure-Python path's continuity correction
        )
        return float(res.statistic), float(res.pvalue)
    return _wilcoxon_pure(diffs)


def cohens_dz(baseline_scores: Sequence[float], candidate_scores: Sequence[float]) -> float:
    """Standardized mean difference of paired scores (candidate - baseline)."""
    diffs = [y - x for x, y in zip(baseline_scores, candidate_scores)]
    n = len(diffs)
    if n < 2:
        return 0.0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0.0:
        return 0.0
    return mean / math.sqrt(var)


# ---------------------------------------------------------------------------
# Suite comparison
# ---------------------------------------------------------------------------


@dataclass
class CaseComparison:
    case_id: str
    baseline_score: float
    candidate_score: float
    delta: float  # candidate - baseline
    regression: bool  # candidate strictly worse than baseline
    improvement: bool  # candidate strictly better than baseline
    metadata: dict = field(default_factory=dict)


@dataclass
class SuiteComparison:
    suite_name: str
    baseline_name: str
    candidate_name: str
    n_cases: int
    mean_baseline: float
    mean_candidate: float
    mean_delta: float
    n_regressions: int
    n_improvements: int
    n_unchanged: int
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    effect_size: float
    effect_label: str
    verdict: str  # "REGRESSION DETECTED" or "NO SIGNIFICANT REGRESSION"
    cases: list[CaseComparison] = field(default_factory=list)

    @property
    def regression_detected(self) -> bool:
        return self.verdict == "REGRESSION DETECTED"


def _is_binary(scores: Sequence[float]) -> bool:
    return all(s in (0.0, 1.0) for s in scores)


def compare_suites(
    baseline: SuiteResult,
    candidate: SuiteResult,
    *,
    alpha: float = 0.05,
    method: str = "auto",
) -> SuiteComparison:
    """Paired comparison of two runs of the same suite.

    Args:
        baseline: Reference run (e.g. the model currently in production).
        candidate: Challenger run (e.g. the new model version).
        alpha: Significance level for the regression verdict.
        method: ``"auto"`` (McNemar for binary scores, paired t-test
            otherwise), ``"mcnemar"``, ``"ttest"``, or ``"wilcoxon"``.

    A regression verdict requires BOTH statistical significance (p < alpha)
    AND the candidate to be worse on average — a significant *improvement*
    is never flagged as a regression.
    """
    pairs = align_results(baseline, candidate)
    b_scores = [b.score for b, _ in pairs]
    c_scores = [c.score for _, c in pairs]

    cases = [
        CaseComparison(
            case_id=b.case_id,
            baseline_score=b.score,
            candidate_score=c.score,
            delta=c.score - b.score,
            regression=c.score < b.score,
            improvement=c.score > b.score,
            metadata=dict(b.metadata),
        )
        for b, c in pairs
    ]

    binary = _is_binary(b_scores) and _is_binary(c_scores)
    if method == "auto":
        method = "mcnemar" if binary else "ttest"

    if method == "mcnemar":
        if not binary:
            raise ValueError("mcnemar requires binary (0/1) scores")
        test_name = "McNemar's test (continuity corrected)"
        statistic, p_value = mcnemar_test(b_scores, c_scores)
        b_discord = sum(1 for cc in cases if cc.baseline_score == 1.0 and cc.candidate_score == 0.0)
        c_discord = sum(1 for cc in cases if cc.baseline_score == 0.0 and cc.candidate_score == 1.0)
        effect_size = (b_discord / c_discord) if c_discord else (float("inf") if b_discord else 1.0)
        effect_label = f"discordant odds ratio (regressions/improvements) = {effect_size:.3f}"
        worse = b_discord > c_discord
    elif method == "ttest":
        test_name = "paired t-test"
        statistic, p_value = paired_ttest(b_scores, c_scores)
        effect_size = cohens_dz(b_scores, c_scores)
        effect_label = f"Cohen's dz = {effect_size:.3f}"
        worse = sum(c.delta for c in cases) / len(cases) < 0 if cases else False
    elif method == "wilcoxon":
        test_name = "Wilcoxon signed-rank test"
        statistic, p_value = wilcoxon_signed_rank(b_scores, c_scores)
        effect_size = cohens_dz(b_scores, c_scores)
        effect_label = f"Cohen's dz = {effect_size:.3f}"
        worse = sum(c.delta for c in cases) / len(cases) < 0 if cases else False
    else:
        raise ValueError(f"unknown method {method!r}; use auto/mcnemar/ttest/wilcoxon")

    significant = bool(p_value < alpha)
    regression_detected = significant and worse
    verdict = "REGRESSION DETECTED" if regression_detected else "NO SIGNIFICANT REGRESSION"

    n = len(cases)
    return SuiteComparison(
        suite_name=baseline.suite_name,
        baseline_name=baseline.client_name,
        candidate_name=candidate.client_name,
        n_cases=n,
        mean_baseline=sum(b_scores) / n if n else 0.0,
        mean_candidate=sum(c_scores) / n if n else 0.0,
        mean_delta=(sum(c_scores) - sum(b_scores)) / n if n else 0.0,
        n_regressions=sum(1 for c in cases if c.regression),
        n_improvements=sum(1 for c in cases if c.improvement),
        n_unchanged=sum(1 for c in cases if not c.regression and not c.improvement),
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        significant=significant,
        effect_size=effect_size,
        effect_label=effect_label,
        verdict=verdict,
        cases=cases,
    )

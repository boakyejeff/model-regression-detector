"""Known-answer tests for the paired comparison statistics."""

import math

import pytest

from regression_detector.compare import (
    CaseComparison,
    _betai,
    _mcnemar_pure,
    _paired_ttest_pure,
    _wilcoxon_pure,
    cohens_dz,
    compare_suites,
    mcnemar_test,
    paired_ttest,
    wilcoxon_signed_rank,
)
from regression_detector.runners import CaseResult, SuiteResult

scipy_stats = pytest.importorskip("scipy.stats")


def _suite_result(name, scores, ids=None):
    ids = ids or [f"c{i}" for i in range(len(scores))]
    return SuiteResult(
        suite_name="s",
        client_name=name,
        results=[
            CaseResult(case_id=i, prompt=i, response="", score=s,
                       passed=s >= 0.5)
            for i, s in zip(ids, scores)
        ],
    )


# --- incomplete beta sanity -------------------------------------------------


def test_betai_uniform():
    assert _betai(1.0, 1.0, 0.3) == pytest.approx(0.3, abs=1e-10)


def test_betai_known_value():
    # I_x(2, 1) = x^2
    assert _betai(2.0, 1.0, 0.5) == pytest.approx(0.25, abs=1e-10)


def test_betai_edges():
    assert _betai(2.0, 3.0, 0.0) == 0.0
    assert _betai(2.0, 3.0, 1.0) == 1.0


# --- McNemar -----------------------------------------------------------------


def test_mcnemar_pure_known_answer():
    # b=8 regressions, c=0 improvements: chi2 = 49/8 = 6.125
    chi2, p = _mcnemar_pure(8, 0)
    assert chi2 == pytest.approx(6.125, abs=1e-12)
    assert p == pytest.approx(float(scipy_stats.chi2.sf(6.125, 1)), abs=1e-12)
    assert p < 0.05


def test_mcnemar_pure_8_vs_1():
    # chi2 = (|8-1|-1)^2 / 9 = 4.0
    chi2, p = _mcnemar_pure(8, 1)
    assert chi2 == pytest.approx(4.0, abs=1e-12)
    assert p == pytest.approx(0.04550026389635842, abs=1e-9)


def test_mcnemar_pure_no_discordant():
    assert _mcnemar_pure(0, 0) == (0.0, 1.0)


def test_mcnemar_test_counts_discordant_pairs():
    baseline = [1.0] * 16
    candidate = [1.0] * 8 + [0.0] * 8
    chi2, p = mcnemar_test(baseline, candidate)
    assert chi2 == pytest.approx(6.125, abs=1e-9)
    assert p < 0.05


# --- paired t-test ------------------------------------------------------------


def test_paired_ttest_pure_matches_scipy():
    diffs = [0.5, -0.2, 0.8, 0.1, -0.4, 0.6, 0.3, -0.1]
    baseline = [0.0] * len(diffs)
    candidate = list(diffs)
    t_pure, p_pure = _paired_ttest_pure(diffs)
    res = scipy_stats.ttest_rel(candidate, baseline)
    assert t_pure == pytest.approx(float(res.statistic), abs=1e-9)
    assert p_pure == pytest.approx(float(res.pvalue), abs=1e-9)


def test_paired_ttest_pure_known_answer():
    # diffs 2..10 step 2: mean 6, sd^2 = 10 -> t = 6 / sqrt(10/5) = 4.242640...
    diffs = [2.0, 4.0, 6.0, 8.0, 10.0]
    t, p = _paired_ttest_pure(diffs)
    assert t == pytest.approx(4.242640687119285, abs=1e-9)
    assert p == pytest.approx(float(scipy_stats.t.sf(abs(t), 4) * 2), abs=1e-9)


def test_paired_ttest_zero_variance():
    t, p = _paired_ttest_pure([0.0, 0.0, 0.0])
    assert (t, p) == (0.0, 1.0)


def test_paired_ttest_constant_shift_is_certain():
    t, p = _paired_ttest_pure([-0.5, -0.5, -0.5, -0.5])
    assert t == float("-inf") and p == 0.0


def test_paired_ttest_needs_two_observations():
    with pytest.raises(ValueError):
        _paired_ttest_pure([1.0])


def test_paired_ttest_public_fn():
    t, p = paired_ttest([0.5, 0.5, 0.5, 0.5], [0.9, 0.8, 0.9, 0.7])
    assert p < 0.05 and t > 0


# --- Wilcoxon -----------------------------------------------------------------


def test_wilcoxon_pure_matches_scipy():
    diffs = [0.5, -0.2, 0.8, 0.1, -0.4, 0.6, 0.3, -0.1, 0.0, 0.45]
    baseline = [0.0] * len(diffs)
    candidate = list(diffs)
    _, p_pure = _wilcoxon_pure(diffs)
    nz_c = [c for c, d in zip(candidate, diffs) if d != 0.0]
    nz_b = [b for b, d in zip(baseline, diffs) if d != 0.0]
    # method="approx": both use the normal approximation with continuity
    # correction (scipy's default "auto" uses the exact distribution for n<=50,
    # and its approx default correction=False differs from ours)
    res = scipy_stats.wilcoxon(nz_c, nz_b, method="approx", correction=True)
    assert p_pure == pytest.approx(float(res.pvalue), abs=1e-9)


def test_wilcoxon_all_zero():
    assert _wilcoxon_pure([0.0, 0.0]) == (0.0, 1.0)


def test_wilcoxon_public_fn():
    _, p = wilcoxon_signed_rank([0.1] * 10, [0.9] * 10)
    assert p < 0.05


# --- effect size ----------------------------------------------------------------


def test_cohens_dz_zero_when_no_variance():
    assert cohens_dz([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_cohens_dz_sign():
    # varying diffs with negative mean -> negative effect
    assert cohens_dz([1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]) < 0
    assert cohens_dz([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]) == 0.0  # zero variance


# --- compare_suites ------------------------------------------------------------


def test_compare_detects_clear_regression():
    base = _suite_result("v1", [1.0] * 16)
    cand = _suite_result("v2", [1.0] * 8 + [0.0] * 8)
    comp = compare_suites(base, cand, alpha=0.05)
    assert comp.verdict == "REGRESSION DETECTED"
    assert comp.regression_detected
    assert comp.n_regressions == 8
    assert comp.n_improvements == 0
    assert comp.p_value < 0.05
    assert comp.mean_delta == pytest.approx(-0.5)
    assert "McNemar" in comp.test_name


def test_compare_no_change_not_flagged():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 3
    comp = compare_suites(_suite_result("v1", scores), _suite_result("v2", list(scores)))
    assert comp.verdict == "NO SIGNIFICANT REGRESSION"
    assert comp.p_value == pytest.approx(1.0)
    assert comp.n_regressions == 0


def test_compare_significant_improvement_is_not_a_regression():
    base = _suite_result("v1", [1.0] * 8 + [0.0] * 8)
    cand = _suite_result("v2", [1.0] * 16)
    comp = compare_suites(base, cand, alpha=0.05)
    assert comp.significant  # the change is real...
    assert comp.verdict == "NO SIGNIFICANT REGRESSION"  # ...but it is an improvement
    assert comp.n_improvements == 8


def test_compare_small_change_not_significant():
    # 1 regression out of 16: chi2 = 0 -> not significant
    base = _suite_result("v1", [1.0] * 16)
    cand = _suite_result("v2", [1.0] * 15 + [0.0])
    comp = compare_suites(base, cand, alpha=0.05)
    assert comp.verdict == "NO SIGNIFICANT REGRESSION"
    assert comp.n_regressions == 1  # counted per-case, just not significant


def test_compare_continuous_uses_ttest():
    base = _suite_result("v1", [0.90, 0.80, 0.85, 0.92, 0.78, 0.86, 0.90, 0.81])
    cand = _suite_result("v2", [0.40, 0.32, 0.35, 0.41, 0.29, 0.36, 0.40, 0.30])
    comp = compare_suites(base, cand, alpha=0.05)
    assert "t-test" in comp.test_name
    assert comp.verdict == "REGRESSION DETECTED"


def test_compare_wilcoxon_method():
    base = _suite_result("v1", [0.9] * 10)
    cand = _suite_result("v2", [0.2] * 10)
    comp = compare_suites(base, cand, method="wilcoxon")
    assert "Wilcoxon" in comp.test_name
    assert comp.verdict == "REGRESSION DETECTED"


def test_compare_rejects_mcnemar_on_continuous():
    base = _suite_result("v1", [0.9, 0.8])
    cand = _suite_result("v2", [0.4, 0.3])
    with pytest.raises(ValueError, match="binary"):
        compare_suites(base, cand, method="mcnemar")


def test_compare_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        compare_suites(_suite_result("v1", [1.0, 1.0]), _suite_result("v2", [1.0, 1.0]),
                       method="bogus")


def test_case_comparison_fields():
    comp = compare_suites(_suite_result("v1", [1.0, 0.0]), _suite_result("v2", [0.0, 1.0]))
    by_id = {c.case_id: c for c in comp.cases}
    assert by_id["c0"].regression and by_id["c0"].delta == pytest.approx(-1.0)
    assert by_id["c1"].improvement and by_id["c1"].delta == pytest.approx(1.0)

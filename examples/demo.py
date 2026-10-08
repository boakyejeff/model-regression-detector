"""Demo: run the same eval suite against a good model and a regressed one.

Fully offline. Prints the verdict and writes report.md / report.html next
to this file.
"""

from pathlib import Path

from regression_detector import (
    EvalCase,
    EvalSuite,
    MockModelClient,
    compare_suites,
    run_suite,
    write_report,
)

HERE = Path(__file__).parent


def build_suite() -> EvalSuite:
    cases = [
        EvalCase("capitals-1", "What is the capital of France?",
                 "exact_match", {"expected": "Paris"}),
        EvalCase("capitals-2", "What is the capital of Japan?",
                 "exact_match", {"expected": "Tokyo"}),
        EvalCase("math-1", "What is 15% of 200? Reply with just the number.",
                 "numeric_tolerance", {"expected": 30, "tolerance": 0.5}),
        EvalCase("math-2", "What is 7 * 8? Reply with just the number.",
                 "numeric_tolerance", {"expected": 56, "tolerance": 0.5}),
        EvalCase("keyword-1", "Name a programming language used for data science.",
                 "contains", {"substring": "python"}),
        EvalCase("keyword-2", "Name a deep learning framework.",
                 "contains", {"substring": "pytorch"}),
        EvalCase("format-1", "Write a date in YYYY-MM-DD format for New Year's Day 2026.",
                 "regex_match", {"pattern": r"2026-01-01"}),
        EvalCase("format-2", "Reply with a valid US phone number format like (555) 123-4567.",
                 "regex_match", {"pattern": r"\(\d{3}\) \d{3}-\d{4}"}),
        EvalCase("capitals-3", "What is the capital of Italy?",
                 "exact_match", {"expected": "Rome"}),
        EvalCase("math-3", "What is the square root of 144? Reply with just the number.",
                 "numeric_tolerance", {"expected": 12, "tolerance": 0.5}),
        EvalCase("keyword-3", "Name a cloud provider.",
                 "contains", {"substring": "aws"}),
        EvalCase("capitals-4", "What is the capital of Canada?",
                 "exact_match", {"expected": "Ottawa"}),
        EvalCase("math-4", "What is 2 to the power of 10? Reply with just the number.",
                 "numeric_tolerance", {"expected": 1024, "tolerance": 1}),
        EvalCase("keyword-4", "Name a version control system.",
                 "contains", {"substring": "git"}),
        EvalCase("capitals-5", "What is the capital of Australia?",
                 "exact_match", {"expected": "Canberra"}),
        EvalCase("math-5", "What is 100 divided by 4? Reply with just the number.",
                 "numeric_tolerance", {"expected": 25, "tolerance": 0.5}),
    ]
    return EvalSuite(name="demo-regression-suite", cases=cases)


GOOD_ANSWERS = {
    "What is the capital of France?": "Paris",
    "What is the capital of Japan?": "Tokyo",
    "What is 15% of 200? Reply with just the number.": "30",
    "What is 7 * 8? Reply with just the number.": "56",
    "Name a programming language used for data science.": "Python",
    "Name a deep learning framework.": "PyTorch",
    "Write a date in YYYY-MM-DD format for New Year's Day 2026.": "2026-01-01",
    "Reply with a valid US phone number format like (555) 123-4567.": "(555) 123-4567",
    "What is the capital of Italy?": "Rome",
    "What is the square root of 144? Reply with just the number.": "12",
    "Name a cloud provider.": "AWS",
    "What is the capital of Canada?": "Ottawa",
    "What is 2 to the power of 10? Reply with just the number.": "1024",
    # v1.0 gets this one wrong on purpose: gives the candidate a genuine fix
    "Name a version control system.": "SVN",
    "What is the capital of Australia?": "Canberra",
    "What is 100 divided by 4? Reply with just the number.": "25",
}

# A "new model version": regressed on 8 cases, fixed 1.
REGRESSED_OVERRIDES = {
    "What is the capital of France?": "London",          # regression
    "What is 15% of 200? Reply with just the number.": "35",  # regression
    "Name a programming language used for data science.": "Java",  # regression
    "What is the capital of Italy?": "Milan",           # regression
    "What is the square root of 144? Reply with just the number.": "14",  # regression
    "Name a cloud provider.": "Azure",                   # regression
    "What is 2 to the power of 10? Reply with just the number.": "1000",  # regression
    "What is 100 divided by 4? Reply with just the number.": "20",  # regression
    "Name a version control system.": "Git",             # genuine improvement
}


def main() -> None:
    suite = build_suite()
    baseline = MockModelClient(GOOD_ANSWERS, name="v1.0-good")
    regressed_answers = dict(GOOD_ANSWERS)
    regressed_answers.update(REGRESSED_OVERRIDES)
    # make one genuine improvement: baseline gets keyword-4 wrong via override below
    candidate = MockModelClient(regressed_answers, name="v1.1-regressed")

    base_res = run_suite(baseline, suite)
    cand_res = run_suite(candidate, suite)

    print(f"Baseline  ({base_res.client_name}): {base_res.pass_rate:.0%} pass "
          f"({len(base_res.results)} cases)")
    print(f"Candidate ({cand_res.client_name}): {cand_res.pass_rate:.0%} pass "
          f"({len(cand_res.results)} cases)")

    comp = compare_suites(base_res, cand_res, alpha=0.05)
    print(f"\nTest: {comp.test_name}")
    print(f"statistic={comp.statistic:.4f}  p-value={comp.p_value:.4f}  alpha={comp.alpha}")
    print(f"Effect: {comp.effect_label}")
    print(f"\nVerdict: {comp.verdict}")

    md_path = write_report(comp, HERE / "report.md", format="markdown")
    html_path = write_report(comp, HERE / "report.html", format="html")
    print(f"\nReports written:\n  {md_path}\n  {html_path}")


if __name__ == "__main__":
    main()

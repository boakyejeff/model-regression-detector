# Model Regression Detector

An offline eval harness that runs the **same eval suite against two model versions** (baseline vs candidate) and flags **statistically significant regressions** before they ship. Think of it as CI for model behavior: green means the new version is safe, red means it got worse somewhere that matters.

## Why this exists

Model updates routinely fix some things and silently break others. Spot-checking a few prompts doesn't catch it; you need the same suite, paired per-case scoring, and a real significance test so you're not chasing noise. This tool gives you that loop in ~50 lines of glue code.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. `numpy`/`scipy` are optional (used for p-values when installed; pure-Python fallbacks are built in).

## Quickstart

```bash
python examples/demo.py
```

This runs a 16-case suite against a good mock model and a deliberately regressed mock, prints the verdict, and writes `examples/report.md` and `examples/report.html`.

Expected output (abridged):

```
Baseline  (v1.0-good): 94% pass (16 cases)
Candidate (v1.1-regressed): 50% pass (16 cases)

Test: McNemar's test (continuity corrected)
statistic=4.0000  p-value=0.0455  alpha=0.05
Effect: discordant odds ratio (regressions/improvements) = 8.000

Verdict: REGRESSION DETECTED
```

## Use it on your own models

1. Write a suite (YAML shown in `examples/suite.yaml`):

```yaml
name: my-suite
cases:
  - id: capitals-1
    prompt: What is the capital of France?
    scorer: exact_match
    scorer_params: { expected: Paris }
```

2. Implement one method:

```python
from regression_detector import EvalSuite, run_suite, compare_suites, write_report

class MyClient:
    def generate(self, prompt: str) -> str:
        return my_llm_call(prompt)  # your code here

suite = EvalSuite.from_yaml("my-suite.yaml")
baseline = run_suite(MyClientV1(), suite)
candidate = run_suite(MyClientV2(), suite)

comp = compare_suites(baseline, candidate, alpha=0.05)
print(comp.verdict)
write_report(comp, "report.md")
```

Built-in scorers: `exact_match`, `contains`, `regex_match`, `numeric_tolerance`.
For custom logic, wrap any function with `CustomScorer`. For model-as-judge
scoring, `LLMJudge` accepts your judge function — the package itself never
makes network calls.

## How the statistics work

- **Binary scores (pass/fail):** McNemar's test on discordant pairs, with continuity correction. Counts how many cases flipped pass→fail vs fail→pass and tests whether the asymmetry is real.
- **Continuous scores:** paired t-test by default (`method="wilcoxon"` for the signed-rank alternative).
- A **regression verdict requires both** statistical significance (p < alpha) **and** the candidate being worse on average. A significant *improvement* is never flagged as a regression.
- Reports include per-case deltas, effect sizes (discordant odds ratio / Cohen's dz), and a worst-regressions table.

## Limitations (honest)

- **Statistical power depends on suite size.** A 10-case suite can only detect large regressions; McNemar needs roughly 6+ discordant pairs to reach p < 0.05. Bigger, more diverse suites detect smaller regressions.
- **This tool measures score differences, not root causes.** A flagged regression tells you *what* got worse, not *why*. Diagnosis is still your job.
- **Scores are only as good as the scorers.** Exact-match and regex scorers are brittle by nature; for open-ended tasks, invest in good rubrics or a calibrated judge.
- **No multiple-testing correction** across cases; the verdict rests on the single suite-level test, which is the right unit of decision.

## Project layout

```
regression_detector/
  evals.py      # EvalCase / EvalSuite, YAML+JSON loading
  scorers.py    # built-in scorers, CustomScorer, LLMJudge stub
  runners.py    # ModelClient protocol, MockModelClient, run_suite
  compare.py    # McNemar / paired t-test / Wilcoxon, verdicts
  report.py     # Markdown + HTML reports
examples/
  demo.py       # offline end-to-end demo (good vs regressed mock)
  suite.yaml    # example suite file
tests/          # pytest suite
```

## License

MIT — see [LICENSE](LICENSE).

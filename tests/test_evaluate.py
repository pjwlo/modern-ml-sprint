"""
tests/test_evaluate.py

This is the automated regression gate: it runs the groundedness evaluator
against a small fixed set of known summary/source pairs and fails the
build if groundedness drops below the required threshold.

Note: this test does NOT call the live Gemini API. It uses fixed
summary/source pairs instead. This is a deliberate design choice:
CI tests should be fast, deterministic, and not dependent on external
API availability, quota, or non-determinism in model output. The live
Gemini call (src/generate.py) is exercised separately -- e.g. in a
scheduled job or manual smoke test -- not on every push/PR.
"""

import pytest
from src.evaluate import evaluate_groundedness

# Fixed regression cases: (source_text, summary, expected_min_score)
# In a real system, this set would grow over time as a "golden dataset"
# of known-good and known-bad examples -- exactly the kind of dataset
# curation this evaluation layer is meant to support.
REGRESSION_CASES = [
    {
        "name": "well_grounded_summary",
        "source": (
            "Bridgewater Associates is an investment management firm founded in 1975 "
            "by Ray Dalio. The firm is known for its systems-driven approach to "
            "understanding markets and economies."
        ),
        "summary": (
            "Bridgewater is an investment firm founded by Ray Dalio in 1975. "
            "It uses a systems-driven approach to markets."
        ),
        "expect_pass": True,
    },
    {
        "name": "ungrounded_summary",
        "source": (
            "Bridgewater Associates is an investment management firm founded in 1975 "
            "by Ray Dalio. The firm is known for its systems-driven approach to "
            "understanding markets and economies."
        ),
        "summary": (
            "Bridgewater is a technology company founded in 2010 by Bill Gates. "
            "It focuses on cloud computing infrastructure."
        ),
        "expect_pass": False,
    },
]


@pytest.mark.parametrize(
    "case", REGRESSION_CASES, ids=[c["name"] for c in REGRESSION_CASES]
)
def test_groundedness_regression(case):
    """
    Regression test: each known case should evaluate to its expected
    pass/fail outcome. If this drifts, either the evaluator logic changed
    unexpectedly, or the thresholds need to be revisited.
    """
    result = evaluate_groundedness(case["summary"], case["source"])
    assert result.passed == case["expect_pass"], (
        f"Case '{case['name']}' expected passed={case['expect_pass']} "
        f"but got passed={result.passed} (score={result.score})"
    )


def test_groundedness_score_is_bounded():
    """Sanity check: groundedness score should always be between 0 and 1."""
    source = "The sky is blue."
    summary = "The sky is blue and green."
    result = evaluate_groundedness(summary, source)
    assert 0.0 <= result.score <= 1.0

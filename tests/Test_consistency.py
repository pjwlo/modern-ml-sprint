"""
tests/test_consistency.py

Regression test for the consistency-sampling evaluator. Like
tests/test_evaluate.py, this uses fixed, pre-generated sample data rather
than calling the live Gemini API -- keeping CI fast, deterministic, and
independent of API availability.
"""

import pytest
from src.consistency import evaluate_consistency

CONSISTENCY_CASES = [
    {
        "name": "consistent_samples",
        "samples": [
            "Bridgewater is an investment firm founded by Ray Dalio in 1975.",
            "Bridgewater was founded in 1975 by Ray Dalio as an investment firm.",
            "Ray Dalio founded Bridgewater, an investment management firm, in 1975.",
        ],
        "expect_pass": True,
    },
    {
        "name": "inconsistent_samples",
        "samples": [
            "Bridgewater is an investment firm founded by Ray Dalio in 1975.",
            "The Eiffel Tower is a famous landmark in Paris, France.",
            "Python is a popular programming language for data science.",
        ],
        "expect_pass": False,
    },
]


@pytest.mark.parametrize(
    "case", CONSISTENCY_CASES, ids=[c["name"] for c in CONSISTENCY_CASES]
)
def test_consistency_regression(case):
    """
    Regression test: each known set of samples should evaluate to its
    expected pass/fail outcome.
    """
    result = evaluate_consistency(case["samples"])
    assert result.passed == case["expect_pass"], (
        f"Case '{case['name']}' expected passed={case['expect_pass']} "
        f"but got passed={result.passed} (score={result.score})"
    )


def test_consistency_requires_at_least_two_samples():
    """A single sample can't be checked for consistency against anything."""
    with pytest.raises(ValueError):
        evaluate_consistency(["only one sample here."])


def test_consistency_score_is_bounded():
    """Sanity check: consistency score should always be between 0 and 1."""
    samples = ["The sky is blue.", "The sky is blue and clear today."]
    result = evaluate_consistency(samples)
    assert 0.0 <= result.score <= 1.0

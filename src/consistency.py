"""
src/consistency.py

Consistency sampling: a second evaluation method, complementary to
groundedness scoring in evaluate.py.

Instead of checking outputs against a source document, this method checks
outputs against *each other*. The same prompt is sent to Gemini multiple
times; if the model gives substantially different answers each time, that's
a signal of unreliability on this particular input -- even without knowing
which (if any) answer is correct. This is especially useful for catching
issues that groundedness scoring alone would miss, e.g. cases where there
is no clear source document to check against.

Design notes:
- Reuses the same lexical-overlap technique from evaluate.py, applied
  pairwise across multiple generations instead of against a source text.
  Keeping the underlying comparison method consistent across both
  evaluators makes the codebase easier to reason about and maintain.
- Sampling multiple times multiplies API cost linearly (N calls instead
  of 1), so this method trades cost for a different kind of signal than
  groundedness scoring -- it's a deliberate tradeoff, not free.
"""

from dataclasses import dataclass
from itertools import combinations

from src.evaluate import _claim_overlap_score


@dataclass
class ConsistencyResult:
    score: float  # 0.0 - 1.0, average pairwise agreement across samples
    samples: list  # the raw generated outputs, for inspection/debugging
    pairwise_scores: list  # each pair's agreement score, for debugging
    passed: bool


def evaluate_consistency(
    samples: list,
    agreement_threshold: float = 0.6,
) -> ConsistencyResult:
    """
    Evaluate consistency across multiple generations of the same prompt.

    Args:
        samples: A list of generated text outputs (from running the same
            prompt multiple times). Must contain at least 2 samples.
        agreement_threshold: Minimum average pairwise overlap score for
            the set of samples to be considered "consistent".

    Returns:
        A ConsistencyResult with the overall agreement score, the raw
        samples, per-pair scores, and pass/fail.
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to evaluate consistency.")

    pairwise_scores = []
    for sample_a, sample_b in combinations(samples, 2):
        # Overlap is checked in both directions and averaged, since
        # _claim_overlap_score(a, b) is not necessarily symmetric --
        # a shorter sample overlapping fully with a longer one shouldn't
        # be penalized just because it's shorter.
        score_ab = _claim_overlap_score(sample_a, sample_b)
        score_ba = _claim_overlap_score(sample_b, sample_a)
        pair_score = (score_ab + score_ba) / 2
        pairwise_scores.append(round(pair_score, 3))

    overall_score = sum(pairwise_scores) / len(pairwise_scores)
    passed = overall_score >= agreement_threshold

    return ConsistencyResult(
        score=round(overall_score, 3),
        samples=samples,
        pairwise_scores=pairwise_scores,
        passed=passed,
    )


def generate_consistency_samples(
    source_text: str,
    client=None,
    model_name: str = "gemini-3.6-flash",
    n_samples: int = 3,
) -> list:
    """
    Live helper: generates n_samples summaries of the same source_text by
    calling Gemini multiple times. This makes real API calls -- intended
    for manual/smoke testing, not for use inside the automated CI test
    (see tests/test_consistency.py, which uses fixed sample data instead).
    """
    from src.generate import generate_summary, get_client

    client = client or get_client()
    return [
        generate_summary(source_text, client=client, model_name=model_name)
        for _ in range(n_samples)
    ]


if __name__ == "__main__":
    sample_text = (
        "Bridgewater Associates is an investment management firm founded in 1975 "
        "by Ray Dalio. The firm is known for its systems-driven approach to "
        "understanding markets and economies."
    )
    samples = generate_consistency_samples(sample_text, n_samples=3)
    for i, s in enumerate(samples, 1):
        print(f"Sample {i}: {s}\n")

    result = evaluate_consistency(samples)
    print(f"Consistency score: {result.score}")
    print(f"Passed: {result.passed}")

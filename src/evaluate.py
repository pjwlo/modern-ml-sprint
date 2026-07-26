"""
src/evaluate.py

Groundedness evaluation: checks whether claims in a generated summary
are actually supported by the source text.

This is a lightweight, mechanical groundedness check (not LLM-as-judge).
It splits the summary into sentence-level claims and checks each one
for lexical overlap with the source text. This is intentionally simple
and fast -- the point is to demonstrate the *pattern*, which can later
be swapped for a more sophisticated retrieval-based or embedding-based
groundedness scorer without changing the pipeline shape.
"""

import re
from dataclasses import dataclass


@dataclass
class GroundednessResult:
    score: float  # 0.0 - 1.0, fraction of claims that are grounded
    claim_scores: list  # per-claim breakdown, for debugging/reporting
    passed: bool


def _split_into_claims(summary: str) -> list:
    """Split a summary into individual sentence-level claims."""
    # Simple sentence split; good enough for a first-pass evaluator.
    claims = re.split(r"(?<=[.!?])\s+", summary.strip())
    return [c.strip() for c in claims if c.strip()]


def _claim_overlap_score(claim: str, source_text: str) -> float:
    """
    Compute a simple lexical overlap score between a claim and the source.
    Returns the fraction of the claim's significant words that appear
    in the source text.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "of",
        "in", "on", "at", "to", "for", "with", "by", "as", "that", "this",
        "it", "its", "be", "been", "has", "have", "had",
    }

    def significant_words(text: str) -> set:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    claim_words = significant_words(claim)
    source_words = significant_words(source_text)

    if not claim_words:
        return 1.0  # nothing substantive to check

    overlap = claim_words & source_words
    return len(overlap) / len(claim_words)


def evaluate_groundedness(
    summary: str,
    source_text: str,
    claim_threshold: float = 0.6,
    overall_threshold: float = 0.8,
) -> GroundednessResult:
    """
    Evaluate how well a summary is grounded in its source text.

    Args:
        summary: The generated summary to check.
        source_text: The original source text.
        claim_threshold: Minimum overlap score for a single claim to count as "grounded".
        overall_threshold: Minimum fraction of grounded claims for the summary to pass overall.

    Returns:
        A GroundednessResult with the overall score, per-claim breakdown, and pass/fail.
    """
    claims = _split_into_claims(summary)

    claim_scores = []
    grounded_count = 0

    for claim in claims:
        score = _claim_overlap_score(claim, source_text)
        is_grounded = score >= claim_threshold
        if is_grounded:
            grounded_count += 1
        claim_scores.append({
            "claim": claim,
            "overlap_score": round(score, 3),
            "grounded": is_grounded,
        })

    overall_score = grounded_count / len(claims) if claims else 1.0
    passed = overall_score >= overall_threshold

    return GroundednessResult(
        score=round(overall_score, 3),
        claim_scores=claim_scores,
        passed=passed,
    )


if __name__ == "__main__":
    source = (
        "Bridgewater Associates is an investment management firm founded in 1975 "
        "by Ray Dalio. The firm is known for its systems-driven approach to "
        "understanding markets and economies."
    )
    summary = (
        "Bridgewater is an investment firm founded by Ray Dalio in 1975. "
        "It uses a systems-driven approach to markets."
    )
    result = evaluate_groundedness(summary, source)
    print(f"Groundedness score: {result.score}")
    print(f"Passed: {result.passed}")
    for c in result.claim_scores:
        print(f"  - [{c['overlap_score']}] {c['claim']}")

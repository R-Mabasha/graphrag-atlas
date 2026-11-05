from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Computes Reciprocal Rank Fusion (RRF) across multiple ranked candidate lists.
    Formula: score(d) = sum(1.0 / (k + rank)) for each list where document appears.

    Returns:
        List of (doc_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = defaultdict(float)

    for r_list in ranked_lists:
        for rank, doc_id in enumerate(r_list):
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_results

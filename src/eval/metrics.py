"""Retrieval and end-to-end metrics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float


def evaluate_retrieval(
    retrieved_case_ids: list[str], relevant_case_ids: set[str], k: int
) -> RetrievalMetrics:
    """Standard set-style metrics over case_ids (deduplicated, ordered)."""
    seen: list[str] = []
    for cid in retrieved_case_ids:
        if cid not in seen:
            seen.append(cid)
        if len(seen) >= k:
            break
    if not relevant_case_ids:
        return RetrievalMetrics(0.0, 0.0, 0.0)
    hits = [cid for cid in seen if cid in relevant_case_ids]
    recall = len(hits) / len(relevant_case_ids)
    precision = len(hits) / max(1, len(seen))
    mrr = 0.0
    for i, cid in enumerate(seen, start=1):
        if cid in relevant_case_ids:
            mrr = 1.0 / i
            break
    return RetrievalMetrics(recall, precision, mrr)


def aggregate(rows: list[RetrievalMetrics]) -> RetrievalMetrics:
    if not rows:
        return RetrievalMetrics(0.0, 0.0, 0.0)
    n = len(rows)
    return RetrievalMetrics(
        recall_at_k=sum(r.recall_at_k for r in rows) / n,
        precision_at_k=sum(r.precision_at_k for r in rows) / n,
        mrr=sum(r.mrr for r in rows) / n,
    )

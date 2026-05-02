"""Run the gold set across retrieval modes and report metrics."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..config import SETTINGS
from ..index import load_index
from ..retrieve import Retriever
from .metrics import RetrievalMetrics, aggregate, evaluate_retrieval

GOLD = SETTINGS.gold_dir / "gold_set.jsonl"


def load_gold() -> list[dict]:
    rows = []
    with GOLD.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(k: int = 10) -> dict[str, RetrievalMetrics]:
    gold = load_gold()
    bundle = load_index()
    retriever = Retriever.from_bundle(bundle)
    out: dict[str, RetrievalMetrics] = {}
    for mode in ("semantic", "lexical", "hybrid"):
        per_query = []
        for row in gold:
            hits = retriever.retrieve(row["query"], mode=mode, k_final=k)  # type: ignore[arg-type]
            ids = [h.chunk.case_id for h in hits]
            per_query.append(
                evaluate_retrieval(ids, set(row["relevant_case_ids"]), k=k)
            )
        out[mode] = aggregate(per_query)
    return out


def main() -> None:
    console = Console()
    metrics = evaluate(k=10)
    table = Table(title="Retrieval metrics @ k=10 (gold set)")
    table.add_column("mode")
    table.add_column("recall@10", justify="right")
    table.add_column("precision@10", justify="right")
    table.add_column("MRR", justify="right")
    for mode, m in metrics.items():
        table.add_row(
            mode,
            f"{m.recall_at_k:.3f}",
            f"{m.precision_at_k:.3f}",
            f"{m.mrr:.3f}",
        )
    console.print(table)


if __name__ == "__main__":
    main()

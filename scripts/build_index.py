"""End-to-end indexing: ingest cases -> chunk -> FAISS + BM25."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SETTINGS  # noqa: E402
from src.ingest import build_cases  # noqa: E402
from src.index import build_index  # noqa: E402


def main() -> None:
    cases_jsonl = SETTINGS.cases_dir / "cases.jsonl"
    if not cases_jsonl.exists():
        print(f"Normalizing FAA dataset -> {cases_jsonl}")
        n = build_cases()
        print(f"  wrote {n} cases")
    else:
        print(f"Reusing {cases_jsonl} (delete to rebuild)")
    build_index()


if __name__ == "__main__":
    main()

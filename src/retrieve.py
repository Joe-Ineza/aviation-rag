"""Hybrid retrieval (semantic + lexical) with Reciprocal Rank Fusion."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import SETTINGS
from .embeddings import Embedder, HashingLsaBackend, make_embedder
from .index import IndexBundle, _tokenize
from .schema import RetrievedChunk

Mode = Literal["semantic", "lexical", "hybrid"]


@dataclass
class Retriever:
    bundle: IndexBundle
    embedder: Embedder

    @classmethod
    def from_bundle(cls, bundle: IndexBundle) -> "Retriever":
        embedder = cls._restore_embedder(bundle)
        return cls(bundle=bundle, embedder=embedder)

    @staticmethod
    def _restore_embedder(bundle: IndexBundle) -> Embedder:
        # Try the persisted LSA fallback first.
        embedder_path = SETTINGS.index_dir / "embedder.pkl"
        if embedder_path.exists():
            try:
                with embedder_path.open("rb") as f:
                    return pickle.load(f)
            except Exception:
                pass  # fall through and refit
        # Try the production embedders.
        try:
            return make_embedder(bundle.embed_model_name)
        except Exception:
            pass
        # Last resort: rebuild the LSA fallback from chunks in memory.
        lsa = HashingLsaBackend()
        lsa.fit([c.text for c in bundle.chunks])
        return lsa

    def _semantic(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query]).astype("float32")
        scores, idxs = self.bundle.faiss_index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def _lexical(self, query: str, k: int) -> list[tuple[int, float]]:
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self.bundle.bm25.get_scores(toks)
        if k >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            top = np.argpartition(scores, -k)[-k:]
            order = top[np.argsort(scores[top])[::-1]]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        mode: Mode = "hybrid",
        k_semantic: int | None = None,
        k_lexical: int | None = None,
        k_final: int | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k_semantic = k_semantic or SETTINGS.top_k_semantic
        k_lexical = k_lexical or SETTINGS.top_k_lexical
        k_final = k_final or SETTINGS.top_k_final
        rrf_k = rrf_k or SETTINGS.rrf_k

        if mode == "semantic":
            ranked = self._semantic(query, k_semantic)
            source = "semantic"
        elif mode == "lexical":
            ranked = self._lexical(query, k_lexical)
            source = "lexical"
        else:
            sem = self._semantic(query, k_semantic)
            lex = self._lexical(query, k_lexical)
            ranked = self._rrf(sem, lex, rrf_k)
            source = "hybrid"

        return [
            RetrievedChunk(chunk=self.bundle.chunks[i], score=s, source=source)
            for i, s in ranked[:k_final]
        ]

    @staticmethod
    def _rrf(
        semantic: list[tuple[int, float]],
        lexical: list[tuple[int, float]],
        k: int,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for ranked in (semantic, lexical):
            for rank, (idx, _) in enumerate(ranked, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def format_evidence(retrieved: list[RetrievedChunk]) -> str:
    lines = []
    for r in retrieved:
        c = r.chunk
        lines.append(
            f"[case_id={c.case_id} | field={c.field} | score={r.score:.4f}]\n{c.text}"
        )
    return "\n\n".join(lines)

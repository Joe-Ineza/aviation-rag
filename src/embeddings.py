"""Embedding backend abstraction.

Default (production): fastembed (ONNX runtime, no PyTorch).
Fallback: sentence-transformers if torch is installed.
Last resort (no network): HashingVectorizer + TruncatedSVD via scikit-learn.
"""
from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Iterable[str]) -> np.ndarray: ...


class FastEmbedBackend:
    """fastembed-backed BGE embedder. Returns L2-normalized float32 vectors."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding
        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)
        probe = next(self._model.embed(["probe"]))
        self.dim = int(probe.shape[0])

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = np.vstack(list(self._model.embed(texts))).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype("float32")


class SentenceTransformersBackend:
    """sentence-transformers-backed embedder. Requires torch."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        return self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")


class HashingLsaBackend:
    """Network-free fallback. HashingVectorizer (16k features) + TruncatedSVD
    projection to a dense space. LSA, not real semantics, but it lets the
    pipeline run end-to-end without any model downloads.
    """

    def __init__(self, dim: int = 256, n_features: int = 16384) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        self.name = f"hashing-lsa-{dim}"
        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None
        )
        self._svd = TruncatedSVD(n_components=dim, random_state=0)
        self._normalizer = Normalizer(copy=False)
        self._fitted = False

    def fit(self, texts) -> None:
        X = self._vectorizer.transform(list(texts))
        self._svd.fit(X)
        self._fitted = True

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if not self._fitted:
            self.fit(texts)
        X = self._vectorizer.transform(texts)
        Z = self._svd.transform(X)
        Z = self._normalizer.transform(Z)
        return Z.astype("float32")


def make_embedder(model_name: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    """Pick the best available backend.

    Priority: fastembed -> sentence-transformers -> hashing-lsa fallback.
    The first two need network access on first run; the last runs anywhere.
    """
    for cls in (FastEmbedBackend, SentenceTransformersBackend):
        try:
            return cls(model_name)
        except Exception:
            continue
    return HashingLsaBackend()

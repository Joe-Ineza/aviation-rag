"""Build and load FAISS (semantic) and BM25 (lexical) indices over chunks."""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .config import SETTINGS
from .embeddings import Embedder, HashingLsaBackend, make_embedder
from .ingest import load_cases
from .chunker import chunk_cases
from .schema import Chunk

TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    embeddings: np.ndarray
    faiss_index: faiss.Index
    bm25: BM25Okapi
    embed_model_name: str


def build_index(out_dir: Path | None = None) -> IndexBundle:
    out_dir = out_dir or SETTINGS.index_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    chunks = chunk_cases(cases)
    if not chunks:
        raise RuntimeError("No chunks produced; did ingest run?")

    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print(f"Loading embedding model: {SETTINGS.embed_model}")
    embedder: Embedder = make_embedder(SETTINGS.embed_model)
    print(f"Using embedder backend: {embedder.name}")
    texts = [c.text for c in chunks]

    fit = getattr(embedder, "fit", None)
    if callable(fit):
        fit(texts)

    BATCH = 256
    parts: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), BATCH), desc="embedding chunks"):
        parts.append(embedder.encode(texts[i : i + BATCH]))
    embeddings = np.vstack(parts).astype("float32")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    np.save(out_dir / "embeddings.npy", embeddings)
    faiss.write_index(faiss_index, str(out_dir / "faiss.index"))
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(
            {"bm25": bm25, "model": SETTINGS.embed_model, "backend": embedder.name},
            f,
        )
    if isinstance(embedder, HashingLsaBackend):
        with (out_dir / "embedder.pkl").open("wb") as f:
            pickle.dump(embedder, f)

    print(
        f"Indexed {len(chunks)} chunks across {len(cases)} cases "
        f"(dim={dim}, backend={embedder.name}) at {out_dir}"
    )
    return IndexBundle(
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=faiss_index,
        bm25=bm25,
        embed_model_name=SETTINGS.embed_model,
    )


def load_index(out_dir: Path | None = None) -> IndexBundle:
    out_dir = out_dir or SETTINGS.index_dir
    chunks: list[Chunk] = []
    with (out_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    embeddings = np.load(out_dir / "embeddings.npy")
    faiss_index = faiss.read_index(str(out_dir / "faiss.index"))
    with (out_dir / "bm25.pkl").open("rb") as f:
        payload = pickle.load(f)
    return IndexBundle(
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=faiss_index,
        bm25=payload["bm25"],
        embed_model_name=payload["model"],
    )


if __name__ == "__main__":
    build_index()

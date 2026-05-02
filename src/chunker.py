"""Structure-aware chunker.

Splits per case, then per field (summary / description / comments / metadata).
Falls back to recursive sentence/character splitting only when a field exceeds
the chunk budget. Preserves provenance (case_id, field) on every chunk.
"""
from __future__ import annotations

import re
from typing import Iterable

from .schema import Case, Chunk

CHUNK_TARGET_CHARS = 1200  # soft target; most FAA narratives are far shorter
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _recursive_split(text: str, target: int = CHUNK_TARGET_CHARS) -> list[str]:
    """Split overlong text along sentence boundaries; hard-cut as last resort."""
    text = text.strip()
    if len(text) <= target:
        return [text]
    sentences = SENTENCE_SPLIT.split(text)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= target:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                out.append(buf)
            if len(s) <= target:
                buf = s
            else:
                # hard-cut a single very long sentence
                for i in range(0, len(s), target):
                    out.append(s[i : i + target])
                buf = ""
    if buf:
        out.append(buf)
    return [c for c in out if c]


def _metadata_text(case: Case) -> str:
    md = case.metadata
    parts = [
        f"date: {md.event_date}" if md.event_date else "",
        f"location: {', '.join(p for p in [md.city, md.state, md.region] if p)}" if (md.city or md.state) else "",
        f"aircraft: {' '.join(p for p in [md.aircraft_make, md.aircraft_model] if p)}" if md.aircraft_make else "",
        f"aircraft class: {md.aircraft_class}" if md.aircraft_class else "",
        f"weight class: {md.weight_class}" if md.weight_class else "",
        f"weather: {md.weather}" if md.weather else "",
        f"light: {md.light_condition}" if md.light_condition else "",
        f"flight phase: {md.flight_phase}" if md.flight_phase else "",
        f"flight purpose: {md.flight_purpose}" if md.flight_purpose else "",
    ]
    return " | ".join(p for p in parts if p)


def chunk_case(case: Case) -> list[Chunk]:
    out: list[Chunk] = []

    # summary   always one chunk
    out.append(
        Chunk(
            chunk_id=f"{case.case_id}::summary",
            case_id=case.case_id,
            field="summary",
            text=case.summary,
        )
    )

    # description   split if long
    desc = case.description.strip()
    if desc:
        for i, part in enumerate(_recursive_split(desc)):
            out.append(
                Chunk(
                    chunk_id=f"{case.case_id}::description::{i}",
                    case_id=case.case_id,
                    field="description",
                    text=part,
                )
            )

    # comments   concatenate; one chunk unless the joined text is huge
    if case.comments:
        joined = " ; ".join(case.comments)
        for i, part in enumerate(_recursive_split(joined)):
            out.append(
                Chunk(
                    chunk_id=f"{case.case_id}::comments::{i}",
                    case_id=case.case_id,
                    field="comments",
                    text=part,
                )
            )

    # metadata   single dense line so structured fields are searchable too
    md_text = _metadata_text(case)
    if md_text:
        out.append(
            Chunk(
                chunk_id=f"{case.case_id}::metadata",
                case_id=case.case_id,
                field="metadata",
                text=md_text,
            )
        )

    return out


def chunk_cases(cases: Iterable[Case]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for case in cases:
        chunks.extend(chunk_case(case))
    return chunks

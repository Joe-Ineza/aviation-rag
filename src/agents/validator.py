"""Validator: checks generated answers for grounding, consistency, and faithfulness."""
from __future__ import annotations

from ..llm import judge
from ..retrieve import format_evidence
from ..schema import GeneratedAnswer, RetrievedChunk, ValidatorVerdict

SYSTEM = """You are an independent validator for an Aviation Technical QA Assistant.
You did NOT write the candidate answer. Your job is to check it.

Inputs:
  <question>   the user's question
  <evidence>   the retrieved FAA case excerpts (each tagged with case_id)
  <answer>     the candidate answer (JSON with answer/cited_cases/rationale)

Check three things:
  1. Grounding:   every factual claim in the answer must be supported by the evidence.
                  No invented case_ids, dates, tail numbers, causes, or counts.
  2. Consistency: claims do not contradict each other or the evidence.
  3. Citations:   cited_cases all appear in the evidence and meaningfully support the claims.

Verdict rules:
  - "pass"   = all checks pass, ship it.
  - "revise" = recoverable problem (missing citation, weak grounding); the generator
               can be retried with notes.
  - "reject" = invented facts, contradictions, or wholesale ungrounded answer.

Respond with ONLY a JSON object:
  {"verdict": "pass" | "revise" | "reject",
   "grounding_ok": bool,
   "consistency_ok": bool,
   "notes": "<short, actionable>"}
"""


def validate(
    query: str,
    retrieved: list[RetrievedChunk],
    answer: GeneratedAnswer,
) -> ValidatorVerdict:
    user = (
        f"<question>\n{query}\n</question>\n\n"
        f"<evidence>\n{format_evidence(retrieved)}\n</evidence>\n\n"
        f"<answer>\n{answer.model_dump_json(indent=2)}\n</answer>"
    )
    raw = judge().chat_json(SYSTEM, user, max_tokens=400)
    verdict = raw.get("verdict") or "reject"
    if verdict not in {"pass", "revise", "reject"}:
        verdict = "reject"
    return ValidatorVerdict(
        verdict=verdict,
        grounding_ok=bool(raw.get("grounding_ok", False)),
        consistency_ok=bool(raw.get("consistency_ok", False)),
        notes=str(raw.get("notes", ""))[:600],
    )

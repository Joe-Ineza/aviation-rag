"""ReAct-style planner.

Owns the loop: scope check -> retrieve -> (refine if weak) -> generate ->
validate -> (one revise retry on validator notes) -> return trace.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import SETTINGS
from ..llm import judge
from ..retrieve import Retriever
from ..schema import (
    AgentRunTrace,
    GeneratedAnswer,
    RetrievedChunk,
    UserQuery,
    ValidatorVerdict,
)
from . import generator as gen_agent
from . import scope_router
from . import validator as val_agent

log = logging.getLogger(__name__)

# Heuristic: rerun retrieval if all hybrid scores are too low to trust.
WEAK_EVIDENCE_FLOOR = 0.005

REFINE_SYSTEM = """\
You refine search queries for an aviation incident knowledge base.
The original query produced weak evidence. Propose ONE alternate query that
emphasises concrete aviation terms (aircraft make/model, system names, phase
of flight, weather, location).

OUTPUT FORMAT: raw JSON only — no markdown, no explanation, no preamble.
{"query": "<improved query string>"}
"""


@dataclass
class RunResult:
    trace: AgentRunTrace
    out_of_scope_message: str | None = None


def _refine_query(original: str) -> str:
    try:
        raw = judge().chat_json(
            REFINE_SYSTEM,
            f"Original query:\n{original}",
            max_tokens=SETTINGS.max_tokens_judge,
        )
        refined = str(raw.get("query") or original).strip()
        log.info("Query refined: %r → %r", original, refined)
        return refined
    except Exception as exc:
        log.warning("Query refinement failed (%s); keeping original", exc)
        return original


def _evidence_is_weak(retrieved: list[RetrievedChunk]) -> bool:
    if not retrieved:
        return True
    return all(r.score < WEAK_EVIDENCE_FLOOR for r in retrieved)


def run(
    raw_query: str,
    retriever: Retriever,
    *,
    prompt_style: str = "cot",
    retrieval_mode: str = "hybrid",
) -> RunResult:
    # 1. Sanitize input (defensive schema validation).
    query = UserQuery(text=raw_query).text

    trace = AgentRunTrace(query=query, scope=None)  # type: ignore[arg-type]

    # 2. Scope routing.
    scope = scope_router.route(query)
    trace.scope = scope
    if not scope.in_scope:
        return RunResult(trace=trace, out_of_scope_message=scope_router.OUT_OF_SCOPE_MESSAGE)

    # 3. ReAct retrieval loop (bounded).
    current_query = query
    retrieved: list[RetrievedChunk] = []
    for step in range(SETTINGS.max_react_steps):
        retrieved = retriever.retrieve(current_query, mode=retrieval_mode)  # type: ignore[arg-type]
        trace.react_steps.append(
            {
                "step": step,
                "query": current_query,
                "n_retrieved": len(retrieved),
                "top_score": retrieved[0].score if retrieved else 0.0,
            }
        )
        if not _evidence_is_weak(retrieved):
            break
        if step + 1 >= SETTINGS.max_react_steps:
            break
        current_query = _refine_query(current_query)

    trace.retrieved = retrieved

    if not retrieved:
        trace.answer = GeneratedAnswer(
            answer="I couldn't find evidence in the case knowledge base to answer that.",
            cited_cases=[],
            rationale="No retrieval hits above threshold.",
            confidence="low",
            caveats="Insufficient evidence.",
        )
        trace.verdict = ValidatorVerdict(
            verdict="pass",
            grounding_ok=True,
            consistency_ok=True,
            notes="Refusal due to no retrieval hits.",
        )
        return RunResult(trace=trace)

    # 4. Generate.
    answer = gen_agent.generate(query, retrieved, style=prompt_style)  # type: ignore[arg-type]
    trace.answer = answer

    # 5. Validate, with one revise-retry on revisable notes.
    verdict = val_agent.validate(query, retrieved, answer)
    if verdict.verdict == "revise":
        retry_query = (
            f"{query}\n\n[validator notes: {verdict.notes}] "
            f"Address the notes and re-answer with proper citations."
        )
        answer = gen_agent.generate(retry_query, retrieved, style=prompt_style)  # type: ignore[arg-type]
        verdict = val_agent.validate(query, retrieved, answer)
        trace.answer = answer

    trace.verdict = verdict
    return RunResult(trace=trace)

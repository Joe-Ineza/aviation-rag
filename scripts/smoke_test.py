"""Architecture smoke test with a mocked LLM.

Verifies the planner -> scope router -> retriever -> generator -> validator
flow wires up correctly WITHOUT calling the gateway. Useful for CI and for
proving the pipeline is sound before plugging in live keys.

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.index import load_index  # noqa: E402
from src.retrieve import Retriever  # noqa: E402
from src.agents import scope_router, generator as gen_agent, validator as val_agent  # noqa: E402
from src.agents import planner  # noqa: E402
from src.schema import GeneratedAnswer, ScopeDecision, ValidatorVerdict  # noqa: E402


def _fake_scope_route(query: str) -> ScopeDecision:
    aviation_terms = ("fuel", "engine", "aircraft", "tailwheel", "cessna", "piper",
                      "takeoff", "landing", "maintenance", "faa", "incident",
                      "accident", "flight", "pilot")
    in_scope = any(t in query.lower() for t in aviation_terms)
    return ScopeDecision(in_scope=in_scope, reason="keyword heuristic (mock)")


def _fake_generate(query, retrieved, style="cot"):
    cited = list({r.chunk.case_id for r in retrieved[:3]})
    return GeneratedAnswer(
        answer=f"[mock] Based on {len(retrieved)} retrieved chunks, "
               f"the most relevant cases are {', '.join(cited)}.",
        cited_cases=cited,
        rationale="Mock generator: cited the top-3 distinct case_ids.",
        confidence="medium",
        caveats="This is a mocked response.",
    )


def _fake_validate(query, retrieved, answer):
    cited_set = set(answer.cited_cases)
    evidence_ids = {r.chunk.case_id for r in retrieved}
    grounding = bool(cited_set) and cited_set.issubset(evidence_ids)
    return ValidatorVerdict(
        verdict="pass" if grounding else "reject",
        grounding_ok=grounding,
        consistency_ok=True,
        notes="mock validator",
    )


def main() -> None:
    # Patch the agent functions used by planner.run.
    scope_router.route = _fake_scope_route  # type: ignore[assignment]
    gen_agent.generate = _fake_generate  # type: ignore[assignment]
    val_agent.validate = _fake_validate  # type: ignore[assignment]

    bundle = load_index()
    retriever = Retriever.from_bundle(bundle)

    print("=== In-scope query ===")
    res = planner.run(
        "fuel exhaustion in single-engine Cessna",
        retriever,
        prompt_style="cot",
        retrieval_mode="hybrid",
    )
    print(json.dumps(res.trace.model_dump(), indent=2, default=str)[:1200])
    print()
    print("=== Out-of-scope query ===")
    res = planner.run("what's the weather in Pittsburgh today?", retriever)
    print("OOS message:", res.out_of_scope_message)


if __name__ == "__main__":
    main()

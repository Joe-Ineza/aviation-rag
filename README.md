# Aviation Operations & Safety Assistant

Agentic RAG over the FAA Accident/Incident (OMIn) maintenance dataset.

## What's in the box

- **2,748 normalized cases** from the OMIn FAA dataset, mapped into a
  Jira-style schema (case_id, summary, description, comments, severity, metadata).
- **Structure-aware chunker** (per case → per field, with sentence-level fallback).
- **Hybrid retrieval**: FAISS semantic + BM25 lexical, fused with Reciprocal Rank Fusion.
  Semantic backend swaps automatically: `fastembed` → `sentence-transformers` → an
  in-process LSA fallback that runs without network access.
- **Agent layer** (CMU AI Gateway, OpenAI-compatible):
  - `scope_router` (Haiku)   gates out-of-scope queries with a fixed refusal.
  - `planner`   bounded ReAct loop with one query refinement on weak evidence.
  - `generator` (Sonnet)   zero-shot or chain-of-thought prompts; returns
    `answer / cited_cases / rationale / confidence / caveats` as structured JSON.
  - `validator` (Haiku)   independent grounding, consistency, and citation check
    with one revise-retry on revisable verdicts.
- **Input-validation schema** (Pydantic) that strips control characters and caps
  query length to blunt trivial prompt injection.
- **Evaluation harness** with recall@k / precision@k / MRR over a small gold set,
  comparing semantic vs. lexical vs. hybrid retrieval.

## Quickstart (your machine   gateway is reachable here)

```bash
python -m venv .venv && .venv\Scripts\activate         # PowerShell
# or:  python -m venv .venv && source .venv/bin/activate   # bash

pip install -r requirements.txt

cp .env.example .env
# edit .env: paste your CMU_AI_GATEWAY_API_KEY (the rotated one).
# Base URL is already set to https://ai-gateway.andrew.cmu.edu/v1

python scripts/build_index.py                          # ~30s on the LSA fallback
python scripts/ask.py "Which fuel-system maintenance findings appear in single-engine piston accidents?"
python scripts/run_eval.py                             # retrieval metrics
python scripts/smoke_test.py                           # mocked agents (no API needed)
```

## Architecture (one-line per layer)

```
UserQuery (Pydantic, sanitized)
  -> scope_router (Haiku, JSON verdict)         # OOS -> canonical refusal
  -> planner ReAct loop (<= 4 steps)
       -> Retriever.retrieve(mode=hybrid)        # FAISS + BM25 + RRF
       -> if weak: judge refines query, retry
  -> generator (Sonnet, zero_shot | cot)         # JSON answer + cited_cases
  -> validator (Haiku)                           # grounding/consistency/citations
       -> verdict in {pass, revise, reject}      # revise -> one retry
  -> AgentRunTrace (full step-by-step record)
```

## Layout

```
src/
  config.py         central env-driven settings
  schema.py         Pydantic models (Case, Chunk, agent I/O)
  ingest.py         FAA c-codes -> normalized Case JSONL
  chunker.py        structure-aware splitter
  embeddings.py     fastembed -> sentence-transformers -> LSA fallback
  index.py          FAISS + BM25 builders
  retrieve.py       hybrid retrieval + RRF
  llm.py            CMU AI Gateway client (OpenAI SDK + custom base_url)
  agents/
    scope_router.py
    planner.py      ReAct loop with refinement + revise-retry
    generator.py    zero-shot + CoT prompts
    validator.py
  eval/
    metrics.py      recall@k, precision@k, MRR
    harness.py      retrieval A/B across modes
scripts/
  build_index.py
  ask.py            CLI entry point
  run_eval.py       retrieval evaluation
  smoke_test.py     mocked end-to-end (no gateway required)
gold_standard/
  gold_set.jsonl    seed labels for evaluation
```

## Notes on the dataset and design

The OMIn dataset is ~2.7k FAA records pre-filtered to maintenance-related
accident-type codes. Each row carries a short narrative (`c119`) plus dense
structured columns. We derive `summary`, `comments`, `severity`, and `metadata`
from those columns so the assistant can reason over both the prose and the
structured signal. Chunks are emitted per field (summary / description /
comments / metadata) so retrieval respects case structure, not arbitrary
character windows.

In production the embedder is BGE-small (384d). The sandbox used to build this
project couldn't reach the HuggingFace CDN, so an LSA fallback is wired in to
keep the pipeline runnable offline; it is not a real semantic model and should
be swapped to BGE wherever the network allows. Drop the LSA `embedder.pkl`
and `embeddings.npy`, set up real BGE access, and rerun `build_index.py`.

See `aviation_rag_implementation_brief.md` for the full design rationale.

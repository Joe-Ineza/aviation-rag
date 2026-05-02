# Aviation Operations & Safety Assistant   Agentic RAG Implementation Brief

## 1. Mission and Role Definition

Build a maintainable, evaluation-driven agentic RAG system that answers aviation operations and safety questions grounded in a corpus of aviation accident and incident reports. The assistant operates strictly as an **Aviation Technical QA Assistant**: it retrieves, reasons over, and synthesizes evidence from the case knowledge base, declines out-of-scope requests, and never speculates beyond retrieved records.

The system must minimize hallucinations, surface its evidence, and behave like a production software component   observable, testable, and recoverable.

---

## 2. Dataset and Knowledge Base

### 2.1 Source

The knowledge base is built from a public aviation accident and incident dataset (Zenodo: https://zenodo.org/records/17096333). Each record contains both structured fields and unstructured narrative content, including:

- Event details (date, location, flight phase, weather)
- Aircraft information (make, model, registration, operator)
- Injury and damage statistics
- Probable cause and contributing findings
- Report status and full extracted report text

This mixed structured/unstructured shape is an ideal fit for RAG: structured metadata supports filtering and exact-match retrieval, while narrative fields require semantic search and reasoning.

### 2.2 Logical Schema (Case Representation)

Each record is normalized into a Jira-style "case" object so the assistant can reason about incidents as discrete tickets:

| Field | Source | Use |
|---|---|---|
| `case_id` | Report ID | Primary key, exact-match retrieval |
| `summary` | Event title / synopsis | Short context, embedding |
| `description` | Full report narrative | Primary semantic retrieval target |
| `status` | Report status (e.g., final, preliminary) | Filtering, freshness |
| `comments` | Findings, probable cause, recommendations | Secondary semantic retrieval |
| `severity` | Derived from injury/damage attributes | Prioritization, filtering |
| `metadata` | Date, location, aircraft type, operator | Structured filters |

### 2.3 Chunking Strategy

Use **structure-aware recursive splitting**, not fixed-length windows:

1. Split first by `case_id` so retrieval boundaries align with incident boundaries.
2. Within a case, split by field section (`summary`, `description`, `findings`, `comments`).
3. If a section exceeds the chunk budget, fall back to recursive character splitting that respects sentence boundaries.

This preserves aviation context (e.g., a "probable cause" passage stays attached to its incident) and prevents cross-incident bleed during retrieval.

---

## 3. Core Agentic Architecture

The assistant follows a **ReAct loop** with explicit reasoning before action. Retrieval and synthesis are tool calls, not implicit steps.

```
Perceive → Reason → Act → Observe → (refine | answer)
```

- **Perceive**: Receive and validate the user query against an input schema.
- **Reason**: Plan a retrieval strategy   decide which fields to filter on, which retriever to invoke, and what evidence is needed.
- **Act**: Call the retriever tool (semantic, lexical, or hybrid).
- **Observe**: Inspect retrieved results. If evidence is sparse, ambiguous, or contradictory, refine the query and loop. Cap iterations to prevent runaway agents.
- **Synthesize**: Once evidence is sufficient, hand off to the generator.

A coordinating planner agent owns the loop. Specialist agents (retriever, generator, validator) are stateless tools it calls.

---

## 4. Retrieval Design

### 4.1 Indexing

- **Vector index**: FAISS (approximate nearest neighbor) over chunk embeddings. Default to a strong general-purpose embedding model; leave the choice swappable behind a config flag.
- **Lexical index**: BM25 over the same chunks for exact-match recall on report IDs, tail numbers (e.g., `N123AB`), ICAO codes, airport identifiers, aircraft model designators, and aviation jargon that embeddings often blur.

### 4.2 Hybrid Retrieval

Run semantic and BM25 retrieval in parallel and fuse with **Reciprocal Rank Fusion (RRF)**:

```
score(d) = Σ over retrievers r of  1 / (k + rank_r(d))
```

RRF requires no score calibration between retrievers and is robust to outliers. Default `k = 60`.

### 4.3 Retrieval Variations to Compare

The system must support A/B comparison between:

1. Pure semantic (FAISS only).
2. Pure lexical (BM25 only).
3. Hybrid with RRF.

Comparisons are logged per-query to an evaluation table for offline analysis.

---

## 5. Prompting Strategy

Two prompting modes, switchable for experimentation:

- **Zero-shot**: Direct instruction with retrieved context. Baseline.
- **Chain-of-thought (CoT)**: Structured prompt that walks the model through *understand → analyze → reason → synthesize*, with explicit instructions to cite the originating `case_id` for each claim.

The generator prompt always:

- Pins the assistant role and scope.
- Forbids answering from parametric knowledge when the question is aviation-specific.
- Requires inline citations of `case_id`s.
- Returns a structured response (`answer`, `cited_cases`, `confidence`, `caveats`).

---

## 6. Evaluation Strategy

The system embeds a **review-and-critique pattern** so generation is paired with verification.

### 6.1 Retrieval Metrics

- **Recall@k** and **Precision@k** against a curated gold set of (query, relevant `case_id`s) pairs.
- **MRR** for the top relevant case.
- Logged per retrieval variation to support direct comparison.

### 6.2 Validator Agent

A separate validator agent runs after the generator and checks:

- **Grounding**: Each claim in the answer maps to a retrieved chunk.
- **Consistency**: Claims do not contradict each other or the source.
- **Faithfulness**: No invented `case_id`s, dates, tail numbers, or causes.

The validator returns a verdict (`pass`, `revise`, `reject`) and structured findings. On `revise`, the generator gets one bounded retry with the validator notes appended.

### 6.3 Self-Explanation

The generator must produce a brief rationale for *why these specific cases were selected* for the final summary (e.g., "Selected NTSB-2019-0042 and -0177 because both involve fuel exhaustion in single-engine piston aircraft during night VFR   matching the user's scenario."). This rationale is logged and surfaced to the user on request.

---

## 7. Safety, Guardrails, and Out-of-Scope Handling

### 7.1 Persona Lock

System prompt establishes the role   *Aviation Technical QA Assistant grounded in the case knowledge base*   and explicitly forbids role overrides from user input.

### 7.2 Scope Decision Node

Before retrieval, a lightweight classifier (LLM-as-judge with a tight prompt, or a small fine-tuned classifier) routes the query:

- **In scope**: Aviation operations, safety, incident analysis, regulatory references found in the corpus → proceed to retrieval.
- **Out of scope**: Unrelated topics (general chit-chat, non-aviation domains, personal advice) → return a predefined refusal:

> *"I'm an aviation technical QA assistant scoped to this incident knowledge base. I can't help with that request, but I'd be glad to answer questions about aviation incidents, safety findings, or operational context drawn from the case records."*

### 7.3 Input Validation

All user input passes through a Pydantic (or equivalent) schema that:

- Enforces length limits.
- Strips or escapes control sequences and known prompt-injection patterns ("ignore previous instructions", role-reset attempts, embedded system tags).
- Separates user content from system instructions in the prompt template   user input is always wrapped in delimited tags the model is instructed never to treat as instructions.

### 7.4 Refusal Behavior

When evidence is insufficient, the assistant says so explicitly rather than inventing. The validator enforces this; ungrounded answers are rejected before reaching the user.

---

## 8. Multi-Agent Topology

| Agent | Responsibility |
|---|---|
| **Planner** | Owns the ReAct loop, decides retrieval strategy, calls tools, terminates when evidence is sufficient. |
| **Retriever** | Stateless tool. Runs hybrid retrieval, returns ranked chunks with metadata. |
| **Generator** | Produces the candidate answer with citations and self-explanation. |
| **Validator** | Independently checks grounding, consistency, and faithfulness. |
| **Scope Router** | Pre-retrieval gate; decides in-scope vs. out-of-scope. |

Separating generator and validator enables parallel error checking and prevents the generator from grading its own work.

---

## 9. State, Persistence, and Recovery

- **Checkpointed state**: The planner persists loop state (query, plan, retrieved evidence, validator verdicts) after each step so a crashed or timed-out run can resume rather than restart.
- **Trace logging**: Every retrieval, generation, and validation is logged with inputs, outputs, latency, and token usage for offline evaluation and debugging.
- **Determinism aids**: Seed where possible; record model and embedding versions per run so evaluation results stay reproducible.

---

## 10. Implementation Phases

1. **Phase 1   Data and indexing**: Normalize records into the case schema; build FAISS and BM25 indices; verify chunking preserves case structure.
2. **Phase 2   Baseline RAG**: Single-shot retrieval + zero-shot generation, no agent loop. Establish a working baseline and a gold evaluation set.
3. **Phase 3   Agentic loop**: Add the planner, ReAct loop, and validator. Wire scope routing and input validation.
4. **Phase 4   Hybrid retrieval and prompting variants**: Add BM25, RRF fusion, CoT prompting. Run head-to-head evaluations.
5. **Phase 5   Hardening**: Persistence, checkpointing, observability, prompt-injection regression suite.

---

## 11. Limitations and Forward Work

- **Scale**: At millions of records, flat FAISS retrieval and single-pass RAG degrade. Plan for hierarchical retrieval (case-level summary index → chunk-level detail index), metadata pre-filtering, and possibly a planner that decomposes queries before retrieval.
- **Embedding drift**: Aviation terminology evolves; embeddings should be re-evaluated periodically against the gold set, and re-indexing should be a routine, not a one-off.
- **Validator coverage**: The validator catches grounding failures but not factual errors *within* the source documents. Source quality is assumed; a separate ingestion-time data-quality check is out of scope for this phase.
- **Latency**: The agent loop and validator add round-trips. Cache scope decisions and retrieval results per query, and cap loop iterations.

---

## 12. Acceptance Criteria

The first production-ready milestone is met when:

- Retrieval recall@10 on the gold set exceeds an agreed threshold (set during Phase 2).
- Validator rejection rate on a held-out set is below an agreed threshold, and rejected answers correlate with reviewer-flagged hallucinations.
- Out-of-scope queries are refused with the standard message in 100% of a curated adversarial set.
- A prompt-injection regression suite (role override, instruction smuggling, citation forgery) passes.
- Every answer surfaced to the user includes at least one valid `case_id` citation or an explicit "insufficient evidence" refusal.

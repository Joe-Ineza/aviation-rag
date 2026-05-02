"""
Aviation Operations & Safety Assistant   Streamlit Presentation App
====================================================================
Run:
    pip install streamlit plotly
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap: make src/ importable regardless of working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page configuration  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aviation Safety Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Minimal, non-interfering CSS    does NOT touch tab bar or containers
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="metric-container"] {
        background: #f7f8fc;
        border: 1px solid #dde2f0;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
    }
    .arch-box {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        font-family: monospace;
        font-size: 0.82rem;
        line-height: 1.65;
        white-space: pre;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dependency checks   catch missing packages early and surface them clearly
# ---------------------------------------------------------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cases() -> list[dict]:
    path = ROOT / "artifacts" / "cases" / "cases.jsonl"
    if not path.exists():
        return []
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


@st.cache_resource(show_spinner=False)
def load_retriever():
    """Returns (retriever, error_message_or_None)."""
    try:
        from src.index import load_index
        from src.retrieve import Retriever
        bundle = load_index()
        return Retriever.from_bundle(bundle), None
    except Exception:
        return None, traceback.format_exc()


def gateway_configured() -> bool:
    try:
        from src.config import SETTINGS
        return bool(SETTINGS.api_key and SETTINGS.base_url)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "The Problem",
    "Data Analysis",
    "Architecture",
    "Live Demo",
    "What's Next",
])


# ===========================================================================
# TAB 1   THE PROBLEM
# ===========================================================================
with tab1:
    st.title("Aviation Safety Intelligence")
    st.markdown("#### The Problem")

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown(
            """
The FAA maintains tens of thousands of aviation accident and incident reports
spanning several decades. Each report contains structured data   aircraft details,
weather conditions, flight phase, cause codes   alongside a free-text narrative
written by investigators.

**The problem is access.** A safety analyst who wants to answer a question like:

> *"What maintenance findings appear most often before fuel-system failures in
> single-engine piston aircraft?"*

...must manually search a dense CSV, cross-reference numeric codes against lookup
tables, and read through hundreds of rows. There is no way to ask the data a
plain-English question and receive a grounded, cited answer.

This project builds exactly that: an **Agentic RAG system** that treats the FAA
OMIn dataset as a structured knowledge base and answers natural-language queries
with evidence, case citations, and a confidence grade.
            """
        )

        st.markdown("---")
        st.markdown("#### Technical Challenges")
        st.markdown(
            """
**Mixed signal types.** Each case has a short narrative, approximately 15
structured cause and condition columns, and rich metadata. A retrieval
system must search across all of them simultaneously without losing the
signal that lives in each separate field.

**Domain vocabulary.** FAA reports use abbreviations, coded fields, and jargon
specific to aviation maintenance. A general-purpose search engine misses this
without domain-aware indexing and tokenisation.

**Grounding requirement.** In a safety context, every factual claim in an answer
must trace back to a real case in the database, not model training memory.
This demands an independent validation step after generation.

**Scope control.** The assistant must stay focused. Off-topic questions, prompt
injection attempts, and adversarial inputs need to be blocked before any
retrieval or generation runs.
            """
        )

    with col_r:
        st.markdown("#### Dataset at a Glance")
        c1, c2 = st.columns(2)
        c1.metric("Total Cases", "2,748")
        c2.metric("Date Range", "1975 – 2008")
        c3, c4 = st.columns(2)
        c3.metric("Searchable Chunks", "10,992")
        c4.metric("Source Columns", "181")

        st.markdown("---")
        st.markdown("#### Core Design Principles")

        with st.expander("Grounded answers only"):
            st.markdown(
                "Every factual claim must cite a real `case_id` from the retrieved "
                "evidence. The generator is explicitly instructed not to draw on "
                "model memory or speculate beyond what the evidence supports."
            )
        with st.expander("Hybrid retrieval"):
            st.markdown(
                "Semantic search (FAISS / BGE embeddings) and lexical search (BM25) "
                "are run independently and fused with Reciprocal Rank Fusion. "
                "Neither alone is sufficient for this domain."
            )
        with st.expander("Scope gate before retrieval"):
            st.markdown(
                "A lightweight judge model checks whether the query is in scope "
                "before any retrieval or generation runs. Out-of-scope queries "
                "receive a canonical refusal immediately."
            )
        with st.expander("Self-correcting validator loop"):
            st.markdown(
                "An independent validator checks every answer for grounding, "
                "consistency, and citation quality. On a 'revise' verdict the "
                "generator retries once with the validator's notes."
            )
        with st.expander("Field-aware chunking"):
            st.markdown(
                "Cases are split by semantic field (summary, description, comments, "
                "metadata) rather than arbitrary character windows. This preserves "
                "the structure of each case record across all retrieval paths."
            )

        st.markdown("---")
        st.markdown("#### Data Source")
        st.markdown(
            """
**OMIn Dataset**   Open Maintenance Intelligence for aviation, published by the
University of Notre Dame CRANE lab. File used: `Maintenance_Text_data_nona.csv`
(maintenance-related accident/incident codes; rows with no narrative pre-removed).
            """
        )


# ===========================================================================
# TAB 2   DATA ANALYSIS
# ===========================================================================
with tab2:
    st.title("Data Analysis")
    st.markdown("Exploratory analysis of the 2,748 normalized FAA cases.")

    if not PLOTLY_OK:
        st.error(
            "Plotly is not installed. Run `pip install plotly` then restart the app.",
        )
    elif not PANDAS_OK:
        st.error("pandas is not installed. Run `pip install pandas` then restart.")
    else:
        cases = load_cases()
        if not cases:
            st.warning(
                "No cases found. Run `python scripts/build_index.py` first to "
                "generate `artifacts/cases/cases.jsonl`."
            )
        else:
            df = pd.DataFrame(cases)
            meta = pd.json_normalize([c["metadata"] for c in cases])
            df = pd.concat([df.drop(columns=["metadata"]), meta], axis=1)

            # Headline metrics
            st.markdown("### Overview")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Total Cases", f"{len(df):,}")
            mc2.metric("Accidents", f"{(df['status'] == 'ACCIDENT').sum():,}")
            mc3.metric("Incidents", f"{(df['status'] == 'INCIDENT').sum():,}")
            mc4.metric("Critical Cases", f"{(df['severity'] == 'CRITICAL').sum():,}")
            mc5.metric("States Covered", f"{df['state'].nunique()}")

            st.markdown("---")

            # Row 1: Status + Severity
            ca, cb = st.columns(2)

            with ca:
                status_counts = df["status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig = px.pie(
                    status_counts, names="Status", values="Count",
                    color="Status",
                    color_discrete_map={"ACCIDENT": "#c0392b", "INCIDENT": "#2980b9"},
                    title="Accident vs Incident Split",
                )
                fig.update_traces(textinfo="percent+label", hole=0.4)
                fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with cb:
                sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
                sev_colors = {
                    "CRITICAL": "#c0392b", "HIGH": "#e67e22",
                    "MEDIUM": "#f1c40f", "LOW": "#27ae60",
                }
                sev_counts = (
                    df["severity"].value_counts()
                    .reindex(sev_order)
                    .reset_index()
                )
                sev_counts.columns = ["Severity", "Count"]
                fig = px.bar(
                    sev_counts, x="Severity", y="Count",
                    color="Severity", color_discrete_map=sev_colors,
                    title="Severity Distribution", text="Count",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, margin=dict(t=40, b=10), yaxis_title="Cases"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Row 2: Timeline
            st.markdown("---")
            st.markdown("### Events Over Time")
            year_df = (
                df[df["event_year"].notna()]
                .assign(event_year=lambda x: x["event_year"].astype(int))
                .groupby(["event_year", "status"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.area(
                year_df, x="event_year", y="Count", color="status",
                color_discrete_map={"ACCIDENT": "#c0392b", "INCIDENT": "#2980b9"},
                title="Cases by Year",
                labels={"event_year": "Year", "status": "Type"},
            )
            fig.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Row 3: Aircraft + States
            st.markdown("---")
            cc, cd = st.columns(2)

            with cc:
                makes = (
                    df["aircraft_make"].dropna().value_counts()
                    .head(10).reset_index()
                )
                makes.columns = ["Make", "Count"]
                fig = px.bar(
                    makes, x="Count", y="Make", orientation="h",
                    color="Count", color_continuous_scale="Blues",
                    title="Top 10 Aircraft Manufacturers", text="Count",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(t=40, b=10), yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with cd:
                states = (
                    df["state"].dropna().value_counts()
                    .head(15).reset_index()
                )
                states.columns = ["State", "Count"]
                fig = px.bar(
                    states, x="State", y="Count",
                    color="Count", color_continuous_scale="Oranges",
                    title="Top 15 States by Case Count", text="Count",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Row 4: Flight phase + Weather
            st.markdown("---")
            ce, cf = st.columns(2)

            with ce:
                phases = (
                    df["flight_phase"].dropna().value_counts()
                    .head(10).reset_index()
                )
                phases.columns = ["Phase", "Count"]
                phases["Phase"] = phases["Phase"].str[:28]
                fig = px.bar(
                    phases, x="Count", y="Phase", orientation="h",
                    color="Count", color_continuous_scale="Purples",
                    title="Top 10 Flight Phases at Time of Event", text="Count",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(t=40, b=10), yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with cf:
                weather_counts = (
                    df["weather"].dropna().value_counts().reset_index()
                )
                weather_counts.columns = ["Condition", "Count"]
                fig = px.pie(
                    weather_counts, names="Condition", values="Count",
                    color="Condition",
                    color_discrete_map={
                        "VFR": "#27ae60", "IFR": "#e67e22", "Unknown": "#95a5a6"
                    },
                    title="Weather Conditions",
                )
                fig.update_traces(textinfo="percent+label", hole=0.4)
                fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # Row 5: Flight purpose + Light
            st.markdown("---")
            cg, ch = st.columns(2)

            with cg:
                purpose = (
                    df["flight_purpose"].dropna().str[:28]
                    .value_counts().head(8).reset_index()
                )
                purpose.columns = ["Purpose", "Count"]
                fig = px.bar(
                    purpose, x="Count", y="Purpose", orientation="h",
                    color="Count", color_continuous_scale="Teal",
                    title="Flight Purpose at Time of Event", text="Count",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(t=40, b=10), yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with ch:
                light = df["light_condition"].dropna().value_counts().reset_index()
                light.columns = ["Light", "Count"]
                light_colors = {
                    "Day": "#f39c12", "Night": "#1a252f",
                    "Dusk": "#e67e22", "Dawn": "#8e44ad", "Unknown": "#95a5a6",
                }
                fig = px.pie(
                    light, names="Light", values="Count",
                    color="Light", color_discrete_map=light_colors,
                    title="Light Conditions",
                )
                fig.update_traces(textinfo="percent+label", hole=0.4)
                fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.info(
                "77% of cases occurred under VFR (Visual Flight Rules) and 83% in "
                "daylight, indicating most maintenance-related events are not "
                "caused by poor visibility. Cessna accounts for 29% of all cases, "
                "reflecting its dominance in general aviation. Personal and "
                "recreational flights make up 53% of the dataset."
            )


# ===========================================================================
# TAB 3   ARCHITECTURE
# ===========================================================================
with tab3:
    st.title("System Architecture")
    st.markdown(
        "A two-phase system: a one-time **offline build** that prepares the "
        "knowledge base, and an **online agent pipeline** that answers each query."
    )

    # Phase 1
    st.markdown("## Phase 1   Offline Build")
    st.markdown(
        "Run once via `python scripts/build_index.py`. Produces all artifacts "
        "the online pipeline loads at startup."
    )

    b1, b2, b3 = st.columns(3, gap="medium")

    with b1:
        st.markdown("**Step 1   Ingest**  `src/ingest.py`")
        st.markdown(
            """
Reads the raw FAA CSV (181 coded columns) and normalises each row into a
**Case** Pydantic model with these fields:

- `case_id` from column `c5` (trailing A = Accident, I = Incident)
- `summary` derived from accident type, aircraft, and location
- `description`   the free-text narrative from `c119`
- `comments`   structured facts from the cause columns
- `severity`   derived from the damage code `c41`
- `metadata`   date, location, aircraft, weather, flight phase

Output: `artifacts/cases/cases.jsonl`   2,748 cases
            """
        )

    with b2:
        st.markdown("**Step 2   Chunk**  `src/chunker.py`")
        st.markdown(
            """
Each Case is split into up to 4 field-aware chunks:

| Chunk | Content |
|---|---|
| summary | Short title   always one chunk |
| description | Narrative, split at sentence boundaries if over 1,200 chars |
| comments | Structured facts joined with semicolons |
| metadata | Dense one-liner: date, location, aircraft, weather |

This is intentional. A query about aircraft type hits the metadata chunk.
A query about what happened hits the description chunk. No arbitrary
sliding windows.

Output: 10,992 chunks across 2,748 cases
            """
        )

    with b3:
        st.markdown("**Step 3   Index**  `src/index.py`")
        st.markdown(
            """
Two parallel indices over all 10,992 chunks:

**FAISS (semantic)**
Each chunk is encoded by `BAAI/bge-small-en-v1.5` (384 dimensions)
via fastembed, with automatic fallback to sentence-transformers,
then an offline LSA approximation if no network is available.
Stored as `IndexFlatIP` (inner product / cosine similarity).

**BM25 (lexical)**
Each chunk is tokenised (alphanumeric tokens, lowercased) and indexed
with `BM25Okapi` from the rank_bm25 library.

Outputs saved to `artifacts/index/`
            """
        )

    # Pipeline diagram
    st.markdown("---")
    st.markdown("## Phase 2   Online Query Pipeline")
    st.markdown(
        "Every query follows these five steps in order. "
        "A failure or rejection at any step stops the pipeline early."
    )

    def _step(num, title, badge, badge_col, desc, note=""):
        badge_bg = {"Pydantic": "#f0fdf4", "Claude Haiku": "#faf5ff",
                    "FAISS + BM25": "#eff6ff", "Claude Sonnet": "#fff7ed",
                    }.get(badge, "#f9fafb")
        note_html = (
            f'<div style="margin-top:7px; font-size:0.78rem; color:#6b7280; '
            f'border-left:3px solid #e5e7eb; padding-left:8px;">{note}</div>'
            if note else ""
        )
        return (
            f'<div style="display:flex; gap:14px; margin-bottom:4px;">'
            f'<div style="min-width:30px; height:30px; background:#1e3a5f; color:#fff; '
            f'border-radius:50%; display:flex; align-items:center; justify-content:center; '
            f'font-size:0.82rem; font-weight:700; flex-shrink:0;">{num}</div>'
            f'<div style="flex:1; border:1px solid #e5e7eb; border-radius:8px; '
            f'padding:10px 14px; background:#fff;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'align-items:center; margin-bottom:5px;">'
            f'<strong style="font-size:0.93rem;">{title}</strong>'
            f'<span style="font-size:0.72rem; background:{badge_bg}; color:{badge_col}; '
            f'padding:2px 9px; border-radius:4px; font-weight:600; '
            f'border:1px solid #e5e7eb;">{badge}</span></div>'
            f'<div style="font-size:0.84rem; color:#374151; line-height:1.55;">'
            f'{desc}</div>{note_html}</div></div>'
        )

    _arrow = (
        '<div style="text-align:center; padding:2px 0 2px 44px; '
        'color:#9ca3af; font-size:1.15rem;">&#8595;</div>'
    )

    _steps = [
        _step("1", "Input Validation", "Pydantic", "#166534",
              "The raw query is validated before anything else runs. "
              "Control characters and null bytes are stripped; length is capped at 2,000 "
              "characters. This is a lightweight defence against prompt injection at "
              "the schema layer, with no LLM involved.",
              "Output: a clean UserQuery string"),

        _step("2", "Scope Router", "Claude Haiku", "#6d28d9",
              "A single fast call to Haiku asks: is this question about aviation "
              "incidents or safety findings? If not, a fixed refusal is returned "
              "immediately and the pipeline stops. This keeps the assistant on-topic "
              "and prevents expensive downstream calls on irrelevant queries.",
              "Output: {in_scope: bool, reason: str} — out-of-scope ends here"),

        _step("3", "Hybrid Retrieval + RRF", "FAISS + BM25", "#1d4ed8",
              "The query runs against two indices in parallel. FAISS finds the 20 "
              "most semantically similar chunks using BGE embeddings. BM25 finds the "
              "20 most lexically relevant chunks using token overlap. "
              "Reciprocal Rank Fusion merges the two lists by rank position, "
              "rewarding chunks that score well on both. The top 8 survive. "
              "If all scores fall below 0.005, Haiku rewrites the query and "
              "retrieval retries (up to 4 attempts total).",
              "Output: 8 ranked chunks, each tagged with case_id, field, and score"),

        _step("4", "Answer Generation", "Claude Sonnet", "#c2410c",
              "Sonnet receives the question and the 8 retrieved chunks as evidence. "
              "Using chain-of-thought prompting, it reasons through which cases are "
              "relevant, what they say, and where they agree or conflict — then writes "
              "a concise answer. It is explicitly instructed that every factual claim "
              "must come from the evidence, not model memory.",
              "Output: {answer, cited_cases, rationale, confidence, caveats}"),

        _step("5", "Answer Validation", "Claude Haiku", "#6d28d9",
              "A second independent Haiku call reviews the answer without knowing "
              "that Sonnet wrote it. It checks three things: are all claims supported "
              "by the evidence (grounding)? Do the claims agree with each other "
              "(consistency)? Are the cited case IDs real and relevant (citations)? "
              "A 'revise' verdict triggers one retry with the validator's notes. "
              "A 'reject' verdict flags invented facts.",
              "Output: {verdict: pass | revise | reject, grounding_ok, consistency_ok, notes}"),
    ]

    _html_out = (
        '<div style="max-width:720px; margin:0 auto;">'
        + _arrow.join(_steps)
        + _arrow
        + '<div style="display:flex; gap:14px;">'
        + '<div style="min-width:30px; flex-shrink:0;"></div>'
        + '<div style="flex:1; border:1px solid #d1fae5; border-radius:8px; '
        + 'padding:10px 14px; background:#f0fdf4;">'
        + '<strong style="font-size:0.93rem;">AgentRunTrace</strong>'
        + '<div style="font-size:0.84rem; color:#374151; margin-top:5px; line-height:1.55;">'
        + 'The complete run is recorded: query, scope decision, every retrieval step, '
        + 'the answer, and the validator verdict. This trace is what the Live Demo tab '
        + 'surfaces when you expand "Agent trace".'
        + '</div></div></div></div>'
    )
    st.markdown(_html_out, unsafe_allow_html=True)

    # Design decisions
    st.markdown("---")
    st.markdown("## Key Design Decisions")
    st.markdown(
        "The choices below are not defaults — each one was made deliberately "
        "to address a specific constraint in this domain."
    )

    da, db = st.columns(2, gap="large")

    with da:
        with st.expander("Hybrid retrieval with RRF", expanded=True):
            st.markdown(
                "Semantic search alone misses exact FAA codes and aircraft model names. "
                "Lexical search alone misses conceptual matches like 'ran out of fuel' "
                "versus 'fuel exhaustion'. Running both and fusing the ranked lists with "
                "Reciprocal Rank Fusion (`score += 1 / (60 + rank)`) captures both. "
                "Chunks that rank highly on both signals get a compounding boost. "
                "RRF requires no score normalisation — only rank positions matter — "
                "which makes it more robust than weighted score blending."
            )

        with st.expander("Two models: Haiku for judgment, Sonnet for generation"):
            st.markdown(
                "The scope check, query refinement, and validation steps all need "
                "fast structured decisions, not long-form reasoning. Haiku handles "
                "those. Sonnet is reserved for the one step where answer quality "
                "actually matters: generation. This keeps the median latency low "
                "and concentrates cost where it has the most impact."
            )

        with st.expander("Field-aware chunking instead of sliding windows"):
            st.markdown(
                "A sliding window over a full case record would mix the narrative, "
                "structured cause codes, and metadata into the same text blobs. "
                "Instead, each case produces four separate chunks — one per field. "
                "The metadata chunk (`date | aircraft | weather | flight phase`) "
                "is independently searchable, so a query about aircraft type retrieves "
                "it directly without competing against narrative text. "
                "Every chunk carries its `case_id` and `field` for full provenance."
            )

    with db:
        with st.expander("Scope gate before any retrieval runs", expanded=True):
            st.markdown(
                "Retrieval and generation are the expensive steps. Running them on "
                "an off-topic or adversarial query wastes latency and creates an "
                "injection surface. A single Haiku call at the front checks scope "
                "and returns a fixed refusal if the query fails. The rest of the "
                "pipeline never runs. The cost of the gate is roughly 50ms; "
                "the cost of retrieval plus generation is 3-8 seconds."
            )

        with st.expander("Validator as an independent second opinion"):
            st.markdown(
                "The generator and validator are given identical evidence but run "
                "as separate LLM calls with separate system prompts. The validator "
                "does not know Sonnet wrote the answer. It checks grounding "
                "(every claim traceable to a cited case), consistency (no internal "
                "contradictions), and citations (case IDs are real and relevant). "
                "A 'revise' verdict triggers one retry with the notes appended. "
                "This catches hallucinations that confident generation can produce "
                "even with evidence in context."
            )

    st.markdown("---")
    st.markdown("## Technology Stack")
    ts1, ts2, ts3, ts4 = st.columns(4)
    with ts1:
        st.markdown("**Data**")
        st.markdown("- pandas\n- Pydantic v2\n- FAA OMIn CSV")
    with ts2:
        st.markdown("**Retrieval**")
        st.markdown("- FAISS IndexFlatIP\n- BM25Okapi\n- fastembed / BGE-small-en")
    with ts3:
        st.markdown("**LLM Layer**")
        st.markdown("- Claude Sonnet (generator)\n- Claude Haiku (judge)\n- OpenAI SDK + CMU Gateway")
    with ts4:
        st.markdown("**Infrastructure**")
        st.markdown("- Python 3.10+\n- Streamlit\n- dotenv + dataclass config")


# ===========================================================================
# TAB 4   LIVE DEMO
# ===========================================================================
with tab4:
    st.title("Live Demo")
    st.markdown(
        "Every answer produced by this system is fully traceable. "
        "For each query you can inspect: which chunks were retrieved and why "
        "(case ID, field, retrieval score), the raw evidence passed to the model, "
        "the cited case IDs, a confidence grade, and an independent validator "
        "verdict that confirms whether every claim is grounded in the evidence. "
        "Nothing in the answer comes from model memory."
    )
    st.markdown("---")

    # ------------------------------------------------------------------
    # System status diagnostic — surfaces exactly what is and isn't ready
    # ------------------------------------------------------------------
    def _check(label: str, fn) -> tuple[bool, str]:
        try:
            msg = fn()
            return True, msg or "OK"
        except Exception as exc:
            return False, str(exc)

    def _chk_faiss():
        import faiss; return f"faiss {faiss.__version__}"  # noqa: E702
    def _chk_pydantic():
        import pydantic; return f"pydantic {pydantic.__version__}"  # noqa: E702
    def _chk_dotenv():
        from importlib.metadata import version
        return f"python-dotenv {version('python-dotenv')}"
    def _chk_bm25():
        import rank_bm25; return "rank-bm25 OK"  # noqa: E702
    def _chk_openai():
        import openai; return f"openai {openai.__version__}"  # noqa: E702
    def _chk_index_files():
        needed = ["faiss.index", "bm25.pkl", "embeddings.npy", "chunks.jsonl"]
        missing = [n for n in needed if not (ROOT / "artifacts" / "index" / n).exists()]
        if missing:
            raise FileNotFoundError(f"Missing: {', '.join(missing)}")
        return "All index files present"
    def _chk_cases_file():
        p = ROOT / "artifacts" / "cases" / "cases.jsonl"
        if not p.exists():
            raise FileNotFoundError("artifacts/cases/cases.jsonl not found — run build_index.py")
        lines = sum(1 for _ in p.open())
        return f"{lines:,} cases"
    def _chk_env():
        from src.config import SETTINGS
        missing = []
        if not SETTINGS.api_key:
            missing.append("CMU_AI_GATEWAY_API_KEY")
        if not SETTINGS.base_url:
            missing.append("CMU_AI_GATEWAY_BASE_URL")
        if missing:
            raise ValueError(f"Not set in .env: {', '.join(missing)}")
        return f"Gateway configured ({SETTINGS.base_url[:40]}...)"

    checks = [
        ("faiss-cpu",        _chk_faiss),
        ("pydantic",         _chk_pydantic),
        ("python-dotenv",    _chk_dotenv),
        ("rank-bm25",        _chk_bm25),
        ("openai",           _chk_openai),
        ("Index files",      _chk_index_files),
        ("cases.jsonl",      _chk_cases_file),
        ("Gateway (.env)",   _chk_env),
    ]

    with st.expander("System Status", expanded=True):
        col_reload, _ = st.columns([1, 5])
        with col_reload:
            if st.button("Reload / Retry"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

        status_cols = st.columns(4)
        for i, (label, fn) in enumerate(checks):
            ok, msg = _check(label, fn)
            with status_cols[i % 4]:
                if ok:
                    st.success(f"**{label}**  \n{msg}")
                else:
                    st.error(f"**{label}**  \n{msg}")

    st.markdown("---")

    # Check gateway
    gw_ok = gateway_configured()
    if not gw_ok:
        st.warning(
            "Gateway not configured. Set `CMU_AI_GATEWAY_API_KEY` and "
            "`CMU_AI_GATEWAY_BASE_URL` in your `.env` file to enable the full "
            "agentic pipeline.  Retrieval-only mode is active below."
        )

    # Load index — show error inline, do not stop page rendering
    with st.spinner("Loading index..."):
        retriever, idx_err = load_retriever()

    if idx_err:
        st.error("Could not load the index. Check the System Status panel above.")
        with st.expander("Full error traceback"):
            st.code(idx_err, language="text")
        st.markdown(
            "**Common fixes:**\n"
            "- Run `python scripts/build_index.py` if the index files are missing\n"
            "- Make sure you activated your virtual environment before running Streamlit\n"
            "- Install missing packages shown above with `pip install <package>`\n"
            "- Click **Reload / Retry** in the System Status panel after fixing"
        )
    else:
        # ── Mode selector ────────────────────────────────────────────────
        demo_mode = st.radio(
            "Mode",
            ["Q&A Assistant", "Test Case Generator"],
            horizontal=True,
            help=(
                "Q&A Assistant answers questions from the case records. "
                "Test Case Generator derives structured test cases from the same evidence."
            ),
        )

        st.markdown("### Configuration")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            retrieval_mode = st.selectbox(
                "Retrieval mode",
                ["hybrid", "semantic", "lexical"],
                help="hybrid fuses FAISS and BM25 with Reciprocal Rank Fusion.",
            )
        with cc2:
            if demo_mode == "Q&A Assistant":
                prompt_style = st.selectbox(
                    "Prompt style",
                    ["cot", "zero_shot"],
                    help="cot uses chain-of-thought reasoning before writing the answer.",
                )
            else:
                tc_style = st.selectbox(
                    "Test case style",
                    ["gherkin", "scenario"],
                    help=(
                        "gherkin: formal Given/When/Then (for QA tools). "
                        "scenario: narrative prose (for training materials)."
                    ),
                )
        with cc3:
            show_evidence = st.checkbox("Show retrieved evidence", value=True)

        # ── Sample inputs (mode-aware) ────────────────────────────────────
        st.markdown("### Sample Inputs")
        if demo_mode == "Q&A Assistant":
            samples = [
                "What fuel-system findings appear most often in single-engine accidents?",
                "Which Cessna models appear most frequently in the accident records?",
                "What are the most common causes of engine failure during takeoff?",
                "Are there cases involving foreign object blockage of the fuel system?",
                "What weather conditions appear in the most severe maintenance incidents?",
            ]
        else:
            samples = [
                "Generate test cases for fuel system pre-flight inspection",
                "Test cases for engine failure during takeoff roll",
                "Maintenance check scenarios for tailwheel aircraft landing",
                "Test cases covering contaminated fuel detection procedures",
                "Safety procedure tests for night flight mechanical incidents",
            ]

        s_cols = st.columns(len(samples))
        chosen_sample = st.session_state.get(f"chosen_{demo_mode}", "")
        for col, sample in zip(s_cols, samples):
            with col:
                if st.button(
                    sample[:48] + "...",
                    use_container_width=True,
                    key=f"sbtn_{demo_mode[:3]}_{sample[:18]}",
                ):
                    chosen_sample = sample
                    st.session_state[f"chosen_{demo_mode}"] = chosen_sample

        # ── Query input ───────────────────────────────────────────────────
        st.markdown("### Input")
        placeholder = (
            "e.g. What maintenance issues are most common before takeoff accidents?"
            if demo_mode == "Q&A Assistant"
            else "e.g. Generate test cases for fuel system pre-flight inspection"
        )
        query = st.text_area(
            "Topic or question (must relate to aviation incidents or maintenance):",
            value=chosen_sample,
            height=80,
            placeholder=placeholder,
            key=f"query_{demo_mode}",
        )

        run_btn = st.button("Run", type="primary", disabled=not bool(query.strip()))

        if run_btn and query.strip():
            st.markdown("---")

            if gw_ok:

                # ── Q&A pipeline ──────────────────────────────────────────
                if demo_mode == "Q&A Assistant":
                    try:
                        from src.agents.planner import run as agent_run
                    except ImportError as e:
                        st.error(f"Could not import agent pipeline: {e}")
                        agent_run = None

                    if agent_run is not None:
                        with st.status("Running pipeline...", expanded=True) as status:
                            st.write("Scope check...")
                            try:
                                result = agent_run(
                                    query,
                                    retriever,
                                    prompt_style=prompt_style,
                                    retrieval_mode=retrieval_mode,
                                )
                                status.update(label="Complete", state="complete")
                            except Exception as exc:
                                status.update(label="Error", state="error")
                                st.error(f"Pipeline error: {exc}")
                                st.code(traceback.format_exc(), language="text")
                                result = None

                        if result is not None:
                            if result.out_of_scope_message:
                                st.warning(result.out_of_scope_message)
                            else:
                                trace = result.trace

                                with st.expander("Agent trace", expanded=False):
                                    if trace.scope:
                                        st.markdown(
                                            f"**Scope check:** in_scope = "
                                            f"`{trace.scope.in_scope}` — "
                                            f"{trace.scope.reason}"
                                        )
                                    for step in trace.react_steps:
                                        st.markdown(
                                            f"- Step {step['step']}: "
                                            f"{step['n_retrieved']} chunks | "
                                            f"query: `{step['query'][:80]}` | "
                                            f"top score: `{step['top_score']:.4f}`"
                                        )

                                if show_evidence and trace.retrieved:
                                    with st.expander(
                                        f"Retrieved evidence "
                                        f"({len(trace.retrieved)} chunks)",
                                        expanded=False,
                                    ):
                                        for i, r in enumerate(trace.retrieved, 1):
                                            c = r.chunk
                                            st.markdown(
                                                f"**{i}.** `{c.case_id}` — "
                                                f"field: `{c.field}` — "
                                                f"score: `{r.score:.4f}`"
                                            )
                                            st.markdown(
                                                f"> {c.text[:300]}"
                                                + ("..." if len(c.text) > 300 else "")
                                            )

                                if trace.answer:
                                    ans = trace.answer
                                    st.markdown("### Answer")
                                    st.markdown(ans.answer)
                                    st.markdown("---")
                                    am1, am2 = st.columns(2)
                                    with am1:
                                        st.markdown(
                                            f"**Confidence:** `{ans.confidence}`"
                                        )
                                        if ans.caveats:
                                            st.markdown(f"**Caveats:** {ans.caveats}")
                                    with am2:
                                        if ans.cited_cases:
                                            st.markdown(
                                                "**Cited cases:** "
                                                + ", ".join(
                                                    f"`{c}`" for c in ans.cited_cases
                                                )
                                            )
                                        st.markdown(f"**Rationale:** {ans.rationale}")

                                if trace.verdict:
                                    v = trace.verdict
                                    vlabel = {
                                        "pass": "Validator: Pass",
                                        "revise": "Validator: Revised",
                                        "reject": "Validator: Rejected",
                                    }.get(v.verdict, v.verdict)
                                    with st.expander(vlabel, expanded=True):
                                        vc1, vc2, vc3 = st.columns(3)
                                        vc1.metric(
                                            "Grounding",
                                            "Pass" if v.grounding_ok else "Fail",
                                        )
                                        vc2.metric(
                                            "Consistency",
                                            "Pass" if v.consistency_ok else "Fail",
                                        )
                                        vc3.metric("Verdict", v.verdict.upper())
                                        if v.notes:
                                            st.markdown(f"**Notes:** {v.notes}")

                # ── Test case generator ───────────────────────────────────
                else:
                    try:
                        from src.agents.test_case_generator import (
                            generate_test_cases,
                        )
                        from src.agents.scope_router import route as scope_route
                    except ImportError as e:
                        st.error(f"Could not import test case generator: {e}")
                        generate_test_cases = None
                        scope_route = None

                    if generate_test_cases is not None:
                        with st.status(
                            "Generating test cases...", expanded=True
                        ) as tc_status:
                            try:
                                # Scope check first
                                st.write("Scope check...")
                                scope = scope_route(query)
                                if not scope.in_scope:
                                    tc_status.update(
                                        label="Out of scope", state="error"
                                    )
                                    st.warning(
                                        "This topic is outside the aviation "
                                        "knowledge base. Test cases can only be "
                                        "generated from FAA incident records."
                                    )
                                else:
                                    # Retrieve evidence
                                    st.write("Retrieving evidence...")
                                    tc_hits = retriever.retrieve(
                                        query, mode=retrieval_mode
                                    )

                                    # Generate both styles side by side
                                    st.write(
                                        f"Generating {tc_style} test cases..."
                                    )
                                    tc_result = generate_test_cases(
                                        query, tc_hits, style=tc_style
                                    )

                                    tc_status.update(
                                        label="Complete", state="complete"
                                    )

                                    # Show evidence
                                    if show_evidence and tc_hits:
                                        with st.expander(
                                            f"Retrieved evidence "
                                            f"({len(tc_hits)} chunks)",
                                            expanded=False,
                                        ):
                                            for i, r in enumerate(tc_hits, 1):
                                                c = r.chunk
                                                st.markdown(
                                                    f"**{i}.** `{c.case_id}` — "
                                                    f"field: `{c.field}` — "
                                                    f"score: `{r.score:.4f}`"
                                                )
                                                st.markdown(
                                                    f"> {c.text[:250]}"
                                                    + (
                                                        "..."
                                                        if len(c.text) > 250
                                                        else ""
                                                    )
                                                )

                                    # Coverage and gaps
                                    cov1, cov2 = st.columns(2)
                                    with cov1:
                                        if tc_result.coverage_note:
                                            st.success(
                                                f"**Coverage:** "
                                                f"{tc_result.coverage_note}"
                                            )
                                    with cov2:
                                        if tc_result.gaps:
                                            st.warning(
                                                f"**Gaps:** {tc_result.gaps}"
                                            )


                                    # Render test cases
                                    risk_colors = {
                                        "LOW": "#d4edda",
                                        "MEDIUM": "#fff3cd",
                                        "HIGH": "#f8d7da",
                                        "CRITICAL": "#721c24",
                                    }
                                    risk_text = {
                                        "LOW": "#155724",
                                        "MEDIUM": "#856404",
                                        "HIGH": "#721c24",
                                        "CRITICAL": "#f8d7da",
                                    }
                                    st.markdown(
                                        f"### {len(tc_result.test_cases)}"
                                        f" test cases generated"
                                    )
                                    for tc in tc_result.test_cases:
                                        bg = risk_colors.get(tc.risk_level, "#fff3cd")
                                        fg = risk_text.get(tc.risk_level, "#856404")
                                        badge = (
                                            f'<span style="background:{bg};'
                                            f'color:{fg};padding:2px 8px;'
                                            f'border-radius:4px;font-size:0.8em;'
                                            f'font-weight:600;">'
                                            f'{tc.risk_level}</span>'
                                        )
                                        st.markdown(
                                            f"**{tc.test_id}** — {tc.title} "
                                            + badge,
                                            unsafe_allow_html=True,
                                        )
                                        col_g, col_w, col_t = st.columns(3)
                                        col_g.markdown(
                                            f"**Given**\n\n{tc.given}"
                                        )
                                        col_w.markdown(
                                            f"**When**\n\n{tc.when}"
                                        )
                                        col_t.markdown(
                                            f"**Then**\n\n{tc.then}"
                                        )
                                        if tc.source_case_ids:
                                            ids = ", ".join(
                                                f"`{c}`"
                                                for c in tc.source_case_ids
                                            )
                                            st.caption(
                                                f"Source cases: {ids}"
                                            )
                                        st.divider()
                            except Exception as e:
                                tc_status.update(
                                    label="Error", state="error"
                                )
                                st.error(f"Test case generation failed: {e}")

# ---------------------------------------------------------------------------
_pad = None
_pad = None
_pad = None
_pad = None
_pad = None

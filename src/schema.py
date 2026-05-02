"""Pydantic models for normalized cases, queries, retrieval, and agent I/O."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------- Knowledge base ----------

class CaseMetadata(BaseModel):
    event_date: Optional[str] = None      # ISO yyyy-mm-dd if parseable
    event_year: Optional[int] = None
    state: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    aircraft_make: Optional[str] = None
    aircraft_model: Optional[str] = None
    aircraft_class: Optional[str] = None  # e.g., MONOPLANE-LOW WING
    weight_class: Optional[str] = None    # e.g., UNDER 12501 LBS
    weather: Optional[str] = None
    light_condition: Optional[str] = None  # Day/Night/Dawn/Dusk
    flight_phase: Optional[str] = None
    flight_purpose: Optional[str] = None  # Personal, Commercial, etc.


class Case(BaseModel):
    """Jira-ticket-style normalized representation of a single FAA record."""

    case_id: str = Field(..., description="Unique ID (FAA c5)")
    summary: str = Field(..., description="Short title derived from accident-type + event")
    description: str = Field(..., description="Full narrative (FAA c119)")
    status: str = Field(..., description="Report status: ACCIDENT or INCIDENT")
    comments: list[str] = Field(default_factory=list, description="Findings/causes/notes")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"
    metadata: CaseMetadata = Field(default_factory=CaseMetadata)


# ---------- Retrieval ----------

class Chunk(BaseModel):
    chunk_id: str
    case_id: str
    field: Literal["summary", "description", "comments", "metadata"]
    text: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: Literal["semantic", "lexical", "hybrid"] = "hybrid"


# ---------- Query / Agent I/O ----------

class UserQuery(BaseModel):
    """Sanitized query envelope. Defends against trivial prompt injection at the
    schema layer by enforcing length and stripping control characters."""

    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def _scrub(cls, v: str) -> str:
        # Strip null bytes and most control chars; keep newlines and tabs.
        cleaned = "".join(ch for ch in v if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        return cleaned.strip()


class ScopeDecision(BaseModel):
    in_scope: bool
    reason: str


class GeneratedAnswer(BaseModel):
    answer: str
    cited_cases: list[str]
    rationale: str
    confidence: Literal["low", "medium", "high"] = "medium"
    caveats: Optional[str] = None


class ValidatorVerdict(BaseModel):
    verdict: Literal["pass", "revise", "reject"]
    grounding_ok: bool
    consistency_ok: bool
    notes: str


class AgentRunTrace(BaseModel):
    query: str
    scope: Optional[ScopeDecision] = None
    react_steps: list[dict] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    answer: Optional[GeneratedAnswer] = None
    verdict: Optional[ValidatorVerdict] = None


# ---------- Test case generation ----------

class TestCase(BaseModel):
    """A single structured test case derived from retrieved FAA evidence."""

    test_id: str = Field(..., description="Sequential ID, e.g. TC-001")
    title: str = Field(..., description="One-line description of what is being tested")
    category: Literal[
        "failure_mode",
        "maintenance_check",
        "safety_procedure",
        "edge_case",
    ] = Field(..., description="Type of test case")
    given: str = Field(..., description="Preconditions / system state before the test")
    when: str = Field(..., description="Action taken or condition triggered")
    then: str = Field(..., description="Expected outcome or pass criterion")
    source_case_ids: list[str] = Field(
        default_factory=list,
        description="FAA case_ids from which this test case is derived",
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"


class TestCaseSet(BaseModel):
    """The full output of a test-case generation run."""

    query: str
    style: Literal["gherkin", "scenario"]
    test_cases: list[TestCase] = Field(default_factory=list)
    coverage_note: str = Field(
        default="",
        description="Which aspects of the query are covered by the generated tests",
    )
    gaps: str = Field(
        default="",
        description="Failure modes or scenarios not covered due to insufficient evidence",
    )

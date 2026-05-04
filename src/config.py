"""Central configuration loaded from environment / .env / Streamlit secrets.

Priority order for each setting:
  1. OS environment variable (set by Streamlit Cloud secrets injection, CI, etc.)
  2. Streamlit st.secrets (when running inside a Streamlit app)
  3. .env file in the project root (local development)
  4. Hardcoded default

All tunables live here so the rest of the codebase reads from one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of src/) — no-op if file absent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)  # don't override real env vars


def _st_secret(name: str) -> str | None:
    """Read from Streamlit secrets if running inside a Streamlit app."""
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    # 1. Real environment variable (Streamlit Cloud injects secrets here)
    val = os.getenv(name)
    # 2. Streamlit st.secrets fallback (catches nested / un-injected secrets)
    if not val:
        val = _st_secret(name)
    # 3. Explicit default
    if not val:
        val = default
    if required and not val:
        raise RuntimeError(
            f"Missing required env var {name}. "
            "Set it in .env (local) or Streamlit Cloud secrets (deployed)."
        )
    return val or ""


@dataclass(frozen=True)
class Settings:
    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    cases_dir: Path = PROJECT_ROOT / "artifacts" / "cases"
    index_dir: Path = PROJECT_ROOT / "artifacts" / "index"
    gold_dir: Path = PROJECT_ROOT / "gold_standard"

    # Source data file (FAA Maintenance text data, no NA description rows)
    faa_csv: Path = (
        PROJECT_ROOT
        / "data"
        / "nd-crane-trusted_ke-0ac3387"
        / "OMIn_dataset"
        / "data"
        / "FAA_data"
        / "Maintenance_Text_data_nona.csv"
    )

    # Gateway / LLM
    api_key: str = _env("CMU_AI_GATEWAY_API_KEY", "")
    base_url: str = _env("CMU_AI_GATEWAY_BASE_URL", "")
    gen_model: str = _env("GEN_MODEL", "us.anthropic.claude-sonnet-4-6")
    judge_model: str = _env(
        "JUDGE_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    # Embeddings
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # Retrieval
    top_k_semantic: int = int(_env("TOP_K_SEMANTIC", "20"))
    top_k_lexical: int = int(_env("TOP_K_LEXICAL", "20"))
    top_k_final: int = int(_env("TOP_K_FINAL", "8"))
    rrf_k: int = int(_env("RRF_K", "60"))

    # Agent loop
    max_react_steps: int = int(_env("MAX_REACT_STEPS", "4"))

    # Per-role output token budgets (tune up if models truncate responses)
    max_tokens_judge: int = int(_env("MAX_TOKENS_JUDGE", "800"))
    max_tokens_generator: int = int(_env("MAX_TOKENS_GENERATOR", "4096"))

    # LLM retry / backoff
    llm_max_retries: int = int(_env("LLM_MAX_RETRIES", "3"))
    llm_retry_base_delay: float = float(_env("LLM_RETRY_BASE_DELAY", "2.0"))

    def require_gateway(self) -> None:
        if not self.api_key or not self.base_url:
            raise RuntimeError(
                "Gateway not configured. Set CMU_AI_GATEWAY_API_KEY and "
                "CMU_AI_GATEWAY_BASE_URL in .env (see .env.example)."
            )


SETTINGS = Settings()

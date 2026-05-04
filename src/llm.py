"""Thin wrapper over any OpenAI-compatible chat completions endpoint.

All agent calls flow through here so we have a single place to tune timeouts,
parse JSON-mode outputs, retry on transient errors, and swap providers.

Provider notes
--------------
- Gemini (generativelanguage.googleapis.com/openai/):
    * response_format=json_object is NOT supported — we skip it and rely on
      prompt instructions + fallback JSON parsing.
    * Models can produce markdown-fenced JSON (```json … ```) even when told
      not to; the extractor strips fences automatically.
    * Responses can be truncated at max_tokens mid-JSON; _repair_truncated_json
      attempts a best-effort close of open braces/brackets.
- OpenAI / CMU Gateway:
    * json_mode works natively; no fence stripping needed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, APIStatusError

from .config import SETTINGS

log = logging.getLogger(__name__)

# Providers that do NOT support response_format=json_object
_NO_JSON_MODE_HOSTS = ("generativelanguage.googleapis.com",)

# HTTP status codes worth retrying
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _supports_json_mode() -> bool:
    return not any(h in SETTINGS.base_url for h in _NO_JSON_MODE_HOSTS)


# ---------------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` wrappers that some models emit."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    # Drop the opening fence line (```json or ```)
    inner = lines[1:] if lines[0].startswith("```") else lines
    # Drop the closing fence line if present
    if inner and inner[-1].strip() == "```":
        inner = inner[:-1]
    return "\n".join(inner).strip()


def _repair_truncated_json(text: str) -> dict | None:
    """Best-effort repair of JSON cut off mid-stream by a token limit.

    Tracks open braces, brackets, and string state, then appends the minimal
    closing sequence to make the document syntactically valid (though the last
    field may be semantically incomplete).
    """
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack and not in_string:
        return None  # nothing to repair

    closing = ('"' if in_string else "") + "".join(reversed(stack))
    candidate = text.rstrip().rstrip(",") + closing
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_json(text: str, model: str) -> dict:
    """Try progressively more tolerant strategies to parse JSON from *text*."""
    if not text or not text.strip():
        raise ValueError(f"Model returned an empty response (model={model})")

    cleaned = _strip_markdown_fences(text)

    # Strategy 1 — clean parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract outermost { … } blob (handles leading prose)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start: end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3 — repair truncated JSON (token-limit cut-off)
    blob = cleaned[start:] if start != -1 else cleaned
    repaired = _repair_truncated_json(blob)
    if repaired is not None:
        log.warning("Repaired truncated JSON from model=%s", model)
        return repaired

    raise ValueError(
        f"Could not parse JSON from model response.\n"
        f"Model: {model}\n"
        f"Response (first 400 chars): {text[:400]}"
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    model: str
    client: OpenAI

    @classmethod
    def for_model(cls, model: str) -> "LLMClient":
        SETTINGS.require_gateway()
        client = OpenAI(api_key=SETTINGS.api_key, base_url=SETTINGS.base_url)
        return cls(model=model, client=client)

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        if max_tokens is None:
            max_tokens = SETTINGS.max_tokens_judge  # conservative default

        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if json_mode and _supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(SETTINGS.llm_max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUS:
                    delay = SETTINGS.llm_retry_base_delay * (2 ** attempt)
                    log.warning(
                        "LLM %s error (attempt %d/%d), retrying in %.1fs — %s",
                        exc.status_code, attempt + 1, SETTINGS.llm_max_retries,
                        delay, exc.message,
                    )
                    time.sleep(delay)
                    last_exc = exc
                else:
                    raise
            except Exception as exc:
                raise

        raise last_exc  # type: ignore[misc]

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict:
        """Chat and return parsed JSON with multi-strategy fallback."""
        text = self.chat(
            system, user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return _extract_json(text, self.model)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def generator() -> LLMClient:
    return LLMClient.for_model(SETTINGS.gen_model)


def judge() -> LLMClient:
    return LLMClient.for_model(SETTINGS.judge_model)

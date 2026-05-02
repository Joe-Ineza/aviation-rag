"""Thin wrapper over the CMU AI Gateway (OpenAI-compatible chat completions).

All agent calls flow through here so we have a single place to tune timeouts,
parse JSON-mode outputs, and swap providers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import SETTINGS


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
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_json(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> dict:
        """Chat with JSON-mode and tolerant fallback parsing."""
        text = self.chat(
            system, user, temperature=temperature, max_tokens=max_tokens, json_mode=True
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some gateways ignore json_mode; try to recover the first JSON blob.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise


def generator() -> LLMClient:
    return LLMClient.for_model(SETTINGS.gen_model)


def judge() -> LLMClient:
    return LLMClient.for_model(SETTINGS.judge_model)

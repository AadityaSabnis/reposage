"""Ollama LLM client (local dev default).

Uses the /api/chat endpoint so the model's chat template is applied, with
a strict system message. Mirrors the PRAVA project's `ollama_generate`:
temperature 0, generous context, keep_alive so the model stays resident
between questions.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from app.config import settings
from app.llm.base import GROUNDING_SYSTEM, LLMClient


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, url: str = None, model: str = None, keep_alive: str = None):
        self.url = (url or settings.ollama_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.keep_alive = keep_alive or settings.ollama_keep_alive

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 768},
        }

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Stream content deltas from Ollama's newline-delimited JSON."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            async with client.stream(
                "POST", f"{self.url}/api/chat", json=self._payload(prompt, stream=True)
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "replace")
                    raise RuntimeError(f"Ollama HTTP {r.status_code}: {body[:300]}")
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise RuntimeError(f"Ollama error: {obj['error']}")
                    piece = obj.get("message", {}).get("content")
                    if piece:
                        yield piece
                    if obj.get("done"):
                        break

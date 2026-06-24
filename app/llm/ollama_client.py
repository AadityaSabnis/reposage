"""Ollama LLM client (local dev default).

Uses the /api/chat endpoint so the model's chat template is applied, with
a strict system message. Mirrors the PRAVA project's `ollama_generate`:
temperature 0, generous context, keep_alive so the model stays resident
between questions.
"""
from __future__ import annotations

import httpx

from app.config import settings
from app.llm.base import GROUNDING_SYSTEM, LLMClient


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, url: str = None, model: str = None, keep_alive: str = None):
        self.url = (url or settings.ollama_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.keep_alive = keep_alive or settings.ollama_keep_alive

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 768},
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            r = await client.post(f"{self.url}/api/chat", json=payload)
            if r.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
            j = r.json()
            if j.get("error"):
                raise RuntimeError(f"Ollama error: {j['error']}")
            return (j.get("message", {}).get("content") or "").strip()

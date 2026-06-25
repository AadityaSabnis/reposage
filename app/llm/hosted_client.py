"""Hosted LLM client (deployed demo).

OpenAI-compatible Chat Completions — works with Groq (default) and OpenAI
by swapping HOSTED_BASE_URL / HOSTED_MODEL and the API key. Used when
LLM_PROVIDER=hosted, because Ollama won't run on most free hosting tiers.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from app.config import settings
from app.llm.base import GROUNDING_SYSTEM, LLMClient


class HostedClient(LLMClient):
    name = "hosted"

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None):
        self.base_url = (base_url or settings.hosted_base_url).rstrip("/")
        self.model = model or settings.hosted_model
        self.api_key = api_key or settings.hosted_api_key

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Stream content deltas from an OpenAI-compatible SSE response."""
        if not self.api_key:
            raise RuntimeError(
                "No hosted API key set. Set GROQ_API_KEY or OPENAI_API_KEY "
                "(LLM_PROVIDER=hosted)."
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 768,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "replace")
                    raise RuntimeError(f"Hosted LLM HTTP {r.status_code}: {body[:300]}")
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta


def make_llm_client() -> LLMClient:
    """Factory honoring LLM_PROVIDER with Ollama→Groq fallback.

    If LLM_PROVIDER=ollama (the default), tries to connect to the local Ollama
    server. If Ollama is unreachable, silently falls back to Groq (needs
    GROQ_API_KEY in .env). If LLM_PROVIDER=hosted, uses Groq/OpenAI directly.
    """
    import logging
    from app.llm.ollama_client import OllamaClient

    log = logging.getLogger("reposage")

    if settings.llm_provider.lower() == "hosted":
        log.info("Using hosted LLM (Groq/OpenAI)")
        return HostedClient()

    # Try Ollama first (no API key needed, runs locally).
    try:
        import httpx
        # Quick connection check with a 2-second timeout.
        client = OllamaClient()
        with httpx.Client(timeout=2.0) as c:
            r = c.head(f"{client.url}/api/tags")
            if r.status_code == 200:
                log.info("Using local Ollama")
                return client
    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        pass

    # Ollama not available; fall back to Groq if API key is set.
    if settings.hosted_api_key:
        log.info("Ollama unavailable, falling back to hosted LLM (Groq/OpenAI)")
        return HostedClient()

    # Neither works -> error.
    raise RuntimeError(
        "No LLM available. Either run Ollama, or set GROQ_API_KEY/OPENAI_API_KEY in .env"
    )

"""Hosted LLM client (deployed demo).

OpenAI-compatible Chat Completions — works with Groq (default) and OpenAI
by swapping HOSTED_BASE_URL / HOSTED_MODEL and the API key. Used when
LLM_PROVIDER=hosted, because Ollama won't run on most free hosting tiers.
"""
from __future__ import annotations

import httpx

from app.config import settings
from app.llm.base import GROUNDING_SYSTEM, LLMClient


class HostedClient(LLMClient):
    name = "hosted"

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None):
        self.base_url = (base_url or settings.hosted_base_url).rstrip("/")
        self.model = model or settings.hosted_model
        self.api_key = api_key or settings.hosted_api_key

    async def generate(self, prompt: str) -> str:
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
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            if r.status_code != 200:
                raise RuntimeError(f"Hosted LLM HTTP {r.status_code}: {r.text[:300]}")
            j = r.json()
            return (j["choices"][0]["message"]["content"] or "").strip()


def make_llm_client() -> LLMClient:
    """Factory honoring LLM_PROVIDER."""
    from app.llm.ollama_client import OllamaClient

    if settings.llm_provider.lower() == "hosted":
        return HostedClient()
    return OllamaClient()

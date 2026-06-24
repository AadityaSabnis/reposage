"""LLM client interface + the grounding prompt contract.

`LLMClient.generate(prompt) -> str` is the seam: Ollama for local dev,
an OpenAI-compatible hosted API for the deployed demo. The grounding
system prompt and the evidence-prompt builder live here so both clients
(and the tests) share exactly one definition.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.retriever import RetrievedChunk

# Exact string returned (and instructed) when the answer isn't in the chunks.
NOT_FOUND_MESSAGE = "I couldn't find an answer to that in the indexed repository."

GROUNDING_SYSTEM = (
    "You are RepoSage, a precise code assistant. You answer questions about a "
    "codebase using ONLY the numbered code chunks provided in the user message.\n\n"
    "Rules you must never break:\n"
    "1. Base every statement ONLY on the provided chunks. Use no outside knowledge "
    "about the project and make no assumptions about code you were not shown.\n"
    f"2. If the chunks do not contain the answer, reply with EXACTLY: "
    f"'{NOT_FOUND_MESSAGE}' and nothing else.\n"
    "3. Never invent, guess, or alter file paths or line numbers. Cite only the "
    "exact `file_path:start_line-end_line` labels shown on the chunks.\n"
    "4. Cite the chunk(s) supporting each claim inline, e.g. (app/main.py:10-42). "
    "Every factual sentence must carry at least one citation.\n"
    "5. Be concise and technical. No preamble, no markdown headings, no filler."
)


class LLMClient(ABC):
    """Minimal interface: text in, text out (async)."""

    name: str = "base"

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        ...


def build_prompt(question: str, hits: "List[RetrievedChunk]") -> str:
    """Assemble the user message: numbered, citation-labeled code chunks
    followed by the question."""
    blocks: List[str] = []
    for i, h in enumerate(hits, 1):
        header = f"[{i}] {h.citation}  (chunk: {h.symbol_type} {h.symbol_name}, {h.language})"
        blocks.append(f"{header}\n```{h.language}\n{h.snippet}\n```")
    evidence = "\n\n".join(blocks) if blocks else "(no chunks retrieved)"

    return (
        "CODE CHUNKS:\n"
        f"{evidence}\n\n"
        "------\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the chunks above. Cite each claim with its "
        "`file_path:start_line-end_line` label. If the chunks do not contain the "
        f"answer, reply exactly: '{NOT_FOUND_MESSAGE}'"
    )

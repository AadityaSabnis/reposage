"""Q&A endpoint: retrieve grounded chunks, synthesize a cited answer."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.deps import get_indexer, get_llm, get_retriever
from app.llm.base import NOT_FOUND_MESSAGE, build_prompt

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


@router.post("/ask")
async def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    top_k = req.top_k or settings.top_k
    retriever = get_retriever()

    # No index yet -> be explicit rather than calling the LLM with nothing.
    if get_indexer().metadata.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="No repository indexed yet. POST /repos/index first.",
        )

    hits = retriever.retrieve(question, top_k)
    if not hits:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "citations": [],
            "model": get_llm().name,
            "retrieved": 0,
        }

    prompt = build_prompt(question, hits)
    try:
        answer = await get_llm().generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return {
        "answer": answer,
        "citations": [h.to_citation_dict() for h in hits],
        "model": get_llm().name,
        "retrieved": len(hits),
    }

"""Q&A endpoints: retrieve grounded chunks, synthesize a cited answer.

Two routes share the same retrieval core:
  * POST /ask          — blocking; returns the whole answer + citations once.
  * POST /ask/stream   — Server-Sent Events; emits citations *immediately*
                         (evidence appears before the model finishes), then
                         streams the answer token-by-token.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.deps import get_llm, get_metadata_store, get_retriever
from app.llm.base import NOT_FOUND_MESSAGE, build_prompt

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


def _clean(req: AskRequest) -> tuple[str, int]:
    """Validate the request; return (question, top_k) or raise."""
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")
    if get_metadata_store().count() == 0:
        raise HTTPException(
            status_code=409,
            detail="No repository indexed yet. POST /repos/index first.",
        )
    return question, (req.top_k or settings.top_k)


@router.post("/ask")
async def ask(req: AskRequest):
    question, top_k = _clean(req)
    llm = get_llm()
    hits = await run_in_threadpool(get_retriever().retrieve, question, top_k)
    if not hits:
        return {"answer": NOT_FOUND_MESSAGE, "citations": [], "model": llm.name, "retrieved": 0}

    try:
        answer = await llm.generate(build_prompt(question, hits))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return {
        "answer": answer,
        "citations": [h.to_citation_dict() for h in hits],
        "model": llm.name,
        "retrieved": len(hits),
    }


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    question, top_k = _clean(req)
    llm = get_llm()
    # Retrieval is sync (embeds the query) — keep it off the event loop.
    hits = await run_in_threadpool(get_retriever().retrieve, question, top_k)

    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    async def gen():
        # 1) Citations first — evidence renders instantly, before the model runs.
        yield sse({
            "type": "citations",
            "citations": [h.to_citation_dict() for h in hits],
            "model": llm.name,
            "retrieved": len(hits),
        })
        if not hits:
            yield sse({"type": "token", "text": NOT_FOUND_MESSAGE})
            yield sse({"type": "done"})
            return
        # 2) Stream the answer token-by-token.
        try:
            async for piece in llm.astream(build_prompt(question, hits)):
                if piece:
                    yield sse({"type": "token", "text": piece})
        except Exception as e:
            yield sse({"type": "error", "detail": f"LLM call failed: {e}"})
            return
        yield sse({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

"""RepoSage FastAPI application.

Wires routes, CORS, the static chat UI, and a /stats endpoint that makes
the incremental-index claim observable ("exactly N chunks re-embedded").
On startup it loads any previously persisted FAISS index from disk.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.deps import get_embedder, get_indexer, get_llm
from app.routes import ask_routes, index_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reposage")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    indexer = get_indexer()
    try:
        if indexer.load():
            log.info(
                "Loaded persisted index: %d vectors, %d chunks across %d files",
                indexer.vector_store.ntotal,
                indexer.metadata.count(),
                indexer.metadata.file_count(),
            )
        else:
            log.info("No persisted index found — POST /repos/index to build one.")
    except Exception as e:  # don't block startup on a bad/old index file
        log.warning("Could not load persisted index: %s", e)
    log.info("LLM provider: %s", settings.llm_provider)

    # Pre-load the embedding model in the background so the first-run
    # download happens at boot (visible in logs + via /model/status) rather
    # than stalling the first index request.
    def _warm():
        try:
            log.info("Loading embedding model '%s'…", settings.embedding_model_path)
            get_embedder().warmup()
            log.info("Embedding model ready.")
        except Exception as e:
            log.warning("Embedding model failed to load: %s", e)

    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="RepoSage", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # portfolio demo: no auth, open CORS
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(index_routes.router)
app.include_router(ask_routes.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/status")
def model_status():
    """Embedding-model readiness, for the UI's first-run download banner."""
    emb = get_embedder()
    return {
        "status": getattr(emb, "status", "ready"),   # idle|loading|ready|error
        "model": getattr(emb, "model_name", settings.embedding_model_path),
        "error": getattr(emb, "error", None),
    }


@app.get("/stats")
def stats():
    """Index stats + the last incremental-update result.

    This is the endpoint to watch during the demo: after editing one
    function and hitting /webhook/git-push, `last_incremental_update.embedded`
    is 1, not the whole repo.
    """
    s = get_indexer().stats()
    s["llm_provider"] = get_llm().name
    return s


@app.get("/")
def home():
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return JSONResponse({"message": "RepoSage API. Frontend not found.", "docs": "/docs"})

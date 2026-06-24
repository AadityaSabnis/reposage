"""Indexing endpoints: full index + incremental webhook."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.deps import get_indexer

router = APIRouter(tags=["indexing"])


class IndexRequest(BaseModel):
    repo_path: Optional[str] = None  # defaults to settings.repo_path


class GitPushRequest(BaseModel):
    changed_files: List[str] = []
    deleted_files: List[str] = []


@router.post("/repos/index")
def index_repo(req: IndexRequest):
    """Full (re)index of a repository. Heavy — embeds every chunk once."""
    indexer = get_indexer()
    try:
        return indexer.full_index(req.repo_path)
    except Exception as e:  # surface the failure rather than a bare 500
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@router.post("/webhook/git-push")
def git_push(req: GitPushRequest):
    """Incremental update from a list of changed/deleted paths.

    Wire this to a GitHub Actions step (post the diff's file lists) or call
    it directly. Returns a stats object proving how little was re-embedded.
    """
    indexer = get_indexer()
    if not req.changed_files and not req.deleted_files:
        raise HTTPException(status_code=400, detail="No changed_files or deleted_files given.")
    try:
        return indexer.incremental_update(req.changed_files, req.deleted_files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Incremental update failed: {e}")

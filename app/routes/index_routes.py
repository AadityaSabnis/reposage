"""Indexing endpoints: full index (local path or Git URL) + incremental webhook."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.deps import get_indexer, get_retriever
from app.services.git_clone import GitCloneError, clone_repo

router = APIRouter(tags=["indexing"])


class IndexRequest(BaseModel):
    repo_path: Optional[str] = None  # defaults to settings.repo_path


class GitIndexRequest(BaseModel):
    git_url: str
    branch: Optional[str] = None  # defaults to the remote's default branch


class GitPushRequest(BaseModel):
    changed_files: List[str] = []
    deleted_files: List[str] = []


def _sync_retriever(repo_path: str) -> None:
    """Point the (cached) retriever at the just-indexed checkout.

    Snippets are read from ``retriever.repo_path`` at query time, so it must
    track whatever the indexer last indexed — otherwise citations come back
    empty whenever the indexed path differs from the server's cwd.
    """
    get_retriever().repo_path = Path(repo_path).resolve()


@router.post("/repos/index")
def index_repo(req: IndexRequest):
    """Full (re)index of a local repository. Heavy — embeds every chunk once."""
    indexer = get_indexer()
    try:
        stats = indexer.full_index(req.repo_path)
    except Exception as e:  # surface the failure rather than a bare 500
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    _sync_retriever(indexer.repo_path)
    return stats


@router.post("/repos/index-git")
def index_repo_from_git(req: GitIndexRequest):
    """Clone a remote repo from a Git URL, then full-index it.

    A separate source from /repos/index (local path); the pipeline is the
    same. The cloned checkout keeps its ``remote.origin.url``, so GitHub
    citation links are recovered automatically.
    """
    try:
        local_path = clone_repo(req.git_url, req.branch)
    except GitCloneError as e:  # bad URL / clone failure -> client error
        raise HTTPException(status_code=400, detail=str(e))

    indexer = get_indexer()
    try:
        stats = indexer.full_index(local_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    _sync_retriever(local_path)
    stats["git_url"] = req.git_url
    return stats


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

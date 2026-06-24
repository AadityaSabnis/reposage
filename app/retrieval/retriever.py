"""Retrieval: embed a query, search FAISS, hydrate hits with metadata.

The chunk *text* is not stored in SQLite (per the schema), so the snippet
is read back from the file on disk using the stored line range. That keeps
the citation honest: it shows the current source at the cited lines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.indexing.embedder import Embedder
from app.indexing.metadata_store import MetadataStore
from app.indexing.vector_store import VectorStore

MAX_SNIPPET_LINES = 60
MAX_SNIPPET_CHARS = 3000


@dataclass
class RetrievedChunk:
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    symbol_type: str
    language: str
    score: float
    snippet: str
    github_url: Optional[str]

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_citation_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        return d


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        repo_path: Optional[str] = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata = metadata_store
        self.repo_path = Path(repo_path or settings.repo_path).resolve()

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        if self.vector_store.ntotal == 0 or not query.strip():
            return []

        q_emb = self.embedder.embed([query])
        scores, ids = self.vector_store.search(q_emb, top_k)
        metas = self.metadata.get_by_ids(ids)
        score_by_id = dict(zip(ids, scores))

        repo_meta = self.metadata.all_meta()
        hits: List[RetrievedChunk] = []
        for m in metas:
            hits.append(
                RetrievedChunk(
                    file_path=m.file_path,
                    start_line=m.start_line,
                    end_line=m.end_line,
                    symbol_name=m.symbol_name,
                    symbol_type=m.symbol_type,
                    language=m.language,
                    score=round(score_by_id.get(m.chunk_id, 0.0), 4),
                    snippet=self._read_snippet(m.file_path, m.start_line, m.end_line),
                    github_url=self._github_url(repo_meta, m.file_path, m.start_line, m.end_line),
                )
            )
        return hits

    # ── helpers ─────────────────────────────────────────────────────
    def _read_snippet(self, file_path: str, start: int, end: int) -> str:
        abs_path = self.repo_path / file_path
        try:
            lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        # start/end are 1-based inclusive
        selected = lines[start - 1:end]
        if len(selected) > MAX_SNIPPET_LINES:
            selected = selected[:MAX_SNIPPET_LINES] + ["    # … (truncated) …"]
        snippet = "\n".join(selected)
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "\n    # … (truncated) …"
        return snippet

    @staticmethod
    def _github_url(repo_meta: dict, file_path: str, start: int, end: int) -> Optional[str]:
        owner = repo_meta.get("github_owner")
        repo = repo_meta.get("github_repo")
        commit = repo_meta.get("commit_sha") or "HEAD"
        if not owner or not repo:
            return None
        return (
            f"https://github.com/{owner}/{repo}/blob/{commit}/{file_path}"
            f"#L{start}-L{end}"
        )

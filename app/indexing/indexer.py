"""Indexer — orchestrates chunking, embedding, and the two stores.

Two entry points:
  * full_index(repo_path)                  — index a repo from scratch
  * incremental_update(changed, deleted)   — re-embed *only* what changed

The incremental path is the core engineering flex of RepoSage: a touched
file does not trigger a repo (or even file) re-embed — only symbols whose
`content_hash` actually changed are re-embedded, and symbols that merely
shifted line numbers get a cheap metadata-only update.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.chunking.registry import chunk_file, iter_source_files
from app.config import settings
from app.indexing.embedder import Embedder
from app.indexing.metadata_store import MetadataStore
from app.indexing.vector_store import VectorStore
from app.models import Chunk, INT63_MAX, base_chunk_id


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Indexer:
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
        self.repo_path = str(Path(repo_path or settings.repo_path).resolve())
        self.last_update_stats: Dict = {}
        self.last_full_index_stats: Dict = {}

    # ── id allocation ───────────────────────────────────────────────
    def _allocate_id(self, file_path: str, symbol_name: str, used: Set[int]) -> int:
        """Stable id for (file_path, symbol_name); linear-probe on collision."""
        cid = base_chunk_id(file_path, symbol_name)
        while cid in used or self.metadata.id_exists(cid):
            cid = (cid + 1) & INT63_MAX
        return cid

    # ── full index (Phase 2) ────────────────────────────────────────
    def full_index(self, repo_path: Optional[str] = None) -> Dict:
        if repo_path:
            self.repo_path = str(Path(repo_path).resolve())
        root = self.repo_path
        started = time.time()

        self.vector_store.reset()
        self.metadata.reset()

        all_chunks: List[Chunk] = []
        used: Set[int] = set()
        files = list(iter_source_files(root))
        for fpath in files:
            for c in chunk_file(root, fpath):
                c.chunk_id = self._allocate_id(c.file_path, c.symbol_name, used)
                used.add(c.chunk_id)
                all_chunks.append(c)

        # Batch embed everything once.
        embeddings = self.embedder.embed([c.text for c in all_chunks])
        self.vector_store.add([c.chunk_id for c in all_chunks], embeddings)
        for c in all_chunks:
            self.metadata.upsert(c)

        self._record_repo_meta(len(all_chunks))
        self.persist()

        stats = {
            "files_indexed": len(files),
            "chunks_indexed": len(all_chunks),
            "ntotal": self.vector_store.ntotal,
            "elapsed_sec": round(time.time() - started, 2),
            "repo_path": root,
        }
        self.last_full_index_stats = stats
        return stats

    # ── incremental update (Phase 4 — the core flex) ────────────────
    def incremental_update(
        self,
        changed_files: Optional[List[str]] = None,
        deleted_files: Optional[List[str]] = None,
    ) -> Dict:
        changed_files = changed_files or []
        deleted_files = deleted_files or []
        started = time.time()

        stats = {
            "embedded": 0,          # chunks actually re-embedded (changed or new)
            "reembed_skipped": 0,   # unchanged content -> NOT re-embedded
            "lines_updated": 0,     # unchanged content, shifted lines -> metadata only
            "removed": 0,           # chunks deleted (vanished symbols + deleted files)
            "files_changed": 0,
            "files_deleted": 0,
        }

        # 1) Deletions: drop every chunk belonging to the file.
        for raw in deleted_files:
            rel = self._rel(raw)
            ids = self.metadata.get_chunk_ids_for_file(rel)
            if ids:
                stats["removed"] += self.vector_store.remove_ids(ids)
                self.metadata.delete_chunks_for_file(rel)
            stats["files_deleted"] += 1

        # 2) Changes: diff old vs new chunks by symbol_name + content_hash.
        for raw in changed_files:
            rel = self._rel(raw)
            abs_path = Path(self.repo_path) / rel
            if not abs_path.exists():
                # "changed" but gone -> treat as a delete.
                ids = self.metadata.get_chunk_ids_for_file(rel)
                if ids:
                    stats["removed"] += self.vector_store.remove_ids(ids)
                    self.metadata.delete_chunks_for_file(rel)
                stats["files_deleted"] += 1
                continue

            old_chunks = self.metadata.get_chunks_for_file(rel)
            old_by_symbol = {m.symbol_name: m for m in old_chunks}
            new_chunks = chunk_file(self.repo_path, abs_path)
            new_symbols: Set[str] = set()
            used: Set[int] = set(self.metadata.all_chunk_ids())

            for nc in new_chunks:
                new_symbols.add(nc.symbol_name)
                existing_id = self.metadata.get_chunk_id(rel, nc.symbol_name)
                nc.chunk_id = existing_id if existing_id is not None \
                    else self._allocate_id(rel, nc.symbol_name, used)
                used.add(nc.chunk_id)

                old_meta = old_by_symbol.get(nc.symbol_name)
                if old_meta is not None and old_meta.content_hash == nc.content_hash:
                    # Unchanged body -> skip re-embedding. Only fix line numbers
                    # if they drifted (e.g. earlier code grew).
                    if (old_meta.start_line, old_meta.end_line) != (nc.start_line, nc.end_line):
                        self.metadata.update_lines(nc.chunk_id, nc.start_line, nc.end_line)
                        stats["lines_updated"] += 1
                    stats["reembed_skipped"] += 1
                    continue

                # Changed or brand-new symbol -> embed + upsert.
                emb = self.embedder.embed([nc.text])[0]
                self.vector_store.upsert(nc.chunk_id, emb)
                self.metadata.upsert(nc)
                stats["embedded"] += 1

            # Symbols that no longer exist in the file -> remove.
            for m in old_chunks:
                if m.symbol_name not in new_symbols:
                    self.vector_store.remove_ids([m.chunk_id])
                    self.metadata.delete_chunk(m.chunk_id)
                    stats["removed"] += 1

            stats["files_changed"] += 1

        self._record_repo_meta(self.metadata.count())
        self.persist()

        stats["ntotal"] = self.vector_store.ntotal
        stats["elapsed_sec"] = round(time.time() - started, 3)
        self.last_update_stats = stats
        return stats

    # ── helpers ─────────────────────────────────────────────────────
    def _rel(self, path: str) -> str:
        """Normalize an incoming path to a repo-relative POSIX path."""
        p = Path(path)
        root = Path(self.repo_path)
        if p.is_absolute():
            try:
                return p.resolve().relative_to(root).as_posix()
            except ValueError:
                return p.as_posix()
        return p.as_posix()

    def _record_repo_meta(self, chunk_count: int) -> None:
        owner, repo, commit = self._git_meta()
        meta = {
            "repo_path": self.repo_path,
            "github_owner": settings.github_owner or owner or "",
            "github_repo": settings.github_repo or repo or "",
            "commit_sha": settings.github_commit or commit or "",
            "indexed_at": _now(),
            "chunk_count": str(chunk_count),
        }
        for k, v in meta.items():
            self.metadata.set_meta(k, v)

    def _git_meta(self):
        """Best-effort (owner, repo, commit_sha) from the local git checkout."""
        def _git(*args) -> Optional[str]:
            try:
                out = subprocess.run(
                    ["git", "-C", self.repo_path, *args],
                    capture_output=True, text=True, timeout=5,
                )
                return out.stdout.strip() if out.returncode == 0 else None
            except Exception:
                return None

        commit = _git("rev-parse", "HEAD")
        url = _git("config", "--get", "remote.origin.url")
        owner = repo = None
        if url:
            m = re.search(r"github\.com[:/]+([^/]+)/(.+?)(?:\.git)?/?$", url)
            if m:
                owner, repo = m.group(1), m.group(2)
        return owner, repo, commit

    def persist(self) -> None:
        self.vector_store.save(settings.faiss_path)

    def load(self) -> bool:
        """Load a previously persisted FAISS index (metadata is already open)."""
        return self.vector_store.load(settings.faiss_path)

    def stats(self) -> Dict:
        return {
            "indexed": self.metadata.count() > 0,
            "chunks": self.metadata.count(),
            "files": self.metadata.file_count(),
            "ntotal": self.vector_store.ntotal,
            "repo": self.metadata.all_meta(),
            "last_incremental_update": self.last_update_stats,
            "last_full_index": self.last_full_index_stats,
        }

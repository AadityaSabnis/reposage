"""SQLite metadata store — one row per chunk.

Holds everything about a chunk *except* its raw text (the snippet is read
back from the file on disk at query time, so the citation always reflects
the current source). Schema matches the project spec exactly:

    chunk_id, file_path, start_line, end_line, symbol_name,
    symbol_type, language, content_hash, embedding_id

A separate `repo_meta` key/value table stores repo-level facts used to
build GitHub citation URLs (owner, repo, commit_sha, indexed_at).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

from app.models import Chunk, ChunkMeta

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY,
    file_path    TEXT    NOT NULL,
    start_line   INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    symbol_name  TEXT    NOT NULL,
    symbol_type  TEXT    NOT NULL,
    language     TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    embedding_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_file_symbol
    ON chunks(file_path, symbol_name);

CREATE TABLE IF NOT EXISTS repo_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _row_to_meta(row: sqlite3.Row) -> ChunkMeta:
    return ChunkMeta(
        chunk_id=row["chunk_id"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        symbol_name=row["symbol_name"],
        symbol_type=row["symbol_type"],
        language=row["language"],
        content_hash=row["content_hash"],
        embedding_id=row["embedding_id"],
    )


class MetadataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI may touch this from worker threads.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── writes ──────────────────────────────────────────────────────
    def upsert(self, chunk: Chunk) -> None:
        if chunk.chunk_id is None:
            raise ValueError("chunk.chunk_id must be set before upsert")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chunks (chunk_id, file_path, start_line, end_line,
                                    symbol_name, symbol_type, language,
                                    content_hash, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    start_line=excluded.start_line,
                    end_line=excluded.end_line,
                    symbol_name=excluded.symbol_name,
                    symbol_type=excluded.symbol_type,
                    language=excluded.language,
                    content_hash=excluded.content_hash,
                    embedding_id=excluded.embedding_id
                """,
                (
                    chunk.chunk_id, chunk.file_path, chunk.start_line, chunk.end_line,
                    chunk.symbol_name, chunk.symbol_type, chunk.language,
                    chunk.content_hash, chunk.chunk_id,
                ),
            )
            self._conn.commit()

    def update_lines(self, chunk_id: int, start_line: int, end_line: int) -> None:
        """Cheap metadata-only fix for an unchanged symbol whose line numbers
        shifted because earlier code grew/shrank — no re-embedding needed."""
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET start_line=?, end_line=? WHERE chunk_id=?",
                (start_line, end_line, chunk_id),
            )
            self._conn.commit()

    def delete_chunk(self, chunk_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE chunk_id=?", (chunk_id,))
            self._conn.commit()

    def delete_chunks_for_file(self, file_path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE file_path=?", (file_path,))
            self._conn.commit()

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM repo_meta")
            self._conn.commit()

    # ── reads ───────────────────────────────────────────────────────
    def get_chunks_for_file(self, file_path: str) -> List[ChunkMeta]:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE file_path=? ORDER BY start_line", (file_path,)
        )
        return [_row_to_meta(r) for r in cur.fetchall()]

    def get_chunk_ids_for_file(self, file_path: str) -> List[int]:
        cur = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE file_path=?", (file_path,)
        )
        return [r["chunk_id"] for r in cur.fetchall()]

    def get_chunk_id(self, file_path: str, symbol_name: str) -> Optional[int]:
        cur = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE file_path=? AND symbol_name=?",
            (file_path, symbol_name),
        )
        row = cur.fetchone()
        return row["chunk_id"] if row else None

    def id_exists(self, chunk_id: int) -> bool:
        cur = self._conn.execute("SELECT 1 FROM chunks WHERE chunk_id=?", (chunk_id,))
        return cur.fetchone() is not None

    def get_by_ids(self, ids: List[int]) -> List[ChunkMeta]:
        """Fetch metas for ids, preserving the input order (search ranking)."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
        )
        by_id: Dict[int, ChunkMeta] = {r["chunk_id"]: _row_to_meta(r) for r in cur.fetchall()}
        return [by_id[i] for i in ids if i in by_id]

    def all_chunk_ids(self) -> List[int]:
        cur = self._conn.execute("SELECT chunk_id FROM chunks")
        return [r["chunk_id"] for r in cur.fetchall()]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]

    def file_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT file_path) AS c FROM chunks"
        ).fetchone()
        return row["c"]

    def all_files(self) -> List[str]:
        cur = self._conn.execute("SELECT DISTINCT file_path FROM chunks ORDER BY file_path")
        return [r["file_path"] for r in cur.fetchall()]

    # ── repo-level metadata ─────────────────────────────────────────
    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO repo_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    def all_meta(self) -> Dict[str, str]:
        cur = self._conn.execute("SELECT key, value FROM repo_meta")
        return {r["key"]: r["value"] for r in cur.fetchall()}

    def close(self) -> None:
        self._conn.close()

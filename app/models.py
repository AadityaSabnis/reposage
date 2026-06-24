"""Shared data structures and small helpers used across the pipeline.

The `Chunk` dataclass is the single currency that flows from chunking
-> embedding -> vector/metadata stores -> retrieval. Keeping it in one
place keeps the field names consistent everywhere.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

# FAISS IDs are int64; we keep chunk_ids inside the positive 63-bit range.
INT63_MAX = 0x7FFFFFFFFFFFFFFF


def content_hash(text: str) -> str:
    """sha256 of the chunk text.

    Used to answer "did this chunk actually change?" during incremental
    re-indexing — a file being *touched* is not the same as a symbol's
    body actually changing.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_chunk_id(file_path: str, symbol_name: str) -> int:
    """Deterministic, stable integer id from (file_path, symbol_name).

    Stable across runs so an unchanged symbol keeps the same FAISS id and
    SQLite row between incremental updates. Collisions are resolved by the
    indexer via linear probing + a SQLite uniqueness check.
    """
    digest = hashlib.sha256(f"{file_path}::{symbol_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & INT63_MAX


@dataclass
class Chunk:
    """One retrievable unit of code (a function, class, method, or window)."""

    file_path: str          # POSIX path relative to the repo root
    start_line: int         # 1-based, inclusive
    end_line: int           # 1-based, inclusive
    symbol_name: str        # unique within its file (qualified, e.g. "Foo.bar")
    symbol_type: str        # function | class | method | module | window
    language: str           # python | javascript | typescript | text | ...
    text: str               # the chunk's source text (not persisted to SQLite)
    content_hash: str = ""  # sha256(text); filled in __post_init__ if absent
    chunk_id: Optional[int] = None  # stable FAISS id, assigned by the indexer

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.text)

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class ChunkMeta:
    """The persisted view of a chunk (everything except the raw text).

    Matches the SQLite schema one-to-one. `embedding_id` equals `chunk_id`
    because we register vectors in FAISS under their chunk_id via
    IndexIDMap; the column is kept for schema fidelity / future flexibility.
    """

    chunk_id: int
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    symbol_type: str
    language: str
    content_hash: str
    embedding_id: int

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

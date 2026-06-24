"""FAISS vector store wrapper.

Uses `faiss.IndexIDMap(faiss.IndexFlatIP(dim))`. The IndexIDMap wrapper is
the whole point: it lets us address vectors by our own stable chunk_id and
`remove_ids()` individual chunks during incremental updates — without it,
removing a single chunk would mean rebuilding the entire flat index.

Vectors are expected pre-normalized (see embedder), so inner product is
cosine similarity.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._faiss = self._import_faiss()
        self.index = self._new_index()

    @staticmethod
    def _import_faiss():
        import faiss
        return faiss

    def _new_index(self):
        # IndexFlatIP = exact cosine similarity on normalized vectors.
        # IndexIDMap = address/remove vectors by arbitrary int64 id.
        return self._faiss.IndexIDMap(self._faiss.IndexFlatIP(self.dim))

    # ── mutation ────────────────────────────────────────────────────
    def add(self, ids: Sequence[int], embeddings: np.ndarray) -> None:
        if len(ids) == 0:
            return
        ids_arr = np.asarray(ids, dtype="int64")
        embs = np.ascontiguousarray(embeddings, dtype="float32")
        if embs.shape[0] != ids_arr.shape[0]:
            raise ValueError("ids/embeddings length mismatch")
        self.index.add_with_ids(embs, ids_arr)

    def remove_ids(self, ids: Sequence[int]) -> int:
        """Remove vectors by id. Returns count actually removed (0 if absent)."""
        if len(ids) == 0:
            return 0
        sel = np.asarray(ids, dtype="int64")
        return int(self.index.remove_ids(sel))

    def upsert(self, chunk_id: int, embedding: np.ndarray) -> None:
        """Replace a single vector: remove the old id (no-op if absent), add new.

        Keeps `ntotal` exactly correct — no duplicate/orphaned vectors when a
        symbol's body changes but its id (file_path+symbol_name) stays the same.
        """
        self.remove_ids([chunk_id])
        vec = np.ascontiguousarray(embedding, dtype="float32").reshape(1, -1)
        self.index.add_with_ids(vec, np.asarray([chunk_id], dtype="int64"))

    def reset(self) -> None:
        self.index = self._new_index()

    # ── query ───────────────────────────────────────────────────────
    def search(self, query: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        """Return (scores, ids) for the top-k nearest chunks.

        FAISS pads with id=-1 when fewer than k vectors exist; we strip those.
        """
        if self.ntotal == 0:
            return [], []
        q = np.ascontiguousarray(query, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(q, min(k, self.ntotal))
        out_scores: List[float] = []
        out_ids: List[int] = []
        for score, cid in zip(scores[0], ids[0]):
            if cid == -1:
                continue
            out_scores.append(float(score))
            out_ids.append(int(cid))
        return out_scores, out_ids

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        self._faiss.write_index(self.index, str(path))

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        self.index = self._faiss.read_index(str(p))
        self.dim = self.index.d
        return True

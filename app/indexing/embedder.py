"""Embedding model wrapper (sentence-transformers).

Exposes a tiny `Embedder` protocol so the indexer/retriever depend on an
interface, not on sentence-transformers directly. That lets tests inject
a fast deterministic fake (no model download, no torch) while production
uses `all-MiniLM-L6-v2`.

Vectors are L2-normalized so FAISS inner-product == cosine similarity.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np

from app.config import settings


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return a float32 (N, dim) matrix of L2-normalized embeddings."""
        ...


class SentenceTransformerEmbedder:
    """Production embedder. The model is loaded lazily on first use so app
    startup (and importing this module) stays cheap — mirrors PRAVA's
    `get_embedder()` pattern."""

    def __init__(self, model_path: str | None = None, dim: int | None = None):
        self._model_path = model_path or settings.embedding_model_path
        self.dim = dim or settings.embedding_dim
        self._model = None
        # First-run state, surfaced via /model/status so the UI can show a
        # "downloading model…" banner instead of looking hung.
        self.status = "idle"          # idle | loading | ready | error
        self.error: str | None = None

    def _ensure_model(self):
        if self._model is None:
            self.status = "loading"
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_path)
                self.dim = self._model.get_sentence_embedding_dimension()
                self.status = "ready"
                self.error = None
            except Exception as e:
                self.status = "error"
                self.error = str(e)
                raise
        return self._model

    def warmup(self) -> None:
        """Load the model now (used at startup to surface the first-run
        download as a visible step rather than blocking the first query)."""
        self._ensure_model()

    @property
    def model_name(self) -> str:
        return self._model_path

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        model = self._ensure_model()
        embs = model.encode(
            texts,
            normalize_embeddings=True,   # cosine sim via inner product
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype="float32")


def default_embedder() -> Embedder:
    return SentenceTransformerEmbedder()

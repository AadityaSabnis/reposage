"""Shared test fixtures.

`FakeEmbedder` produces deterministic, L2-normalized 384-dim vectors from
text without downloading a model or importing torch. Identical text always
maps to the identical vector, which is all the incremental-index test needs
(it asserts on *counts* — how many chunks were embedded/removed — not on
semantic ranking quality). The real model is exercised by eval/eval_qa.
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pytest


class FakeEmbedder:
    dim = 384

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype("float32")
            norm = float(np.linalg.norm(v))
            out[i] = v / norm if norm else v
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()

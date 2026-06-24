"""Runtime singletons (dependency container).

One embedder / vector store / metadata store is shared by the indexer and
the retriever so they operate on the same in-memory FAISS index and DB.
Built lazily so importing the app is cheap and tests can construct their
own wiring (e.g. with a fake embedder).
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.indexing.embedder import Embedder, default_embedder
from app.indexing.indexer import Indexer
from app.indexing.metadata_store import MetadataStore
from app.indexing.vector_store import VectorStore
from app.llm.base import LLMClient
from app.llm.hosted_client import make_llm_client
from app.retrieval.retriever import Retriever


@lru_cache
def get_embedder() -> Embedder:
    return default_embedder()


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(settings.embedding_dim)


@lru_cache
def get_metadata_store() -> MetadataStore:
    return MetadataStore(settings.db_path)


@lru_cache
def get_indexer() -> Indexer:
    return Indexer(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        metadata_store=get_metadata_store(),
        repo_path=settings.repo_path,
    )


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        metadata_store=get_metadata_store(),
        repo_path=settings.repo_path,
    )


@lru_cache
def get_llm() -> LLMClient:
    return make_llm_client()

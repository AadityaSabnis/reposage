"""Env-driven settings for RepoSage.

A single `settings` object is imported across the app. Reads from the
process environment (and a local `.env` if present). Mirrors the
config style of the PRAVA project: plain, explicit, no magic.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Repo under analysis ──────────────────────────────────────
    repo_path: str = "./"

    # ── Persistence ──────────────────────────────────────────────
    data_dir: str = "./data"

    # ── Embeddings ───────────────────────────────────────────────
    embedding_model_path: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768  # bge-base-en-v1.5 output dimensionality

    # ── LLM provider selection ───────────────────────────────────
    llm_provider: str = "ollama"  # "ollama" | "hosted"

    # ollama
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "phi3:mini"
    ollama_keep_alive: str = "30m"

    # hosted (OpenAI-compatible: Groq / OpenAI / etc.)
    hosted_base_url: str = "https://api.groq.com/openai/v1"
    hosted_model: str = "llama-3.1-8b-instant"
    groq_api_key: str = ""
    openai_api_key: str = ""

    # ── GitHub citation links (optional overrides) ───────────────
    github_owner: str = ""
    github_repo: str = ""
    github_commit: str = ""

    # ── Retrieval ────────────────────────────────────────────────
    top_k: int = 8

    # ── Git URL indexing (clone-then-index feature) ──────────────
    git_clone_timeout: int = 120  # seconds before a clone is aborted

    # ── Derived paths ────────────────────────────────────────────
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def faiss_path(self) -> Path:
        return self.data_path / "faiss.index"

    @property
    def db_path(self) -> Path:
        return self.data_path / "metadata.sqlite"

    @property
    def repos_cache_dir(self) -> Path:
        """Where repos cloned from a Git URL are checked out for indexing."""
        p = self.data_path / "repos"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def hosted_api_key(self) -> str:
        """Whichever hosted key is set (Groq preferred, then OpenAI)."""
        return self.groq_api_key or self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

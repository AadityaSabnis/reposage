"""File-type registry + repo walker + chunk dispatch.

Responsibilities:
  * map a file extension -> (language, parser strategy)
  * decide which files to index (skip vcs/build/binary noise)
  * dispatch each file to the tree-sitter chunker, falling back to the
    line-window chunker when there's no AST parser or AST yields nothing
  * guarantee `symbol_name` is unique within a file (so it can key the
    incremental-update diff and the stable chunk_id)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List

from app.chunking import fallback_chunker, treesitter_chunker
from app.models import Chunk

# Extension -> language understood by the tree-sitter chunker.
EXT_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

# Text files we still want to index, but via the fallback chunker.
FALLBACK_TEXT_EXTS = {
    ".md", ".markdown", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".scss", ".sh", ".sql", ".env",
    ".java", ".go", ".rb", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
}

# Directories we never descend into.
IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    ".venv", "venv", "env", "__pycache__",
    "dist", "build", "out", ".next", ".nuxt",
    "target", ".gradle", ".idea", ".vscode",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "coverage", ".cache", "vendor", "data",
}

# Files we skip by name.
IGNORE_FILES = {".DS_Store", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"}

MAX_FILE_BYTES = 1_000_000  # skip very large files (likely generated/minified)


def language_for(path: str) -> str | None:
    """Return the AST language for a path, or None if not AST-supported."""
    return EXT_TO_LANGUAGE.get(Path(path).suffix.lower())


def _is_binary(path: Path) -> bool:
    """Cheap binary sniff: NUL byte in the first 1KB."""
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return True


def should_index(path: Path) -> bool:
    ext = path.suffix.lower()
    if path.name in IGNORE_FILES:
        return False
    if ext not in EXT_TO_LANGUAGE and ext not in FALLBACK_TEXT_EXTS:
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return not _is_binary(path)


def iter_source_files(repo_root: str | Path) -> Iterator[Path]:
    """Yield indexable files under repo_root, skipping ignored dirs/files."""
    root = Path(repo_root).resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored dirs in-place so os.walk doesn't descend them
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            p = Path(dirpath) / fname
            if should_index(p):
                yield p


def _uniquify(chunks: List[Chunk]) -> List[Chunk]:
    """Ensure symbol_name is unique within the file (stable suffixing)."""
    seen: Dict[str, int] = {}
    for c in chunks:
        if c.symbol_name in seen:
            seen[c.symbol_name] += 1
            c.symbol_name = f"{c.symbol_name}#{seen[c.symbol_name]}"
        else:
            seen[c.symbol_name] = 0
    return chunks


def chunk_file(repo_root: str | Path, file_path: str | Path) -> List[Chunk]:
    """Chunk a single file. `file_path` may be absolute or repo-relative.

    Returned chunks carry a POSIX `file_path` relative to repo_root.
    """
    root = Path(repo_root).resolve()
    abs_path = Path(file_path)
    if not abs_path.is_absolute():
        abs_path = (root / abs_path).resolve()

    try:
        rel = abs_path.relative_to(root).as_posix()
    except ValueError:
        rel = abs_path.name

    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    language = language_for(str(abs_path))
    chunks: List[Chunk] = []
    if language is not None:
        chunks = treesitter_chunker.chunk(text, rel, language)

    # No AST parser, or AST produced nothing -> line-window fallback.
    if not chunks:
        fallback_lang = language or _fallback_language(abs_path)
        chunks = fallback_chunker.chunk(text, rel, fallback_lang)

    return _uniquify(chunks)


def _fallback_language(path: Path) -> str:
    ext = path.suffix.lower()
    return ext.lstrip(".") or "text"

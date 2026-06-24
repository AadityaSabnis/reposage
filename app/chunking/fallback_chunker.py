"""Fixed-size line-window chunker.

Used for file types we have no AST parser for (Markdown, JSON, config,
plain text, or any source language not yet in the tree-sitter registry),
and as a safety net when AST parsing yields no symbols.

~40-line windows with 5-line overlap so a symbol/section straddling a
window boundary still lands wholly inside at least one chunk.
"""
from __future__ import annotations

from typing import List

from app.models import Chunk

WINDOW_LINES = 40
OVERLAP_LINES = 5


def chunk(text: str, file_path: str, language: str = "text") -> List[Chunk]:
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    chunks: List[Chunk] = []
    start = 0
    window_idx = 0
    step = max(1, WINDOW_LINES - OVERLAP_LINES)

    while start < n:
        end = min(start + WINDOW_LINES, n)
        body = "\n".join(lines[start:end])
        if body.strip():
            chunks.append(
                Chunk(
                    file_path=file_path,
                    start_line=start + 1,        # 1-based inclusive
                    end_line=end,                # 1-based inclusive
                    symbol_name=f"window_{window_idx}",
                    symbol_type="window",
                    language=language,
                    text=body,
                )
            )
            window_idx += 1
        if end == n:
            break
        start += step

    return chunks

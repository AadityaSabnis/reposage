"""Phase 1 — chunking correctness.

Verifies AST-aware chunks have the right symbols, types, and (crucially)
line ranges that line up with the actual source, plus the fallback path.
"""
from __future__ import annotations

import pytest

from app.chunking import fallback_chunker
from app.chunking.registry import chunk_file

ts = pytest.importorskip("tree_sitter_languages")  # skip if grammars unavailable


PY_SOURCE = '''\
import os


def top_level(x):
    """A module-level function."""
    return x + 1


class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"hi {self.name}"
'''


def _by_name(chunks):
    return {c.symbol_name: c for c in chunks}


def test_python_symbols_and_types(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text(PY_SOURCE)

    chunks = chunk_file(tmp_path, f)
    by = _by_name(chunks)

    assert set(by) == {"top_level", "Greeter", "Greeter.__init__", "Greeter.greet"}
    assert by["top_level"].symbol_type == "function"
    assert by["Greeter"].symbol_type == "class"
    assert by["Greeter.greet"].symbol_type == "method"
    assert all(c.language == "python" for c in chunks)


def test_python_line_ranges_match_source(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text(PY_SOURCE)
    src_lines = PY_SOURCE.splitlines()

    for c in chunk_file(tmp_path, f):
        # 1-based inclusive ranges should slice back to the chunk text.
        sliced = "\n".join(src_lines[c.start_line - 1:c.end_line])
        assert sliced == c.text, f"line range wrong for {c.symbol_name}"

    by = _by_name(chunk_file(tmp_path, f))
    assert src_lines[by["top_level"].start_line - 1].startswith("def top_level")
    assert src_lines[by["Greeter.greet"].start_line - 1].strip().startswith("def greet")


def test_content_hash_is_stable_and_set(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text(PY_SOURCE)
    c = _by_name(chunk_file(tmp_path, f))["top_level"]
    assert c.content_hash and len(c.content_hash) == 64  # sha256 hex


def test_relative_posix_path(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    f = sub / "mod.py"
    f.write_text("def f():\n    return 1\n")
    c = chunk_file(tmp_path, f)[0]
    assert c.file_path == "pkg/mod.py"


def test_fallback_for_unsupported_extension(tmp_path):
    f = tmp_path / "notes.txt"
    body = "\n".join(f"line {i}" for i in range(100))
    f.write_text(body)

    chunks = chunk_file(tmp_path, f)
    assert len(chunks) >= 2
    assert all(c.symbol_type == "window" for c in chunks)
    assert chunks[0].start_line == 1


def test_fallback_window_overlap():
    body = "\n".join(f"line {i}" for i in range(100))
    chunks = fallback_chunker.chunk(body, "notes.txt", "text")
    # 40-line windows, 5-line overlap -> second window starts at line 36
    assert chunks[0].start_line == 1 and chunks[0].end_line == 40
    assert chunks[1].start_line == 36


JS_SOURCE = '''\
export function add(a, b) {
  return a + b;
}

const mul = (a, b) => a * b;

class Calc {
  square(x) {
    return x * x;
  }
}
'''


def test_javascript_symbols(tmp_path):
    pytest.importorskip("tree_sitter_languages")
    f = tmp_path / "calc.js"
    f.write_text(JS_SOURCE)
    by = _by_name(chunk_file(tmp_path, f))
    assert "add" in by and by["add"].symbol_type == "function"
    assert "mul" in by  # arrow function assigned to const
    assert "Calc" in by and by["Calc"].symbol_type == "class"
    assert "Calc.square" in by and by["Calc.square"].symbol_type == "method"

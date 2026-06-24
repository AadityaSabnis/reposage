"""Phase 4 — the core systems flex.

These tests are the proof that indexing is *incremental*, not a relabeled
full rebuild:

  * edit one function in a 5-function file -> exactly ONE chunk re-embedded
  * FAISS `ntotal` stays correct (no duplicate / orphaned vectors)
  * a removed symbol drops its (one) vector; a renamed symbol swaps 1<->1
  * shifting line numbers without changing content => zero re-embeds
  * deleting a file removes exactly its chunks
"""
from __future__ import annotations

import pytest

pytest.importorskip("faiss")
pytest.importorskip("tree_sitter_languages")

from app.indexing.indexer import Indexer
from app.indexing.metadata_store import MetadataStore
from app.indexing.vector_store import VectorStore

FIVE_FUNCS = '''\
def alpha():
    return 1


def beta():
    return 2


def gamma():
    return 3


def delta():
    return 4


def epsilon():
    return 5
'''


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A temp repo with one 5-function file, plus an indexer wired to a fake
    embedder and temp-dir persistence (so we never touch ./data)."""
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "mod.py").write_text(FIVE_FUNCS)
    return repo_dir


@pytest.fixture
def indexer(repo, tmp_path, fake_embedder):
    vs = VectorStore(fake_embedder.dim)
    ms = MetadataStore(tmp_path / "meta.sqlite")
    return Indexer(fake_embedder, vs, ms, repo_path=str(repo))


def test_full_index_baseline(indexer):
    stats = indexer.full_index()
    assert stats["chunks_indexed"] == 5
    assert indexer.vector_store.ntotal == 5
    assert indexer.metadata.count() == 5


def test_edit_one_function_reembeds_exactly_one(indexer, repo):
    indexer.full_index()
    assert indexer.vector_store.ntotal == 5

    # Change ONLY gamma's body, keeping the same number of lines so no other
    # function's line range shifts.
    src = (repo / "mod.py").read_text().replace("return 3", "return 33")
    (repo / "mod.py").write_text(src)

    stats = indexer.incremental_update(changed_files=["mod.py"], deleted_files=[])

    assert stats["embedded"] == 1, stats          # <-- the whole point
    assert stats["reembed_skipped"] == 4, stats    # the other four untouched
    assert stats["removed"] == 0, stats
    # No duplicate / orphaned vectors: still exactly 5.
    assert indexer.vector_store.ntotal == 5
    assert indexer.metadata.count() == 5


def test_line_shift_without_content_change_is_free(indexer, repo):
    indexer.full_index()

    # Insert a line inside alpha: alpha's content changes (1 re-embed); the
    # other four are byte-identical but shifted down -> metadata-only update.
    src = (repo / "mod.py").read_text().replace(
        "def alpha():\n    return 1", "def alpha():\n    x = 0\n    return 1"
    )
    (repo / "mod.py").write_text(src)

    stats = indexer.incremental_update(changed_files=["mod.py"], deleted_files=[])
    assert stats["embedded"] == 1, stats
    assert stats["reembed_skipped"] == 4, stats
    assert stats["lines_updated"] == 4, stats
    assert indexer.vector_store.ntotal == 5

    # beta's stored line range must reflect the +1 shift.
    beta = next(m for m in indexer.metadata.get_chunks_for_file("mod.py")
                if m.symbol_name == "beta")
    assert beta.start_line == 6  # was 5, shifted by the inserted line


def test_renamed_symbol_swaps_one_for_one(indexer, repo):
    indexer.full_index()
    src = (repo / "mod.py").read_text().replace("def epsilon():", "def zeta():")
    (repo / "mod.py").write_text(src)

    stats = indexer.incremental_update(changed_files=["mod.py"], deleted_files=[])
    assert stats["embedded"] == 1, stats   # zeta is new
    assert stats["removed"] == 1, stats     # epsilon is gone
    assert indexer.vector_store.ntotal == 5
    names = {m.symbol_name for m in indexer.metadata.get_chunks_for_file("mod.py")}
    assert "zeta" in names and "epsilon" not in names


def test_deleting_file_removes_only_its_chunks(indexer, repo):
    # Add a second file so we can prove the first file's deletion is scoped.
    (repo / "other.py").write_text("def keep():\n    return 0\n")
    indexer.full_index()
    assert indexer.vector_store.ntotal == 6  # 5 + 1

    stats = indexer.incremental_update(changed_files=[], deleted_files=["mod.py"])
    assert stats["removed"] == 5, stats
    assert indexer.vector_store.ntotal == 1
    assert indexer.metadata.all_files() == ["other.py"]


def test_idempotent_update_reembeds_nothing(indexer):
    indexer.full_index()
    stats = indexer.incremental_update(changed_files=["mod.py"], deleted_files=[])
    assert stats["embedded"] == 0, stats
    assert stats["reembed_skipped"] == 5, stats
    assert indexer.vector_store.ntotal == 5

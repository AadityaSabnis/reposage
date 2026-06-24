"""Phase 6 — retrieval evaluation harness.

Indexes RepoSage's own codebase with the real embedding model, runs every
question in eval/qa_pairs.json through the retriever, and reports
hit_rate@k: did the expected file+symbol appear among the top-k retrieved
citations? (Retrieval is what /ask cites, so this measures the part that
makes answers trustworthy — no LLM call needed.)

Run:
    python tests/eval_qa.py            # hit_rate@5 over reposage itself
    python tests/eval_qa.py --k 8 --repo /path/to/repo --qa eval/qa_pairs.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Allow `python tests/eval_qa.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.embedder import default_embedder          # noqa: E402
from app.indexing.indexer import Indexer                    # noqa: E402
from app.indexing.metadata_store import MetadataStore       # noqa: E402
from app.indexing.vector_store import VectorStore           # noqa: E402
from app.retrieval.retriever import Retriever               # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QA = REPO_ROOT / "eval" / "qa_pairs.json"


def _hit(citations, expected_file: str, expected_symbol: str) -> bool:
    ef = expected_file.lower()
    es = expected_symbol.lower()
    for c in citations:
        file_ok = ef in c.file_path.lower()
        symbol_ok = es in c.symbol_name.lower()
        if file_ok and symbol_ok:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--qa", default=str(DEFAULT_QA))
    args = ap.parse_args()

    qa_pairs = json.loads(Path(args.qa).read_text())

    print(f"Indexing {args.repo} with the real embedding model "
          f"(first run downloads all-MiniLM-L6-v2)…")
    with tempfile.TemporaryDirectory() as tmp:
        embedder = default_embedder()
        vs = VectorStore(getattr(embedder, "dim", 384))
        ms = MetadataStore(Path(tmp) / "eval.sqlite")
        indexer = Indexer(embedder, vs, ms, repo_path=args.repo)
        stats = indexer.full_index()
        print(f"Indexed {stats['chunks_indexed']} chunks across "
              f"{stats['files_indexed']} files in {stats['elapsed_sec']}s.\n")

        retriever = Retriever(embedder, vs, ms, repo_path=args.repo)

        hits = 0
        rows = []
        for qa in qa_pairs:
            citations = retriever.retrieve(qa["question"], top_k=args.k)
            ok = _hit(citations, qa["expected_file"], qa["expected_symbol"])
            hits += int(ok)
            top = citations[0].symbol_name if citations else "—"
            rows.append((ok, qa["expected_symbol"], top, qa["question"]))

    total = len(qa_pairs)
    rate = hits / total if total else 0.0

    print(f"{'HIT?':<5} {'EXPECTED SYMBOL':<32} {'TOP-1 RETRIEVED':<32} QUESTION")
    print("-" * 110)
    for ok, expected, top, q in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"{mark:<5} {expected:<32} {top:<32} {q[:42]}")
    print("-" * 110)
    print(f"hit_rate@{args.k} = {hits}/{total} = {rate:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

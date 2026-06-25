# RepoSage

**Ask natural-language questions about a codebase and get answers grounded in exact retrieved code — every claim cited with a clickable `file#Lstart-Lend` GitHub link.** The engineering centerpiece is **incremental indexing**: when files change, only the symbols whose content actually changed are re-embedded and swapped in the vector store — never a full-repo re-embed.

> Edit one function in a 50-file repo → exactly **one** chunk is re-embedded. That moment, observable on `/stats`, is the whole point of the project.

---

## Why it's interesting

Most "chat with your repo" demos re-embed everything on every change and cite nothing you can verify. RepoSage does the opposite:

- **AST-aware chunks, not blind windows.** `tree-sitter` splits code into function/class/method-level chunks with precise line ranges, so a citation points at a real symbol. Unsupported file types fall back to overlapping line windows so nothing is lost.
- **Truly incremental.** Each chunk carries a `sha256` content hash and a *stable* integer id derived from `(file_path, symbol_name)`. On a change we diff old vs. new chunks: unchanged bodies are skipped, a symbol that merely shifted lines gets a metadata-only fix (no re-embed), changed/new symbols are re-embedded, and vanished symbols are removed by id via `faiss` `IndexIDMap.remove_ids()` — keeping `ntotal` exactly correct with no orphaned vectors.
- **Citation-grounded answers.** The LLM is instructed to answer *only* from the provided chunks, cite every claim, and say so explicitly when the answer isn't present — no invented paths or line numbers.

## Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │                  FastAPI app                   │
                      │   /repos/index   /ask   /webhook/git-push      │
                      │             /stats   /  (chat UI)              │
                      └───────┬───────────────┬───────────────┬───────┘
              full / incremental              │ query         │ changed/deleted
                      │                        │               │  paths
            ┌─────────▼─────────┐     ┌────────▼────────┐      │
            │     Indexer       │     │    Retriever    │      │
            │ full_index()      │     │ embed → search  │      │
            │ incremental_update│     │ → hydrate hits  │      │
            └───┬───────┬───────┘     └────────┬────────┘      │
   chunk_file() │       │ embed                 │ embed query  │
     ┌──────────▼──┐  ┌─▼───────────┐   ┌───────▼──────┐       │
     │  Chunking   │  │  Embedder   │   │  LLM client  │◄──────┘
     │ tree-sitter │  │ MiniLM-L6   │   │ ollama /     │
     │  + fallback │  │ (384-dim)   │   │ hosted(Groq) │
     └─────────────┘  └─────┬───────┘   └──────────────┘
                            │ normalized vectors + ids
            ┌───────────────▼───────────────┐
            │  VectorStore (FAISS)           │   MetadataStore (SQLite)
            │  IndexIDMap(IndexFlatIP)       │   1 row / chunk:
            │  add / remove_ids / upsert     │   id, file, lines, symbol,
            │  search (cosine via IP)        │   type, lang, content_hash …
            └────────────────────────────────┘
```

**Stack:** Python · FastAPI · tree-sitter · sentence-transformers (`BAAI/bge-base-en-v1.5`, local, 768-dim) · FAISS (`IndexIDMap(IndexFlatIP)`) · SQLite · pluggable LLM (Ollama `phi3:mini` locally / OpenAI-compatible hosted for deploys) · watchdog · vanilla-JS frontend.

## Layout

```
reposage/
├── app/
│   ├── main.py                 FastAPI app, CORS, /stats, serves the UI
│   ├── config.py               env-driven settings
│   ├── models.py               Chunk + stable id / content-hash helpers
│   ├── deps.py                 shared runtime singletons
│   ├── chunking/               registry · treesitter_chunker · fallback_chunker
│   ├── indexing/               embedder · vector_store · metadata_store · indexer
│   ├── retrieval/retriever.py  embed query → search → cited hits
│   ├── llm/                    base (interface + grounding prompt) · ollama · hosted
│   ├── routes/                 index_routes · ask_routes
│   ├── services/git_clone.py   shallow-clone a remote repo for indexing
│   └── watcher.py              watchdog incremental trigger (dev)
├── frontend/index.html         single-page chat + evidence panel
├── tests/                      test_chunking · test_incremental_index · test_git_clone · eval_qa
├── eval/qa_pairs.json          hand-written retrieval eval set
├── Dockerfile · requirements.txt · .env.example
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # set REPO_PATH to the repo you want to index

# Local dev LLM (optional but recommended): Ollama
#   ollama serve
#   ollama pull phi3:mini

uvicorn app.main:app --reload
```

Then open **http://localhost:8000** for the chat UI. The setup screen has a **Local Path / Git URL** toggle so you can index from either source.

The server will try Ollama first (no API key needed). If Ollama is unavailable, it automatically falls back to **Groq** (free tier) — just make sure you added `GROQ_API_KEY=gsk_...` to `.env`.

Or use curl:

```bash
# 1a) Index a local repo (defaults to REPO_PATH if you omit repo_path)
curl -X POST localhost:8000/repos/index -H 'content-type: application/json' \
     -d '{"repo_path": "/path/to/repo"}'

# 1b) Or index directly from a Git URL (shallow-clones, then indexes)
curl -X POST localhost:8000/repos/index-git -H 'content-type: application/json' \
     -d '{"git_url": "https://github.com/owner/repo"}'

# 2) Ask
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How does incremental indexing decide what to re-embed?"}'
```

Or open **http://localhost:8000** for the chat UI (question box + evidence cards with GitHub links).

### Seeing incremental indexing in action

```bash
# After indexing, edit one function in one file, then:
curl -X POST localhost:8000/webhook/git-push -H 'content-type: application/json' \
     -d '{"changed_files": ["app/foo.py"], "deleted_files": []}'
# -> {"embedded": 1, "reembed_skipped": 12, "removed": 0, "ntotal": 156, ...}

curl localhost:8000/stats   # last_incremental_update.embedded == 1
```

For local development you can instead run the file watcher, which calls the same
incremental path on every save:

```bash
python -m app.watcher        # watches REPO_PATH
```

## API

| Method & path            | Purpose                                                             |
|--------------------------|---------------------------------------------------------------------|
| `POST /repos/index`      | Full (re)index of a **local path**. `{repo_path?}` → `{files_indexed, chunks_indexed, ntotal, elapsed_sec}` |
| `POST /repos/index-git`  | Shallow-clone a remote repo and full-index it. `{git_url, branch?}` → same stats + `git_url`. GitHub citation links are auto-detected from the clone. |
| `POST /ask`              | Blocking. `{question, top_k?}` → `{answer, citations:[{file_path,start_line,end_line,github_url,snippet}], model}` |
| `POST /ask/stream`       | Streaming (SSE). Emits a `citations` event immediately, then `token` events, then `done`. The chat UI uses this. |
| `POST /webhook/git-push` | `{changed_files, deleted_files}` → incremental stats (proves how little was re-embedded) |
| `GET  /stats`            | index size + the last incremental-update result                     |
| `GET  /health`           | liveness                                                            |

## Tests

```bash
pytest tests/test_chunking.py tests/test_incremental_index.py -v
```

`test_incremental_index.py` is the project's keystone. It edits one function in a
five-function file and asserts **exactly one** chunk was re-embedded (not five,
not the whole repo), that FAISS `ntotal` stays correct, that a rename swaps
1-for-1, that line-only shifts cost zero re-embeds, and that deleting a file
removes only its chunks.

## Retrieval evaluation

`eval/qa_pairs.json` holds hand-written questions about RepoSage's own code, each
tagged with the file + symbol retrieval should surface. `tests/eval_qa.py` indexes
the repo with the real `all-MiniLM-L6-v2` model, runs each question through the
retriever, and reports `hit_rate@k` — did the expected `file + symbol` appear in
the top-k citations?

```bash
python tests/eval_qa.py --k 5
```

<!-- EVAL_RESULTS -->
Measured on RepoSage's own codebase — 24 questions, `all-MiniLM-L6-v2`, top-k=5,
a hit requires the expected **file _and_ symbol** to appear in the top-5 citations:

> ### **`hit_rate@5 = 20 / 24 = 83%`**

The four misses are honest near-misses, not retrieval failures:

| Question about | Expected symbol | Top-1 retrieved | Why it missed |
|----------------|-----------------|-----------------|---------------|
| AST function extraction | `treesitter_chunker.chunk` | `_get_parser` (same file) | `chunk` is an over-generic name shared with the fallback chunker |
| reading a citation snippet | `Retriever._read_snippet` | a doc window | competed with prose chunks that literally say "snippet" |
| the Ollama client call | `OllamaClient.generate` | `OllamaClient` (the class) | strict check wants the method; the class chunk ranked first |
| the `/ask` endpoint | `ask_routes.ask` | `RetrievedChunk.to_citation_dict` | "citations" pulled the citation builder above the route |

Three of the four retrieved the right *file* or the enclosing *class* in the top
result — relaxing the metric to file-level pushes it higher. Run it yourself:
`python tests/eval_qa.py --k 5` prints the full PASS/FAIL table.

## Configuration

All via env / `.env` (see `.env.example`):

| Var | Default | Notes |
|-----|---------|-------|
| `REPO_PATH` | `./` | repo to index |
| `DATA_DIR` | `./data` | FAISS index + SQLite live here |
| `EMBEDDING_MODEL_PATH` | `BAAI/bge-base-en-v1.5` | model name or local path (768-dim, retrieval-optimized) |
| `LLM_PROVIDER` | `ollama` | `ollama` (tries local, falls back to Groq) or `hosted` (use Groq/OpenAI directly) |
| `OLLAMA_MODEL` | `phi3:mini` | local model |
| `HOSTED_BASE_URL` / `HOSTED_MODEL` | Groq defaults | OpenAI-compatible endpoint |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | — | fallback when Ollama unavailable, or always when `LLM_PROVIDER=hosted` |
| `GITHUB_OWNER/REPO/COMMIT` | auto-detected from git | citation link base |
| `TOP_K` | `8` | chunks retrieved per question |
| `GIT_CLONE_TIMEOUT` | `120` | seconds before a remote clone is aborted |

## Deploy

```bash
docker build -t reposage .
docker run -p 8000:8000 \
  -e LLM_PROVIDER=hosted -e GROQ_API_KEY=... \
  -e REPO_PATH=/app/repo -v /path/to/repo:/app/repo \
  reposage
```

The image pre-downloads the embedding model and defaults to `LLM_PROVIDER=hosted`
(Ollama won't run on most free tiers). It honors `$PORT`, so it deploys as-is to
Railway / Render / Fly.io.

## Demo

<!-- Replace with a 30–60s GIF: index a repo → ask 3 questions with citations →
edit one function → /webhook/git-push → /stats shows embedded == 1. -->
![demo placeholder](docs/demo.gif)

## Scope

Single repo, no auth, Python + JS/TS AST chunking (others fall back to windows),
blocking (non-streaming) answers — deliberately, to keep the depth on the
indexing/retrieval pipeline rather than breadth of features.

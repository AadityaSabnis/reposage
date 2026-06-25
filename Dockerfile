# RepoSage — citation-grounded RAG over codebases.
# Defaults to LLM_PROVIDER=hosted because Ollama won't run on most free tiers.
FROM python:3.11-slim

# Build deps for faiss / sentence-transformers wheels are not needed
# (prebuilt wheels exist), but git is handy for citation commit detection.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so cold starts are fast
# and the container needs no network for embeddings at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

ENV LLM_PROVIDER=hosted \
    DATA_DIR=/app/data \
    REPO_PATH=/app/repo \
    PORT=8000

EXPOSE 8000

# Railway/Render/Fly inject $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

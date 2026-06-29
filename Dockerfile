FROM python:3.11-slim

WORKDIR /app

# System deps for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY docs/ ./docs/

# Create data directory
RUN mkdir -p data/vector_store

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    VECTOR_STORE_PATH=/app/data/vector_store \
    DOCS_PATH=/app/docs \
    CLAUDE_MODEL=claude-sonnet-4-6 \
    MAX_TOKENS=1024 \
    TOP_K=5

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

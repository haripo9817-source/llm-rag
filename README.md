# 🔮 RAG Chatbot — Claude + FAISS

> LLM-powered Retrieval-Augmented Generation pipeline with a FastAPI backend and GitHub Pages demo frontend.

[![CI Pipeline](https://github.com/YOUR_USERNAME/rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rag-chatbot/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Architecture

```
┌─────────────┐     embed      ┌──────────────┐     retrieve    ┌──────────────┐
│  Documents  │ ────────────▶  │  FAISS Index │ ─────────────▶  │  Claude LLM  │
│  (txt/md)   │  MiniLM-L6-v2  │  (cosine sim)│  top-k chunks   │  (answer)    │
└─────────────┘                └──────────────┘                  └──────────────┘
                                                                        │
                                                             ┌──────────▼──────────┐
                                                             │  FastAPI  /chat     │
                                                             │  POST response      │
                                                             └──────────┬──────────┘
                                                                        │
                                                             ┌──────────▼──────────┐
                                                             │  GitHub Pages demo  │
                                                             │  (frontend/)        │
                                                             └─────────────────────┘
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Ingest Documents

```bash
# Ingest a directory of docs
python ingest.py --docs docs/

# Or inline text
python ingest.py --text "Your document content" --source "my-doc"
```

### 4. Run the API

```bash
uvicorn src.api.server:app --reload --port 8000
```

### 5. Query via CLI

```bash
python ingest.py --query "What is RAG and how does it work?"
```

### 6. Open the Demo

Open `frontend/index.html` in your browser, set the API URL to `http://localhost:8000`.

---

## Docker

```bash
docker compose up --build
```

The API will be at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + vector count |
| `GET` | `/stats` | Model and index stats |
| `POST` | `/ingest` | Ingest text documents |
| `POST` | `/chat` | Single-turn RAG chat |
| `POST` | `/chat/stream` | Streaming RAG chat (SSE) |

### POST /chat

```json
{
  "question": "What is retrieval-augmented generation?",
  "history": []
}
```

**Response:**
```json
{
  "answer": "RAG is a technique that...[Source 1]",
  "sources": [
    { "index": 1, "source": "rag-overview.md", "score": 0.923, "preview": "..." }
  ],
  "latency_ms": 1240.5
}
```

---

## GitHub Pages Demo

The `frontend/` directory is automatically deployed to GitHub Pages on pushes to `main`.

**Setup:**
1. Go to **Settings → Pages** → Source: **GitHub Actions**
2. Push to `main` — the CI workflow deploys `frontend/` automatically
3. Enter your backend URL in the sidebar of the demo

---

## CI Pipeline

```
push/PR → lint (ruff + black) → tests (pytest + coverage) → docker build → deploy pages
```

**Secrets required:**
- `ANTHROPIC_API_KEY` — for live API tests (optional; mock tests run without it)

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model |
| `VECTOR_STORE_PATH` | `data/vector_store` | FAISS persistence path |
| `DOCS_PATH` | `docs` | Auto-ingested on startup |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |
| `MAX_TOKENS` | `1024` | LLM max output tokens |
| `TEMPERATURE` | `0.3` | LLM temperature |

---

## Project Structure

```
rag-chatbot/
├── src/
│   ├── rag/
│   │   ├── pipeline.py      # TextSplitter, VectorStore, RAGPipeline
│   │   └── loaders.py       # File loaders (txt, md, json)
│   └── api/
│       ├── server.py         # FastAPI app + endpoints
│       └── claude_client.py  # Anthropic SDK wrapper
├── tests/
│   └── test_pipeline.py     # pytest suite
├── frontend/
│   └── index.html           # GitHub Pages demo
├── docs/
│   └── rag-overview.md      # Sample knowledge base
├── .github/workflows/
│   └── ci.yml               # Lint → Test → Docker → Pages
├── ingest.py                # CLI tool
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## License

MIT © Hari Haran Polina

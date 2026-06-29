"""
FastAPI server — RAG chatbot endpoints.
Endpoints:
  POST /ingest        - ingest documents into vector store
  POST /chat          - single-turn RAG chat
  POST /chat/stream   - streaming RAG chat (SSE)
  GET  /health        - health check
  GET  /stats         - index stats
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.claude_client import ClaudeRAGClient
from src.rag.loaders import load_directory, load_from_strings
from src.rag.pipeline import RAGPipeline

# ── Startup / Shutdown ────────────────────────────────────────────────────────

pipeline: Optional[RAGPipeline] = None
llm: Optional[ClaudeRAGClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, llm
    store_path = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    pipeline = RAGPipeline(
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
        top_k=int(os.getenv("TOP_K", "5")),
        store_path=store_path,
    )
    llm = ClaudeRAGClient(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
    )
    # Auto-ingest docs folder if it exists and index is empty
    docs_path = os.getenv("DOCS_PATH", "docs")
    if pipeline.store.index.ntotal == 0 and os.path.isdir(docs_path):
        docs = load_directory(docs_path)
        if docs:
            pipeline.ingest(docs)
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="LLM-powered retrieval-augmented generation using Claude + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────


class IngestRequest(BaseModel):
    texts: List[str]
    source: str = "api"


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "indexed_vectors": pipeline.store.index.ntotal if pipeline else 0}


@app.get("/stats")
def stats():
    return {
        "model": llm.model if llm else None,
        "indexed_vectors": pipeline.store.index.ntotal if pipeline else 0,
        "top_k": pipeline.top_k if pipeline else None,
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    docs = load_from_strings(req.texts, source=req.source)
    n = pipeline.ingest(docs)
    return {"chunks_indexed": n, "total_vectors": pipeline.store.index.ntotal}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    t0 = time.perf_counter()
    context, chunks = pipeline.query(req.question)
    answer = llm.chat(req.question, context, history=req.history)
    latency = (time.perf_counter() - t0) * 1000

    return ChatResponse(
        answer=answer,
        sources=llm.format_sources(chunks),
        latency_ms=round(latency, 1),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    context, chunks = pipeline.query(req.question)
    sources = llm.format_sources(chunks)

    def generate():
        import json

        # First emit sources as metadata event
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        # Stream answer tokens
        for token in llm.stream(req.question, context, history=req.history):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

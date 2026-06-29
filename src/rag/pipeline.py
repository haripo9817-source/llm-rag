"""
RAG Pipeline - Document ingestion, chunking, embedding, and retrieval.
Uses FAISS for vector store and sentence-transformers for embeddings.
"""

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    content: str
    metadata: Dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class RetrievedChunk:
    content: str
    metadata: Dict
    score: float
    doc_id: str


class TextSplitter:
    """Recursive character text splitter with overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # Try to break on sentence boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    idx = text.rfind(sep, start, end)
                    if idx > start:
                        end = idx + len(sep)
                        break
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        return [c for c in chunks if c]


class VectorStore:
    """FAISS-backed vector store with persistence."""

    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine with normalized vecs)
        self.chunks: List[RetrievedChunk] = []
        self.embedding_dim = embedding_dim

    def add(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        assert len(texts) == len(embeddings) == len(metadata)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for text, meta in zip(texts, metadata):
            self.chunks.append(
                RetrievedChunk(
                    content=text, metadata=meta, score=0.0, doc_id=meta.get("doc_id", "")
                )
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        if self.index.ntotal == 0:
            return []
        query_embedding = query_embedding.copy()
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self.chunks[idx]
                results.append(
                    RetrievedChunk(
                        content=chunk.content,
                        metadata=chunk.metadata,
                        score=float(score),
                        doc_id=chunk.doc_id,
                    )
                )
        return results

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        store.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      ingest(docs) → embed → store in FAISS
      query(text)  → embed → retrieve → format context
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        store_path: Optional[str] = None,
    ):
        self.embedder = SentenceTransformer(model_name)
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.top_k = top_k
        self.store_path = store_path

        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        if store_path and Path(f"{store_path}/index.faiss").exists():
            self.store = VectorStore.load(store_path)
            print(f"[RAG] Loaded existing index ({self.store.index.ntotal} vectors)")
        else:
            self.store = VectorStore(embedding_dim)

    def ingest(self, documents: List[Document]) -> int:
        """Chunk, embed, and index documents. Returns total chunks added."""
        all_texts, all_meta = [], []

        for doc in documents:
            chunks = self.splitter.split(doc.content)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_meta.append(
                    {
                        **doc.metadata,
                        "doc_id": doc.doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )

        if not all_texts:
            return 0

        embeddings = self.embedder.encode(
            all_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=False
        )
        self.store.add(all_texts, embeddings.astype(np.float32), all_meta)

        if self.store_path:
            self.store.save(self.store_path)
            print(f"[RAG] Saved index to {self.store_path}")

        print(f"[RAG] Indexed {len(all_texts)} chunks from {len(documents)} documents")
        return len(all_texts)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Embed query and retrieve top-k chunks."""
        embedding = self.embedder.encode([query], normalize_embeddings=False).astype(np.float32)
        return self.store.search(embedding, top_k=self.top_k)

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not chunks:
            return "No relevant context found."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            parts.append(f"[Source {i}: {source}]\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str) -> Tuple[str, List[RetrievedChunk]]:
        """Retrieve relevant chunks and return (context_str, chunks)."""
        chunks = self.retrieve(question)
        context = self.build_context(chunks)
        return context, chunks

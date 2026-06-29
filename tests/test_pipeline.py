"""
Test suite for RAG pipeline.
Run: pytest tests/ -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.loaders import load_from_strings
from src.rag.pipeline import Document, RAGPipeline, RetrievedChunk, TextSplitter, VectorStore

# ── TextSplitter ──────────────────────────────────────────────────────────────


class TestTextSplitter:
    def test_short_text_not_split(self):
        splitter = TextSplitter(chunk_size=200)
        text = "Hello world."
        assert splitter.split(text) == [text]

    def test_long_text_produces_multiple_chunks(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = splitter.split(text)
        if len(chunks) > 1:
            # Last chars of chunk[0] should appear in start of chunk[1]
            overlap_text = chunks[0][-20:]
            assert any(overlap_text[:5] in chunks[1] for _ in [1])

    def test_no_empty_chunks(self):
        splitter = TextSplitter(chunk_size=30, chunk_overlap=5)
        text = "\n\n".join(["Some text here."] * 20)
        chunks = splitter.split(text)
        assert all(c.strip() for c in chunks)


# ── Document ──────────────────────────────────────────────────────────────────


class TestDocument:
    def test_doc_id_auto_generated(self):
        doc = Document(content="hello")
        assert len(doc.doc_id) == 12

    def test_same_content_same_id(self):
        d1 = Document(content="same text")
        d2 = Document(content="same text")
        assert d1.doc_id == d2.doc_id

    def test_different_content_different_id(self):
        d1 = Document(content="text A")
        d2 = Document(content="text B")
        assert d1.doc_id != d2.doc_id


# ── VectorStore ───────────────────────────────────────────────────────────────


class TestVectorStore:
    def test_add_and_search(self):
        store = VectorStore(embedding_dim=4)
        vecs = np.random.rand(3, 4).astype(np.float32)
        texts = ["alpha", "beta", "gamma"]
        meta = [{"source": f"doc{i}"} for i in range(3)]
        store.add(texts, vecs, meta)
        assert store.index.ntotal == 3

        query = np.random.rand(1, 4).astype(np.float32)
        results = store.search(query, top_k=2)
        assert len(results) == 2

    def test_empty_store_returns_empty(self):
        store = VectorStore(embedding_dim=4)
        query = np.random.rand(1, 4).astype(np.float32)
        assert store.search(query) == []

    def test_top_k_capped_at_ntotal(self):
        store = VectorStore(embedding_dim=4)
        vecs = np.random.rand(2, 4).astype(np.float32)
        store.add(["a", "b"], vecs, [{}, {}])
        results = store.search(np.random.rand(1, 4).astype(np.float32), top_k=10)
        assert len(results) == 2

    def test_save_and_load(self, tmp_path):
        store = VectorStore(embedding_dim=4)
        vecs = np.random.rand(2, 4).astype(np.float32)
        store.add(["hello", "world"], vecs, [{"source": "s1"}, {"source": "s2"}])
        store.save(str(tmp_path))

        loaded = VectorStore.load(str(tmp_path))
        assert loaded.index.ntotal == 2
        assert len(loaded.chunks) == 2


# ── Loaders ───────────────────────────────────────────────────────────────────


class TestLoaders:
    def test_load_from_strings(self):
        docs = load_from_strings(["Hello", "World"], source="test")
        assert len(docs) == 2
        assert docs[0].content == "Hello"
        assert docs[0].metadata["source"] == "test"

    def test_load_text_file(self, tmp_path):
        from src.rag.loaders import load_text_file

        f = tmp_path / "sample.txt"
        f.write_text("Sample content for testing.")
        doc = load_text_file(str(f))
        assert "Sample content" in doc.content
        assert doc.metadata["type"] == "text"

    def test_load_json_file_list(self, tmp_path):
        import json

        from src.rag.loaders import load_json_file

        data = [{"content": "Item 1"}, {"content": "Item 2"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        docs = load_json_file(str(f))
        assert len(docs) == 2

    def test_load_directory(self, tmp_path):
        from src.rag.loaders import load_directory

        (tmp_path / "a.txt").write_text("File A content")
        (tmp_path / "b.txt").write_text("File B content")
        (tmp_path / "skip.pdf").write_text("ignored")
        docs = load_directory(str(tmp_path), extensions=[".txt"])
        assert len(docs) == 2


# ── RAGPipeline (mocked embedder) ─────────────────────────────────────────────


class TestRAGPipeline:
    @pytest.fixture
    def pipeline(self, tmp_path):
        """RAGPipeline with a mocked SentenceTransformer."""
        with patch("src.rag.pipeline.SentenceTransformer") as MockST:
            mock_st = MagicMock()
            mock_st.get_sentence_embedding_dimension.return_value = 16
            mock_st.encode.return_value = np.random.rand(5, 16).astype(np.float32)
            MockST.return_value = mock_st
            p = RAGPipeline(
                chunk_size=100, chunk_overlap=10, top_k=3, store_path=str(tmp_path / "store")
            )
            return p

    def test_ingest_returns_chunk_count(self, pipeline):
        docs = load_from_strings(["Short doc."] * 3)
        pipeline.embedder.encode.return_value = np.random.rand(3, 16).astype(np.float32)
        n = pipeline.ingest(docs)
        assert n > 0

    def test_build_context_format(self, pipeline):
        chunks = [
            RetrievedChunk(
                content="chunk text", metadata={"source": "s1"}, score=0.9, doc_id="abc"
            ),
        ]
        ctx = pipeline.build_context(chunks)
        assert "[Source 1: s1]" in ctx
        assert "chunk text" in ctx

    def test_build_context_empty(self, pipeline):
        ctx = pipeline.build_context([])
        assert "No relevant context" in ctx

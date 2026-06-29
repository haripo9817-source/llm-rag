# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by giving them access to external knowledge at inference time. Instead of relying solely on information encoded in the model's weights during training, RAG retrieves relevant documents from a knowledge base and includes them as context in the prompt.

## How RAG Works

The RAG pipeline consists of two main phases:

### 1. Indexing Phase (Offline)
- **Document loading**: Ingest documents from files, databases, or APIs
- **Chunking**: Split documents into manageable segments (typically 256–1024 tokens)
- **Embedding**: Convert chunks to dense vectors using an embedding model
- **Storing**: Save vectors in a vector database (FAISS, Pinecone, Chroma, Weaviate)

### 2. Retrieval Phase (Online, per query)
- **Query embedding**: Embed the user's question using the same model
- **Similarity search**: Find the top-k most relevant chunks via cosine similarity
- **Context assembly**: Format retrieved chunks into a context string
- **LLM generation**: Send `[context + question]` to the LLM for a grounded answer

## Why Use RAG?

| Problem | RAG Solution |
|---------|-------------|
| LLM knowledge cutoff | Inject current documents |
| Hallucinations | Ground answers in retrieved facts |
| Private/proprietary data | Keep data in your own vector store |
| Auditability | Cite sources inline |
| Cost | Smaller models with good retrieval outperform large models alone |

## Key Components

### Embedding Models
- `all-MiniLM-L6-v2`: Fast, 384-dim, great for English (used in this project)
- `text-embedding-3-small`: OpenAI's efficient embedding model
- `BAAI/bge-large-en-v1.5`: State-of-the-art open-source option

### Vector Databases
- **FAISS**: In-process, no server required, scales to millions of vectors
- **ChromaDB**: Python-native, persistence built-in, easy to start
- **Pinecone**: Managed, production-grade, serverless option
- **Weaviate**: Hybrid search (vector + keyword), GraphQL API

### LLM Integration
The retrieved context is injected into the system or user prompt. Best practices:
1. Cite sources with `[Source N]` notation
2. Instruct the model to answer ONLY from context
3. Include a fallback: "If the answer isn't in the context, say so"

## Advanced Techniques

### Hybrid Search
Combine dense vector search with sparse BM25 keyword search. Reciprocal Rank Fusion (RRF) merges both ranked lists. Effective for queries with exact terminology.

### Re-ranking
After retrieval, use a cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`) to re-score chunks against the query. More accurate than bi-encoder retrieval alone.

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer first, embed it, and search with that embedding instead of the raw query. Reduces the query-document mismatch problem.

### Parent-Child Chunking
Index small chunks for precise retrieval, but pass the parent chunk (larger context) to the LLM. Balances retrieval precision with generation quality.

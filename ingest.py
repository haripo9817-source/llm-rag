#!/usr/bin/env python3
"""
CLI tool for managing the RAG pipeline.

Usage:
  python ingest.py --docs docs/          # Ingest all docs in a directory
  python ingest.py --text "some text"    # Ingest inline text
  python ingest.py --query "my question" # Query the pipeline
  python ingest.py --stats               # Show index stats
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.pipeline import RAGPipeline
from src.rag.loaders import load_directory, load_from_strings
from src.api.claude_client import ClaudeRAGClient


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("--docs", help="Directory of documents to ingest")
    parser.add_argument("--text", help="Inline text to ingest")
    parser.add_argument("--source", default="cli", help="Source name for inline text")
    parser.add_argument("--query", help="Query to run against the index")
    parser.add_argument("--stats", action="store_true", help="Show index stats")
    parser.add_argument("--store", default="data/vector_store", help="Vector store path")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = RAGPipeline(store_path=args.store, top_k=args.top_k)

    if args.stats:
        print(f"Indexed vectors : {pipeline.store.index.ntotal}")
        print(f"Store path      : {args.store}")
        return

    if args.docs:
        docs = load_directory(args.docs)
        if not docs:
            print(f"No supported files found in {args.docs}")
            return
        n = pipeline.ingest(docs)
        print(f"Ingested {n} chunks from {len(docs)} documents.")

    if args.text:
        docs = load_from_strings([args.text], source=args.source)
        n = pipeline.ingest(docs)
        print(f"Ingested {n} chunks.")

    if args.query:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set. Showing raw retrieval only.")
            context, chunks = pipeline.query(args.query)
            print(f"\nRetrieved {len(chunks)} chunks:\n")
            for i, c in enumerate(chunks, 1):
                print(f"[{i}] (score={c.score:.4f}) {c.metadata.get('source','?')}")
                print(f"    {c.content[:200]}...\n")
            return

        llm = ClaudeRAGClient()
        context, chunks = pipeline.query(args.query)
        print(f"\nRetrieved {len(chunks)} chunks. Generating answer...\n")
        answer = llm.chat(args.query, context)
        print("─" * 60)
        print(f"Q: {args.query}")
        print("─" * 60)
        print(answer)
        print("─" * 60)
        for s in llm.format_sources(chunks):
            print(f"  [{s['index']}] {s['source']} (score: {s['score']})")


if __name__ == "__main__":
    main()

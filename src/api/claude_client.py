"""
Claude LLM client — RAG-aware chat with streaming support.
"""

import os
from typing import List, Dict, Iterator, Optional
import anthropic

from src.rag.pipeline import RetrievedChunk


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.

Guidelines:
- Answer ONLY from the provided context. If the answer isn't in the context, say so clearly.
- Cite your sources inline using [Source N] notation when referencing specific content.
- Be concise and direct. Avoid repeating the question back.
- If context is insufficient, suggest what additional information would help.
- For technical topics, prefer structured answers (steps, bullets) when it aids clarity."""


class ClaudeRAGClient:
    """
    Wraps the Anthropic SDK for RAG-augmented chat.
    Supports multi-turn conversation with injected retrieval context.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def _build_user_message(self, question: str, context: str) -> str:
        return f"""Context documents:
---
{context}
---

Question: {question}"""

    def chat(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """Single-turn or multi-turn (with history) RAG response."""
        messages = list(history or [])
        messages.append({"role": "user", "content": self._build_user_message(question, context)})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    def stream(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict]] = None,
    ) -> Iterator[str]:
        """Stream tokens for real-time UI updates."""
        messages = list(history or [])
        messages.append({"role": "user", "content": self._build_user_message(question, context)})

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def format_sources(self, chunks: List[RetrievedChunk]) -> List[Dict]:
        """Format retrieved chunks into a serializable sources list."""
        return [
            {
                "index": i + 1,
                "source": chunk.metadata.get("source", "Unknown"),
                "score": round(chunk.score, 4),
                "preview": chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""),
            }
            for i, chunk in enumerate(chunks)
        ]

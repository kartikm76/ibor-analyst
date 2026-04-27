"""Embedding provider stub — real embeddings disabled pending deployment decision."""
from __future__ import annotations
from typing import List


class EmbeddingProvider:
    """No-op embedding provider. Conversation RAG is disabled; quota/history still work."""

    model_name: str = "none"
    dimension: int = 0

    async def embed(self, text: str) -> List[float]:
        return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[] for _ in texts]

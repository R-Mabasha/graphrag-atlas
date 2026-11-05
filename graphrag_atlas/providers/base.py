from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from graphrag_atlas.models import Triple


class Embedder(ABC):
    """Abstract interface for text embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the output vector dimensionality (e.g. 1024, 1536)."""
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate a dense vector embedding for a single text passage."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vector embeddings for a batch of text passages."""
        ...


class LLM(ABC):
    """Abstract interface for language model generation and extraction."""

    @abstractmethod
    async def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        """Generate a text response or structured JSON from a prompt."""
        ...


class Reranker(ABC):
    """Abstract interface for cross-encoder query-passage relevance reranking."""

    @abstractmethod
    async def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """Return a list of (passage_index, relevance_score) sorted by score descending."""
        ...


class VisionExtractor(ABC):
    """Abstract interface for extracting semantic triples from architectural diagrams and images."""

    @abstractmethod
    async def extract_triples(self, image_bytes: bytes, mime_type: str) -> list[Triple]:
        """Parse image bytes and return extracted knowledge graph relationships."""
        ...

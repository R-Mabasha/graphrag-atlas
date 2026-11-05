from __future__ import annotations

import hashlib
import json
from typing import Any

from graphrag_atlas.providers.base import LLM, Embedder, Reranker


class MockEmbedder(Embedder):
    """Deterministic mock embedding provider for tests and offline development."""

    def __init__(self, dimension: int = 128) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> list[float]:
        # Hash text deterministically into pseudo-random floats
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(float(b) / 255.0) for b in h]
        # Repeat or slice to match target dimension
        while len(vec) < self._dim:
            vec.extend(vec)
        return vec[: self._dim]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]


class MockLLM(LLM):
    """Deterministic mock language model provider returning structured triples or summaries."""

    def __init__(self, default_response: str | None = None) -> None:
        self._default_response = default_response

    async def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        if self._default_response:
            return self._default_response

        # If schema requests triples, return synthetic triples from prompt
        if schema and "triples" in schema.get("properties", {}):
            return json.dumps(
                {
                    "triples": [
                        {
                            "source": "PaymentGateway",
                            "relation": "COMMUNICATES_WITH",
                            "target": "LedgerService",
                        },
                        {
                            "source": "LedgerService",
                            "relation": "POSTS_TO",
                            "target": "JournalDatabase",
                        },
                    ]
                }
            )

        return (
            "This subsystem manages transactional settlement and ledger reconciliation."
        )


class MockReranker(Reranker):
    """Deterministic mock reranker scoring passages by keyword overlap."""

    async def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        query_words = set(query.lower().split())
        scored: list[tuple[int, float]] = []
        for idx, text in enumerate(passages):
            overlap = len(query_words & set(text.lower().split()))
            score = float(overlap) / max(1.0, float(len(query_words)))
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

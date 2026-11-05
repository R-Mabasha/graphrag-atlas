from __future__ import annotations

import os

from graphrag_atlas.providers.base import Reranker


class CohereReranker(Reranker):
    """Cohere API cross-encoder reranking provider (e.g. rerank-v3.5)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-v3.5",
        top_n: int | None = None,
    ) -> None:
        self._model = model
        self._top_n = top_n
        self._api_key = (
            api_key or os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "Cohere API key must be provided or set in CO_API_KEY / COHERE_API_KEY."
            )

        try:
            import cohere

            self._client = cohere.AsyncClientV2(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Package 'cohere' is required. Install via `pip install graphrag-atlas[cohere]`."
            )

    async def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        if not passages:
            return []

        top_n = self._top_n or len(passages)
        response = await self._client.rerank(
            model=self._model, query=query, documents=passages, top_n=top_n
        )
        return [(item.index, float(item.relevance_score)) for item in response.results]

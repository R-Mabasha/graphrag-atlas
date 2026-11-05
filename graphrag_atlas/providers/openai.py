from __future__ import annotations

import os
from typing import Any

from graphrag_atlas.providers.base import LLM, Embedder


class OpenAIEmbedder(Embedder):
    """OpenAI embedding provider (e.g. text-embedding-3-small, text-embedding-3-large)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY."
            )

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Package 'openai' is required. Install via `pip install graphrag-atlas[openai]`."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            input=[text], model=self._model, dimensions=self._dimension
        )
        return resp.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            input=texts, model=self._model, dimensions=self._dimension
        )
        return [item.embedding for item in resp.data]


class OpenAILLM(LLM):
    """OpenAI language model provider (e.g. gpt-4o-mini, gpt-4o)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY."
            )

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Package 'openai' is required. Install via `pip install graphrag-atlas[openai]`."
            )

    async def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if schema:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content

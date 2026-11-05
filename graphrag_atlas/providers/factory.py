from __future__ import annotations

import os
from typing import Literal

from graphrag_atlas.providers.base import LLM, Embedder, Reranker


def resolve_providers(
    provider: Literal["openai", "aws", "mock"] | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
    **kwargs,
) -> tuple[Embedder, LLM, Reranker | None]:
    """
    Resolve and instantiate model providers based on explicit configuration or environment variables.
    Supports switching between 'openai', 'aws' Bedrock, and offline 'mock'.
    """
    # 1. Determine active provider if not explicitly given
    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif (
            os.getenv("AWS_ACCESS_KEY_ID")
            or os.getenv("AWS_DEFAULT_REGION")
            or os.getenv("AWS_REGION")
        ):
            provider = "aws"
        else:
            provider = "openai"  # Default fallback

    # 2. Resolve Embedder and LLM
    if not embedder or not llm:
        if provider == "mock":
            from graphrag_atlas.providers.mock import (
                MockEmbedder,
                MockLLM,
                MockReranker,
            )

            if not embedder:
                embedder = MockEmbedder(dimension=kwargs.get("dimension", 128))
            if not llm:
                llm = MockLLM()
            if not reranker:
                reranker = MockReranker()

        elif provider == "openai":
            from graphrag_atlas.providers.openai import OpenAIEmbedder, OpenAILLM

            if not embedder:
                embedder_model = kwargs.get("embedding_model", "text-embedding-3-small")
                dimension = kwargs.get("dimension", 1536)
                embedder = OpenAIEmbedder(model=embedder_model, dimension=dimension)
            if not llm:
                llm_model = kwargs.get("llm_model", "gpt-4o-mini")
                llm = OpenAILLM(model=llm_model)

        elif provider == "aws":
            from graphrag_atlas.providers.bedrock import BedrockEmbedder, BedrockLLM

            region = kwargs.get(
                "aws_region", os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            )
            if not embedder:
                embedder_model = kwargs.get(
                    "embedding_model", "amazon.titan-embed-text-v2:0"
                )
                dimension = kwargs.get("dimension", 1024)
                embedder = BedrockEmbedder(
                    region=region, model=embedder_model, dimension=dimension
                )
            if not llm:
                llm_model = kwargs.get("llm_model", "us.amazon.nova-lite-v1:0")
                llm = BedrockLLM(region=region, model=llm_model)
        else:
            raise ValueError(
                f"Unknown provider: '{provider}'. Supported providers: 'openai', 'aws'."
            )

    # 3. Resolve optional Reranker (e.g. Cohere)
    if not reranker and (
        os.getenv("CO_API_KEY")
        or os.getenv("COHERE_API_KEY")
        or "cohere_api_key" in kwargs
    ):
        from graphrag_atlas.providers.cohere import CohereReranker

        api_key = kwargs.get("cohere_api_key")
        reranker = CohereReranker(api_key=api_key)

    return embedder, llm, reranker

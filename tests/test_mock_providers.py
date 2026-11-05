import pytest

from graphrag_atlas.providers.factory import resolve_providers
from graphrag_atlas.providers.mock import MockEmbedder, MockReranker


@pytest.mark.asyncio
async def test_mock_embedder():
    embedder = MockEmbedder(dimension=64)
    assert embedder.dimension == 64

    vec1 = await embedder.embed_text("Sample text A")
    vec2 = await embedder.embed_text("Sample text A")
    vec3 = await embedder.embed_text("Different text B")

    assert len(vec1) == 64
    assert vec1 == vec2  # Deterministic hash
    assert vec1 != vec3


@pytest.mark.asyncio
async def test_mock_reranker():
    reranker = MockReranker()
    passages = [
        "Unrelated contents about biology",
        "Payment gateway settlement and ledger reconciliation",
        "Financial reporting overview",
    ]
    query = "payment settlement"
    results = await reranker.rerank(query, passages)

    assert len(results) == 3
    # Passage 1 (index 1) has both 'payment' and 'settlement'
    assert results[0][0] == 1
    assert results[0][1] > results[1][1]


def test_resolve_mock_provider():
    embedder, _llm, reranker = resolve_providers(provider="mock", dimension=32)
    assert embedder.dimension == 32
    assert reranker is not None

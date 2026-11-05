import pytest

from graphrag_atlas.domain.general import GeneralDomainPack
from graphrag_atlas.domain.presets.sap import SAPCleanCorePack
from graphrag_atlas.extractor.entity_extractor import EntityExtractor
from graphrag_atlas.models import Chunk
from graphrag_atlas.providers.mock import MockLLM


@pytest.mark.asyncio
async def test_entity_extractor_fast_path():
    sap_pack = SAPCleanCorePack()
    mock_llm = MockLLM()
    extractor = EntityExtractor(domain_pack=sap_pack, llm=mock_llm)

    text = "The custom program ZCUSTOM_POST posts transactions to ACDOCA via BAPI_ACC_DOCUMENT_POST."
    entities = extractor.extract_fast_path(text)

    # Fast-path regexes should catch ZCUSTOM_POST, BAPI_ACC_DOCUMENT_POST, and ACDOCA
    assert "ZCUSTOM_POST" in entities
    assert "BAPI_ACC_DOCUMENT_POST" in entities
    # ACDOCA should be canonicalized to Universal Journal Entry Line Items
    assert "Universal Journal Entry Line Items" in entities


@pytest.mark.asyncio
async def test_entity_extractor_semantic_triples():
    general_pack = GeneralDomainPack()
    # Mock LLM returning markdown fenced JSON with surrounding text
    mock_llm = MockLLM(
        default_response="""Here are the extracted relationships:
```json
{
  "triples": [
    {"source": "AuthService", "relation": "ISSUES_TOKEN", "target": "TokenStore"},
    {"source": "PaymentGateway", "relation": "DEPENDS_ON", "target": "AuthService"}
  ]
}
```
Hope this is helpful!"""
    )
    extractor = EntityExtractor(domain_pack=general_pack, llm=mock_llm)

    chunk = Chunk(
        chunk_id="test_doc:0",
        document_id="test_doc",
        tenant_id="tenant_1",
        text="AuthService issues tokens stored in TokenStore. PaymentGateway depends on AuthService.",
    )

    triples = await extractor.extract_semantic_triples(chunk)
    assert len(triples) == 2

    trip_map = {(t.source, t.relation, t.target) for t in triples}
    assert ("AuthService", "ISSUES_TOKEN", "TokenStore") in trip_map
    assert ("PaymentGateway", "DEPENDS_ON", "AuthService") in trip_map

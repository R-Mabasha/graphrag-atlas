from graphrag_atlas.domain.general import GeneralDomainPack
from graphrag_atlas.domain.presets.sap import SAPCleanCorePack


def test_general_domain_pack():
    pack = GeneralDomainPack()
    assert pack.name == "general"
    assert pack.canonicalize("OrderProcessor") == "OrderProcessor"
    assert pack.is_valid_entity("OrderProcessor") is True
    # Stoplist check
    assert pack.is_valid_entity("SYSTEM") is False
    assert pack.is_valid_entity("a") is False  # Length < 2


def test_sap_clean_core_pack():
    pack = SAPCleanCorePack()
    assert pack.name == "sap_clean_core"
    # Acronym resolution
    assert pack.canonicalize("ACDOCA") == "Universal Journal Entry Line Items"
    assert pack.canonicalize("RAP") == "RESTful Application Programming Model"
    # Stopwords
    assert pack.is_valid_entity("SAP") is False
    assert pack.is_valid_entity("ABAP") is False
    # Domain vocabulary check
    assert pack.matches_domain("How does posting to acdoca work?") is True
    assert pack.matches_domain("What is the weather in Paris?") is False

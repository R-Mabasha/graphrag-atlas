from __future__ import annotations

from graphrag_atlas.domain.base import DomainPack, RelationSpec

DEFAULT_RELATIONS = [
    RelationSpec(
        "CONTAINS", "Indicates structural containment or composition", "directed"
    ),
    RelationSpec(
        "DEPENDS_ON", "Indicates functional dependency between components", "directed"
    ),
    RelationSpec(
        "RELATES_TO", "Indicates a bidirectional semantic relationship", "undirected"
    ),
    RelationSpec(
        "IMPLEMENTS",
        "Indicates an implementation of a contract or interface",
        "directed",
    ),
    RelationSpec("CALLS", "Indicates an invocation or communication link", "directed"),
    RelationSpec(
        "DEFINES", "Indicates specification or declaration of an entity", "directed"
    ),
    RelationSpec(
        "USES", "Indicates usage of a utility, library, or resource", "directed"
    ),
    RelationSpec("PART_OF", "Indicates membership in a larger subsystem", "directed"),
]

DEFAULT_STOPLIST = {
    "THE",
    "THIS",
    "THAT",
    "THESE",
    "THOSE",
    "SYSTEM",
    "DATA",
    "FUNCTION",
    "PROCESS",
    "METHOD",
    "CODE",
    "TABLE",
    "USER",
    "FILE",
    "DOCUMENT",
    "ITEM",
    "NOTE",
    "EXAMPLE",
    "TEST",
    "INFO",
    "OBJECT",
    "FIELD",
}


class GeneralDomainPack(DomainPack):
    """Default open-domain pack suitable for general software, documentation, and knowledge bases."""

    def __init__(self) -> None:
        super().__init__(
            name="general",
            ontology=DEFAULT_RELATIONS,
            entity_regexes=[],  # Zero-shot LLM extraction is the primary general-purpose path
            stoplist=DEFAULT_STOPLIST,
            acronyms={},
            domain_vocabulary=set(),  # Empty means all queries pass the intent gate
        )

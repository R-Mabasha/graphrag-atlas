from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RelationSpec:
    """Defines a permitted or suggested relationship type in the knowledge graph."""

    name: str
    description: str = ""
    direction: Literal["directed", "undirected"] = "directed"


@dataclass
class DomainPack:
    """Encapsulates domain-specific ontology, regex patterns, acronyms, and vocabulary."""

    name: str = "general"
    ontology: list[RelationSpec] = field(default_factory=list)
    entity_regexes: list[re.Pattern] = field(default_factory=list)
    stoplist: set[str] = field(default_factory=set)
    acronyms: dict[str, str] = field(default_factory=dict)
    domain_vocabulary: set[str] = field(default_factory=set)

    def canonicalize(self, name: str) -> str:
        """Resolve aliases and acronyms to a canonical form."""
        cleaned = name.strip()
        upper = cleaned.upper()
        if upper in self.acronyms:
            return self.acronyms[upper]
        return cleaned

    def is_valid_entity(self, name: str) -> bool:
        """Check if a candidate string is a valid entity (not in stoplist, reasonable length)."""
        cleaned = name.strip()
        if len(cleaned) < 2 or len(cleaned) > 120:
            return False
        if cleaned.upper() in self.stoplist or cleaned.lower() in self.stoplist:
            return False
        # Filter purely numeric or punctuation strings
        return bool(re.search(r"[a-zA-Z]", cleaned))

    def matches_domain(self, text: str) -> bool:
        """Check whether the query or text contains domain-relevant vocabulary."""
        if not self.domain_vocabulary:
            return True
        tokens = set(re.findall(r"\b[A-Za-z0-9_]+\b", text.lower()))
        return bool(tokens & {v.lower() for v in self.domain_vocabulary})

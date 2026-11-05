from __future__ import annotations

import json
import re

from graphrag_atlas.domain.base import DomainPack
from graphrag_atlas.models import Chunk, Triple
from graphrag_atlas.providers.base import LLM

EXTRACTION_PROMPT_TEMPLATE = """You are an expert knowledge graph extractor. Extract meaningful entity relationships from the provided text.

PERMITTED RELATIONS:
{relations_desc}

RULES:
1. Extract relationships as structured triples (source, relation, target).
2. Only use relations from the permitted list if they fit, or descriptive uppercase verbs if none match.
3. Keep entity names concise, singular, and canonical (e.g. "OrderProcessor", not "the order processors").
4. Return ONLY a valid JSON object matching the schema: {{"triples": [{{"source": "...", "relation": "...", "target": "..."}}]}}

TEXT:
{text}
"""


class EntityExtractor:
    """Extracts entities and semantic relationship triples using regex fast-paths and zero-shot LLM."""

    def __init__(self, domain_pack: DomainPack, llm: LLM) -> None:
        self._domain = domain_pack
        self._llm = llm

    def extract_fast_path(self, text: str) -> set[str]:
        """Extract candidate entities using fast-path regex patterns and known acronyms."""
        entities: set[str] = set()
        for pattern in self._domain.entity_regexes:
            for match in pattern.finditer(text):
                entity = match.group(1) if match.groups() else match.group(0)
                if self._domain.is_valid_entity(entity):
                    entities.add(self._domain.canonicalize(entity))

        # Check known domain acronyms/identities
        if self._domain.acronyms:
            for acronym in self._domain.acronyms:
                if re.search(r"\b" + re.escape(acronym) + r"\b", text, re.IGNORECASE):
                    entities.add(self._domain.canonicalize(acronym))

        return entities

    async def extract_semantic_triples(self, chunk: Chunk) -> list[Triple]:
        """Extract semantic triples from chunk text using LLM."""
        relations_desc = (
            "\n".join(f"- {r.name}: {r.description}" for r in self._domain.ontology)
            or "- RELATES_TO, DEPENDS_ON, CONTAINS, CALLS, DEFINES, USES"
        )

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            relations_desc=relations_desc, text=chunk.text
        )

        schema = {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relation": {"type": "string"},
                            "target": {"type": "string"},
                        },
                        "required": ["source", "relation", "target"],
                    },
                }
            },
            "required": ["triples"],
        }

        try:
            raw_response = await self._llm.generate(prompt, schema=schema)
            # Parse json safely, extracting the innermost json object even if preamble text exists
            cleaned_json = raw_response.strip()
            if "```" in cleaned_json:
                code_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_json, re.DOTALL
                )
                if code_match:
                    cleaned_json = code_match.group(1)

            json_match = re.search(r"\{.*\}", cleaned_json, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group(0)

            data = json.loads(cleaned_json)
            raw_triples = data.get("triples", [])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raw_triples = []

        valid_triples: list[Triple] = []
        for item in raw_triples:
            src = self._domain.canonicalize(item.get("source", ""))
            tgt = self._domain.canonicalize(item.get("target", ""))
            rel = item.get("relation", "RELATES_TO").upper().strip().replace(" ", "_")

            if (
                self._domain.is_valid_entity(src)
                and self._domain.is_valid_entity(tgt)
                and src != tgt
            ):
                valid_triples.append(
                    Triple(
                        source=src,
                        relation=rel,
                        target=tgt,
                        weight=1.0,
                        metadata={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                        },
                    )
                )

        return valid_triples

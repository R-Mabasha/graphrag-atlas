from __future__ import annotations

from typing import Any

from graphrag_atlas.core.database import DatabaseManager
from graphrag_atlas.models import Triple


class GraphLookupEngine:
    """Executes bounded multi-hop traversals over MongoDB Atlas collections using $graphLookup."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        tenant_field: str = "tenant_id",
        edges_collection: str = "atlas_edges",
        max_depth: int = 2,
    ) -> None:
        self._db = db_manager
        self._tenant = tenant_field
        self._edges_coll_name = edges_collection
        self._max_depth = max_depth

    async def traverse(
        self,
        tenant_id: str,
        seed_entities: list[str],
        limit: int = 30,
        include_archived: bool = False,
    ) -> tuple[list[str], list[Triple]]:
        """
        Traverses outbound and inbound relationships from seed entities up to max_depth hops.
        Returns:
            (discovered_entity_names, discovered_edges)
        """
        if not seed_entities:
            return [], []

        base_match: dict[str, Any] = {self._tenant: tenant_id}
        if not include_archived:
            base_match["is_archived"] = {"$ne": True}

        # 1. Forward traversal (seed -> target)
        match_forward = dict(base_match)
        match_forward["source"] = {"$in": seed_entities}

        pipeline_forward: list[dict[str, Any]] = [
            {"$match": match_forward},
            {
                "$graphLookup": {
                    "from": self._edges_coll_name,
                    "startWith": "$target",
                    "connectFromField": "target",
                    "connectToField": "source",
                    "as": "network",
                    "maxDepth": max(0, self._max_depth - 1),
                    "restrictSearchWithMatch": base_match,
                }
            },
            {"$limit": limit},
        ]

        discovered_entities: set[str] = set(seed_entities)
        discovered_triples: list[Triple] = []

        async for doc in self._db.edges.aggregate(pipeline_forward):
            root_src = doc.get("source")
            root_tgt = doc.get("target")
            root_rel = doc.get("relation", "RELATES_TO")
            root_wt = doc.get("weight", 1.0)

            if root_tgt:
                discovered_entities.add(root_tgt)
                discovered_triples.append(
                    Triple(
                        source=root_src,
                        relation=root_rel,
                        target=root_tgt,
                        weight=root_wt,
                    )
                )

            for step in doc.get("network", []):
                s = step.get("source")
                t = step.get("target")
                r = step.get("relation", "RELATES_TO")
                w = step.get("weight", 1.0)
                if s and t:
                    discovered_entities.add(s)
                    discovered_entities.add(t)
                    discovered_triples.append(
                        Triple(source=s, relation=r, target=t, weight=w)
                    )

        # 2. Reverse traversal (inbound: source -> seed)
        match_reverse = dict(base_match)
        match_reverse["target"] = {"$in": seed_entities}

        pipeline_reverse: list[dict[str, Any]] = [
            {"$match": match_reverse},
            {"$limit": limit},
        ]

        async for doc in self._db.edges.aggregate(pipeline_reverse):
            root_src = doc.get("source")
            root_tgt = doc.get("target")
            root_rel = doc.get("relation", "RELATES_TO")
            root_wt = doc.get("weight", 1.0)
            if root_src:
                discovered_entities.add(root_src)
                discovered_triples.append(
                    Triple(
                        source=root_src,
                        relation=root_rel,
                        target=root_tgt,
                        weight=root_wt,
                    )
                )

        # Deduplicate discovered entities and triples
        unique_triples: dict[tuple[str, str, str], Triple] = {}
        for tr in discovered_triples:
            key = (tr.source, tr.relation, tr.target)
            if key not in unique_triples:
                unique_triples[key] = tr

        return list(discovered_entities), list(unique_triples.values())

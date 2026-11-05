from __future__ import annotations

from graphrag_atlas.core.database import DatabaseManager


class HubGuard:
    """Circuit breaker that prevents knowledge graph traversal from expanding through super-nodes."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        threshold: int = 200,
        tenant_field: str = "tenant_id",
    ) -> None:
        self._db = db_manager
        self._threshold = threshold
        self._tenant = tenant_field

    async def filter_seeds(
        self, tenant_id: str, candidate_entities: list[str]
    ) -> list[str]:
        """
        Filters out candidate seed entities whose degree exceeds the hub threshold.
        Reads degree via indexed lookup on (tenant_id, canonical_name).
        """
        if not candidate_entities:
            return []

        cursor = self._db.entities.find(
            {
                self._tenant: tenant_id,
                "canonical_name": {"$in": candidate_entities},
                "degree": {"$gt": self._threshold},
            },
            {"canonical_name": 1, "degree": 1},
        )

        hub_set: set[str] = set()
        async for doc in cursor:
            hub_set.add(doc["canonical_name"])

        # Retain only non-hub entities as traversal seeds
        safe_seeds = [e for e in candidate_entities if e not in hub_set]
        return safe_seeds

    def is_hub(self, degree: int) -> bool:
        """Check if a given degree qualifies as a super-node hub."""
        return degree > self._threshold

from __future__ import annotations

import networkx as nx
from networkx.algorithms.community import louvain_communities

from graphrag_atlas.core.database import DatabaseManager
from graphrag_atlas.models import Community
from graphrag_atlas.providers.base import LLM

COMMUNITY_SUMMARY_PROMPT = """You are an architectural analyst. Summarize the role and responsibilities of the following subsystem/community based on its member entities and relationships.

COMMUNITY MEMBERS:
{entities}

Provide a concise, 2-3 sentence overview of this subsystem's architectural responsibility and capabilities.
"""


class CommunityDetector:
    """Detects and summarizes architectural subsystems/communities using modularity graph partitioning."""

    def __init__(
        self, db_manager: DatabaseManager, llm: LLM, tenant_field: str = "tenant_id"
    ) -> None:
        self._db = db_manager
        self._llm = llm
        self._tenant = tenant_field

    async def detect_and_summarize(self, tenant_id: str) -> list[Community]:
        """
        Builds graph from edges, executes community partitioning, generates LLM summaries,
        and persists communities to MongoDB.
        """
        # 1. Fetch all edges for tenant
        cursor = self._db.edges.find(
            {self._tenant: tenant_id}, {"source": 1, "target": 1, "weight": 1}
        )
        graph = nx.Graph()

        async for doc in cursor:
            s = doc.get("source")
            t = doc.get("target")
            w = doc.get("weight", 1.0)
            if s and t:
                graph.add_edge(s, t, weight=w)

        if graph.number_of_nodes() < 3:
            return []

        # 2. Modularity partitioning (Louvain / Leiden equivalent)
        raw_communities = louvain_communities(graph, weight="weight", seed=42)

        communities: list[Community] = []
        for idx, member_set in enumerate(raw_communities):
            members = sorted(member_set)
            if len(members) < 2:
                continue

            # Summarize via LLM
            prompt = COMMUNITY_SUMMARY_PROMPT.format(entities=", ".join(members[:30]))
            summary = await self._llm.generate(prompt)

            comm = Community(
                community_id=f"{tenant_id}:comm_{idx}",
                tenant_id=tenant_id,
                title=f"Subsystem {idx + 1} ({members[0]})",
                summary=summary.strip(),
                entities=members,
                level=0,
            )
            communities.append(comm)

        # 3. Persist to MongoDB (clear previous communities for tenant to prevent stale records)
        await self._db.communities.delete_many({self._tenant: tenant_id})
        if communities:
            comm_docs = [c.model_dump() for c in communities]
            await self._db.communities.insert_many(comm_docs)

        return communities

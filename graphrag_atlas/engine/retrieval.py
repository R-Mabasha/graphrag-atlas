from __future__ import annotations

import asyncio
import re
from typing import Any

from pymongo.errors import PyMongoError

from graphrag_atlas.core.config import AtlasConfig
from graphrag_atlas.core.database import DatabaseManager
from graphrag_atlas.domain.base import DomainPack
from graphrag_atlas.engine.graph_lookup import GraphLookupEngine
from graphrag_atlas.engine.hub_guard import HubGuard
from graphrag_atlas.engine.rrf import reciprocal_rank_fusion
from graphrag_atlas.engine.stitcher import PassageStitcher
from graphrag_atlas.models import Passage, SearchResponse
from graphrag_atlas.providers.base import Embedder, Reranker


class HybridRetrievalEngine:
    """
    Coordinates 8-level single-store hybrid retrieval:
    BM25 + Dense Vector + 2-hop Bounded GraphLookup + Hub Guard + Fair Reranking.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedder: Embedder,
        domain_pack: DomainPack,
        config: AtlasConfig,
        reranker: Reranker | None = None,
        tenant_field: str = "tenant_id",
    ) -> None:
        self._db = db_manager
        self._embedder = embedder
        self._domain = domain_pack
        self._config = config
        self._reranker = reranker
        self._tenant = tenant_field

        self._hub_guard = HubGuard(
            db_manager, threshold=config.hub_degree_threshold, tenant_field=tenant_field
        )
        self._graph_lookup = GraphLookupEngine(
            db_manager,
            tenant_field=tenant_field,
            edges_collection=config.edges_collection,
            max_depth=config.max_graph_depth,
        )

    async def _search_bm25(
        self,
        tenant_id: str,
        query: str,
        limit: int = 25,
        include_archived: bool = False,
        project_id: str | None = None,
    ) -> list[str]:
        """Search chunks using Atlas Search (Lucene BM25). Falls back to regex if index absent."""
        filter_conditions: list[dict[str, Any]] = [
            {"equals": {"path": self._tenant, "value": tenant_id}}
        ]
        if project_id is not None:
            filter_conditions.append(
                {"equals": {"path": "project_id", "value": project_id}}
            )
        if not include_archived:
            filter_conditions.append(
                {"equals": {"path": "is_archived", "value": False}}
            )

        pipeline: list[dict[str, Any]] = [
            {
                "$search": {
                    "index": self._config.search_index_name,
                    "compound": {
                        "filter": filter_conditions,
                        "must": [{"text": {"query": query, "path": "text"}}],
                    },
                }
            },
            {"$limit": limit},
            {"$project": {"chunk_id": 1}},
        ]

        try:
            results: list[str] = []
            async for doc in self._db.chunks.aggregate(pipeline):
                results.append(doc["chunk_id"])
            return results
        except PyMongoError:
            # Fallback for local testing or un-indexed collections
            words = [re_w for re_w in query.split() if len(re_w) > 3]
            if not words:
                return []
            pattern = "|".join(re.escape(re_w) for re_w in words)
            find_filter: dict[str, Any] = {
                self._tenant: tenant_id,
                "text": {"$regex": pattern, "$options": "i"},
            }
            if project_id is not None:
                find_filter["project_id"] = project_id
            if not include_archived:
                find_filter["is_archived"] = {"$ne": True}
            cursor = self._db.chunks.find(find_filter, {"chunk_id": 1}).limit(limit)
            return [doc["chunk_id"] async for doc in cursor]

    async def _search_vector(
        self,
        tenant_id: str,
        query_vector: list[float],
        limit: int = 25,
        include_archived: bool = False,
        project_id: str | None = None,
    ) -> list[str]:
        """Search chunks using Atlas Vector Search ($vectorSearch)."""
        vector_filter: dict[str, Any] = {self._tenant: tenant_id}
        if project_id is not None:
            vector_filter["project_id"] = project_id
        if not include_archived:
            vector_filter["is_archived"] = {"$ne": True}

        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self._config.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": vector_filter,
                }
            },
            {"$project": {"chunk_id": 1}},
        ]

        try:
            results: list[str] = []
            async for doc in self._db.chunks.aggregate(pipeline):
                results.append(doc["chunk_id"])
            return results
        except PyMongoError:
            # If vector search index is not yet built, return empty list gracefully
            return []

    async def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        rerank_score_threshold: float = 0.05,
        include_archived: bool = False,
        project_id: str | None = None,
    ) -> SearchResponse:
        """Execute full hybrid retrieval pipeline with optional project scoping."""
        # 1. Embed query
        query_vector = await self._embedder.embed_text(query)

        # 2. Concurrent BM25 and Vector Search
        bm25_ids, vector_ids = await asyncio.gather(
            self._search_bm25(
                tenant_id,
                query,
                limit=self._config.max_branching_per_hop,
                include_archived=include_archived,
                project_id=project_id,
            ),
            self._search_vector(
                tenant_id,
                query_vector,
                limit=self._config.max_branching_per_hop,
                include_archived=include_archived,
                project_id=project_id,
            ),
        )

        # 3. Extract candidate graph seeds from query and top lexical hits
        candidate_seeds: set[str] = set()
        for pattern in self._domain.entity_regexes:
            for match in pattern.finditer(query):
                candidate_seeds.add(self._domain.canonicalize(match.group(0)))

        # Also collect entities mentioned in the top 3 BM25/Vector chunks
        seed_chunk_ids = bm25_ids[:3] + vector_ids[:3]
        if seed_chunk_ids:
            chunk_seed_filter: dict[str, Any] = {
                self._tenant: tenant_id,
                "chunk_id": {"$in": seed_chunk_ids},
            }
            if project_id is not None:
                chunk_seed_filter["project_id"] = project_id
            cursor = self._db.chunks.find(
                chunk_seed_filter,
                {"entities": 1},
            )
            async for doc in cursor:
                for ent in doc.get("entities", []):
                    candidate_seeds.add(ent)

        # Fallback: if candidate seeds are still empty, perform fast B-Tree lookup of query tokens
        if not candidate_seeds:
            query_tokens = [
                self._domain.canonicalize(w)
                for w in re.findall(r"\b[A-Za-z0-9_]{3,}\b", query)
                if not self._domain.stoplist or w.upper() not in self._domain.stoplist
            ]
            if query_tokens:
                ent_query: dict[str, Any] = {
                    self._tenant: tenant_id,
                    "canonical_name": {"$in": query_tokens},
                }
                if project_id is not None:
                    ent_query["project_id"] = project_id
                ent_cursor = self._db.entities.find(
                    ent_query,
                    {"canonical_name": 1},
                ).limit(self._config.max_graph_seeds)
                async for ent_doc in ent_cursor:
                    candidate_seeds.add(ent_doc["canonical_name"])

        # 4. Hub Guard: Filter out super-nodes exceeding threshold
        safe_seeds = await self._hub_guard.filter_seeds(
            tenant_id,
            list(candidate_seeds)[: self._config.max_graph_seeds],
            project_id=project_id,
        )

        # 5. Bounded Graph Traversal
        discovered_entities, discovered_triples = await self._graph_lookup.traverse(
            tenant_id,
            safe_seeds,
            limit=self._config.max_branching_per_hop,
            include_archived=include_archived,
            project_id=project_id,
        )

        # 6. Graph-Expanded Chunks: Find chunks referencing discovered entities
        graph_chunk_ids: list[str] = []
        if discovered_entities:
            g_filter: dict[str, Any] = {
                self._tenant: tenant_id,
                "entities": {"$in": discovered_entities},
            }
            if project_id is not None:
                g_filter["project_id"] = project_id
            if not include_archived:
                g_filter["is_archived"] = {"$ne": True}

            g_cursor = self._db.chunks.find(
                g_filter,
                {"chunk_id": 1},
            ).limit(self._config.max_branching_per_hop)
            async for doc in g_cursor:
                graph_chunk_ids.append(doc["chunk_id"])

        # 7. Fair Reciprocal Rank Fusion across BM25, Vector, and Graph
        fused_scored_tuples = reciprocal_rank_fusion(
            [bm25_ids, vector_ids, graph_chunk_ids], k=self._config.rrf_k
        )

        top_candidates = fused_scored_tuples[:25]
        if not top_candidates:
            return SearchResponse(
                query=query, tenant_id=tenant_id, project_id=project_id
            )

        # Fetch chunk documents
        candidate_ids = [c_id for c_id, _ in top_candidates]

        chunk_docs: dict[str, dict] = {}
        c_filter: dict[str, Any] = {
            self._tenant: tenant_id,
            "chunk_id": {"$in": candidate_ids},
        }
        if project_id is not None:
            c_filter["project_id"] = project_id
        if not include_archived:
            c_filter["is_archived"] = {"$ne": True}

        cursor = self._db.chunks.find(c_filter)
        async for doc in cursor:
            chunk_docs[doc["chunk_id"]] = doc

        # Assemble candidate passages
        passages: list[Passage] = []
        for c_id, rrf_score in top_candidates:
            doc = chunk_docs.get(c_id)
            if not doc:
                continue

            # Determine discovery channel
            is_bm25 = c_id in bm25_ids
            is_vec = c_id in vector_ids
            is_grp = c_id in graph_chunk_ids

            source_type = "hybrid"
            if is_bm25 and not is_vec and not is_grp:
                source_type = "bm25"
            elif is_vec and not is_bm25 and not is_grp:
                source_type = "vector"
            elif is_grp and not is_bm25 and not is_vec:
                source_type = "graph"

            passages.append(
                Passage(
                    chunk_id=c_id,
                    document_id=doc.get("document_id", ""),
                    text=doc.get("text", ""),
                    score=rrf_score,
                    source_type=source_type,
                    heading_breadcrumbs=doc.get("heading_breadcrumbs", []),
                    metadata={"chunk_index": doc.get("chunk_index", 0)},
                )
            )

        # 8. Fair Cross-Encoder Reranking (if configured)
        if self._reranker and passages:
            texts = [p.text for p in passages]
            rerank_results = await self._reranker.rerank(query, texts)

            # Re-score all passages fairly based on reranker (no exemptions or slot reservation)
            reranked_passages: list[Passage] = []
            for idx, rerank_score in rerank_results:
                if rerank_score >= rerank_score_threshold:
                    p = passages[idx]
                    p.score = rerank_score
                    reranked_passages.append(p)
            passages = reranked_passages

        # 9. Stitch adjacent chunks
        passages = PassageStitcher.stitch(passages)[:top_k]

        # 10. Community Context
        community_summaries: list[str] = []
        if discovered_entities:
            comm_filter: dict[str, Any] = {
                self._tenant: tenant_id,
                "entities": {"$in": discovered_entities},
            }
            if project_id is not None:
                comm_filter["project_id"] = project_id
            comm_cursor = self._db.communities.find(
                comm_filter,
                {"summary": 1},
            ).limit(2)
            async for c_doc in comm_cursor:
                community_summaries.append(c_doc["summary"])

        return SearchResponse(
            query=query,
            tenant_id=tenant_id,
            project_id=project_id,
            passages=passages,
            graph_entities=list(discovered_entities)[:20],
            graph_edges=discovered_triples[:20],
            community_summaries=community_summaries,
            retrieval_metadata={
                "bm25_count": len(bm25_ids),
                "vector_count": len(vector_ids),
                "graph_discovered_entities": len(discovered_entities),
                "graph_chunks_fused": len(graph_chunk_ids),
            },
        )

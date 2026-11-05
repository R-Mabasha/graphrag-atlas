from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from graphrag_atlas.core.config import AtlasConfig
from graphrag_atlas.core.database import DatabaseManager
from graphrag_atlas.core.indexes import IndexManager
from graphrag_atlas.core.lifecycle import DocumentLifecycleManager
from graphrag_atlas.domain.base import DomainPack
from graphrag_atlas.domain.general import GeneralDomainPack
from graphrag_atlas.engine.leiden import CommunityDetector
from graphrag_atlas.engine.retrieval import HybridRetrievalEngine
from graphrag_atlas.extractor.ast_graph import ASTGraphBuilder
from graphrag_atlas.extractor.chunker import HeadingAwareChunker
from graphrag_atlas.extractor.entity_extractor import EntityExtractor
from graphrag_atlas.models import (
    Chunk,
    Community,
    DocumentInfo,
    SearchResponse,
    TenantStats,
)
from graphrag_atlas.providers.base import LLM, Embedder, Reranker
from graphrag_atlas.providers.factory import resolve_providers


class AtlasGraphRAG:
    """
    Unified client for single-store Hybrid GraphRAG on MongoDB Atlas.
    Integrates BM25, Dense Vector Search, AST hierarchies, Bounded GraphLookup,
    Hub Guard circuit breaker, and dynamic Leiden communities.
    """

    def __init__(
        self,
        mongo_uri: str,
        database: str = "graphrag",
        tenant_field: str = "tenant_id",
        provider: Literal["openai", "aws"] | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        domain_pack: DomainPack | None = None,
        config: AtlasConfig | None = None,
        **provider_kwargs,
    ) -> None:
        self._config = config or AtlasConfig()
        self._tenant_field = tenant_field
        self._domain = domain_pack or GeneralDomainPack()

        # 1. Database layer
        self._db = DatabaseManager(
            mongo_uri=mongo_uri, database_name=database, config=self._config
        )
        self._indexes = IndexManager(self._db, tenant_field=self._tenant_field)
        self._lifecycle = DocumentLifecycleManager(
            self._db, tenant_field=self._tenant_field
        )

        # 2. Model providers (automatic resolution for OpenAI or AWS Bedrock)
        self._embedder, self._llm, self._reranker = resolve_providers(
            provider=provider,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            **provider_kwargs,
        )

        # 3. Extractors and retrieval engine
        self._chunker = HeadingAwareChunker()
        self._ast_builder = ASTGraphBuilder()
        self._entity_extractor = EntityExtractor(self._domain, self._llm)
        self._community_detector = CommunityDetector(
            self._db, self._llm, tenant_field=self._tenant_field
        )

        self._retrieval = HybridRetrievalEngine(
            db_manager=self._db,
            embedder=self._embedder,
            domain_pack=self._domain,
            config=self._config,
            reranker=self._reranker,
            tenant_field=self._tenant_field,
        )

    async def setup_indexes(self) -> dict[str, list[str]]:
        """Create and verify all necessary B-Tree, compound tenant, and multikey indexes."""
        return await self._indexes.ensure_indexes()

    async def ingest(
        self,
        file_path: str | Path | None = None,
        text: str | None = None,
        tenant_id: str = "default",
        document_id: str | None = None,
        force_reingest: bool = False,
    ) -> str:
        """
        Ingest a document or text into the MongoDB Atlas Hybrid Graph store.
        Performs content-hashing, chunking, embedding, and AST+semantic graph extraction.
        """
        if not file_path and not text:
            raise ValueError("Either file_path or text must be provided.")

        # Read content
        file_name = "raw_text.txt"
        if file_path:
            p = Path(file_path)
            file_name = p.name
            document_id = document_id or p.stem
            if p.suffix.lower() in [".pdf", ".docx", ".xlsx"]:
                try:
                    from markitdown import MarkItDown

                    conv = MarkItDown()
                    res = conv.convert(str(p))
                    raw_text = res.text_content
                except (ImportError, OSError, ValueError, RuntimeError):
                    raw_text = p.read_text(encoding="utf-8", errors="ignore")
            else:
                raw_text = p.read_text(encoding="utf-8", errors="ignore")
        else:
            raw_text = text or ""
            document_id = document_id or f"doc_{hash(raw_text) & 0xFFFFFFFF}"

        # 1. Check content hash for deduplication and versioning
        content_hash = self._lifecycle.hash_content(raw_text.encode("utf-8"))
        existing = await self._lifecycle.get_document(tenant_id, document_id)
        doc_version = 1
        cached_chunks: dict[str, dict] = {}

        if existing:
            if existing.get("content_hash") == content_hash and not force_reingest:
                return document_id  # Unchanged, skip re-ingestion
            doc_version = existing.get("version", 1) + 1

            # Fetch existing chunks for incremental chunk-level caching
            if not force_reingest:
                async for c_doc in self._db.chunks.find(
                    {self._tenant_field: tenant_id, "document_id": document_id}
                ):
                    if c_doc.get("content_hash"):
                        cached_chunks[c_doc["content_hash"]] = c_doc

            # Clear previous version artifacts cleanly
            await self._lifecycle.delete_document(tenant_id, document_id)

        # 2. Chunking with heading breadcrumbs
        chunks = self._chunker.chunk_document(
            raw_text, document_id=document_id, tenant_id=tenant_id
        )
        if not chunks:
            return document_id

        # 3. Dense Embeddings (reuse cached embeddings for unchanged chunks)
        uncached_chunks: list[Chunk] = []
        for c in chunks:
            if c.content_hash in cached_chunks:
                cached = cached_chunks[c.content_hash]
                c.embedding = cached.get("embedding")
                c.entities = cached.get("entities", [])
            else:
                uncached_chunks.append(c)

        if uncached_chunks:
            chunk_texts = [c.text for c in uncached_chunks]
            embeddings = await self._embedder.embed_batch(chunk_texts)
            for c, emb in zip(uncached_chunks, embeddings):
                c.embedding = emb

        # 4. AST Structural Graph Edges
        ast_edges = self._ast_builder.extract_structural_edges(
            raw_text, document_id=document_id
        )
        all_triples = list(ast_edges)
        entity_degrees: dict[str, int] = {}
        for ast_tr in ast_edges:
            entity_degrees[ast_tr.source] = entity_degrees.get(ast_tr.source, 0) + 1
            entity_degrees[ast_tr.target] = entity_degrees.get(ast_tr.target, 0) + 1

        # 5. Semantic Triples & Entity Extraction (per chunk)
        for chunk in chunks:
            if chunk in uncached_chunks:
                fast_entities = self._entity_extractor.extract_fast_path(chunk.text)
                chunk.entities = list(fast_entities)

                semantic_triples = (
                    await self._entity_extractor.extract_semantic_triples(chunk)
                )
                for tr in semantic_triples:
                    all_triples.append(tr)
                    chunk.entities.append(tr.source)
                    chunk.entities.append(tr.target)
                    entity_degrees[tr.source] = entity_degrees.get(tr.source, 0) + 1
                    entity_degrees[tr.target] = entity_degrees.get(tr.target, 0) + 1

                chunk.entities = sorted(set(chunk.entities))
            else:
                for ent in chunk.entities:
                    entity_degrees[ent] = entity_degrees.get(ent, 0) + 1

        # 6. Bulk Insert Chunks
        chunk_docs = [c.model_dump() for c in chunks]
        for c_doc in chunk_docs:
            c_doc[self._tenant_field] = tenant_id
        if chunk_docs:
            await self._db.chunks.insert_many(chunk_docs)

        # 7. Upsert Edges with Weight Reinforcement (eliminates duplicate edge rows)
        now = datetime.now(timezone.utc).isoformat()
        for tr in all_triples:
            await self._db.edges.update_one(
                {
                    self._tenant_field: tenant_id,
                    "source": tr.source,
                    "relation": tr.relation,
                    "target": tr.target,
                },
                {
                    "$inc": {"weight": tr.weight},
                    "$addToSet": {"document_ids": document_id},
                    "$set": {"is_archived": False, "updated_at": now},
                    "$setOnInsert": {
                        self._tenant_field: tenant_id,
                        "source": tr.source,
                        "relation": tr.relation,
                        "target": tr.target,
                        "created_at": now,
                    },
                },
                upsert=True,
            )

        # 8. Upsert Entities
        for ent_name, added_degree in entity_degrees.items():
            await self._db.entities.update_one(
                {self._tenant_field: tenant_id, "canonical_name": ent_name},
                {
                    "$inc": {"degree": added_degree, "mentions": 1},
                    "$setOnInsert": {
                        self._tenant_field: tenant_id,
                        "name": ent_name,
                        "canonical_name": ent_name,
                        "entity_type": "CONCEPT",
                    },
                },
                upsert=True,
            )

        # 9. Record Document Metadata
        await self._lifecycle.record_document(
            tenant_id=tenant_id,
            document_id=document_id,
            file_name=file_name,
            content_hash=content_hash,
            chunk_count=len(chunks),
            entity_count=len(entity_degrees),
            edge_count=len(all_triples),
            version=doc_version,
            status="active",
        )

        return document_id

    async def reingest(
        self,
        file_path: str | Path | None = None,
        text: str | None = None,
        tenant_id: str = "default",
        document_id: str | None = None,
        force: bool = True,
    ) -> str:
        """
        Re-ingest a document with sequential version increment and full graph synchronization.
        Bypasses content-hash match when force=True.
        """
        return await self.ingest(
            file_path=file_path,
            text=text,
            tenant_id=tenant_id,
            document_id=document_id,
            force_reingest=force,
        )

    async def archive_document(
        self, tenant_id: str, document_id: str, reason: str = ""
    ) -> bool:
        """
        Soft-delete / archive a document. Excludes its chunks and relationships
        from standard retrieval and recalibrates active entity degrees.
        """
        return await self._lifecycle.archive_document(
            tenant_id=tenant_id, document_id=document_id, reason=reason
        )

    async def restore_document(self, tenant_id: str, document_id: str) -> bool:
        """Restore a previously archived document and re-activate its chunks and relationships."""
        return await self._lifecycle.restore_document(
            tenant_id=tenant_id, document_id=document_id
        )

    async def get_tenant_stats(self, tenant_id: str = "default") -> TenantStats:
        """Compute full storage, document status, and graph diagnostics for a tenant."""
        return await self._lifecycle.get_tenant_stats(
            tenant_id=tenant_id, hub_threshold=self._config.hub_degree_threshold
        )

    async def vacuum(self, tenant_id: str = "default") -> dict[str, int]:
        """Perform graph integrity repair, orphan entity cleanup, and degree recalibration."""
        return await self._lifecycle.vacuum(tenant_id=tenant_id)

    async def list_documents(
        self,
        tenant_id: str = "default",
        status: Literal["active", "archived", "all"] = "active",
        limit: int = 50,
        skip: int = 0,
    ) -> list[DocumentInfo]:
        """List documents for a tenant with status filtering and pagination."""
        return await self._lifecycle.list_documents(
            tenant_id=tenant_id, status=status, limit=limit, skip=skip
        )

    async def get_document(
        self, tenant_id: str = "default", document_id: str = ""
    ) -> dict | None:
        """Retrieve existing document record metadata."""
        return await self._lifecycle.get_document(
            tenant_id=tenant_id, document_id=document_id
        )

    async def search(
        self,
        query: str,
        tenant_id: str = "default",
        top_k: int = 5,
        rerank_score_threshold: float = 0.05,
        include_archived: bool = False,
    ) -> SearchResponse:
        """Execute hybrid search combining BM25, Vector, and Bounded Graph expansion."""
        return await self._retrieval.search(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k,
            rerank_score_threshold=rerank_score_threshold,
            include_archived=include_archived,
        )

    async def build_communities(self, tenant_id: str = "default") -> list[Community]:
        """Trigger modularity graph partitioning and LLM subsystem summarization."""
        return await self._community_detector.detect_and_summarize(tenant_id)

    async def delete_document(self, tenant_id: str, document_id: str) -> dict[str, int]:
        """Cascading delete of document chunks, edges, and orphan entities."""
        return await self._lifecycle.delete_document(tenant_id, document_id)

    async def close(self) -> None:
        """Close MongoDB database client connections."""
        await self._db.close()

from __future__ import annotations

import re
import uuid
from typing import Any

import pytest
from pymongo.errors import OperationFailure

from graphrag_atlas.client import AtlasGraphRAG
from graphrag_atlas.core.config import AtlasConfig
from graphrag_atlas.core.lifecycle import DocumentLifecycleManager
from graphrag_atlas.domain.general import GeneralDomainPack
from graphrag_atlas.engine.retrieval import HybridRetrievalEngine
from graphrag_atlas.extractor.chunker import HeadingAwareChunker
from graphrag_atlas.providers.mock import MockEmbedder


def _matches_filter(doc: dict[str, Any], filter_doc: dict[str, Any]) -> bool:
    for k, v in filter_doc.items():
        if k == "$or":
            if not any(_matches_filter(doc, cond) for cond in v):
                return False
        elif isinstance(v, dict):
            if "$in" in v:
                val = doc.get(k)
                target_in = v["$in"]
                if isinstance(val, list):
                    if not any(item in target_in for item in val):
                        return False
                elif val not in target_in:
                    return False
            if "$nin" in v:
                val = doc.get(k)
                target_nin = v["$nin"]
                if isinstance(val, list):
                    if any(item in target_nin for item in val):
                        return False
                elif val in target_nin:
                    return False
            if "$gt" in v and doc.get(k, 0) <= v["$gt"]:
                return False
            if "$ne" in v and doc.get(k) == v["$ne"]:
                return False
            if "$regex" in v:
                pattern = v["$regex"]
                flags = re.IGNORECASE if v.get("$options") == "i" else 0
                val = doc.get(k, "")
                if not re.search(pattern, str(val), flags):
                    return False
        else:
            val = doc.get(k)
            if isinstance(val, list):
                if v not in val:
                    return False
            elif val != v:
                return False
    return True


class AsyncCursorMock:
    def __init__(self, items: list[dict[str, Any]]):
        self.items = list(items)

    def sort(self, key: str, direction: int = 1) -> AsyncCursorMock:
        self.items.sort(key=lambda d: d.get(key, ""), reverse=(direction == -1))
        return self

    def skip(self, n: int) -> AsyncCursorMock:
        self.items = self.items[n:]
        return self

    def limit(self, n: int) -> AsyncCursorMock:
        if n > 0:
            self.items = self.items[:n]
        return self

    def __aiter__(self):
        self._iter = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class AsyncCollectionMock:
    def __init__(self):
        self.docs: list[dict[str, Any]] = []

    async def find_one(self, filter_doc: dict[str, Any]) -> dict[str, Any] | None:
        for doc in self.docs:
            if _matches_filter(doc, filter_doc):
                return doc
        return None

    def find(
        self, filter_doc: dict[str, Any], projection: dict[str, Any] | None = None
    ) -> AsyncCursorMock:
        matched = [dict(doc) for doc in self.docs if _matches_filter(doc, filter_doc)]
        return AsyncCursorMock(matched)

    async def count_documents(self, filter_doc: dict[str, Any]) -> int:
        return sum(1 for doc in self.docs if _matches_filter(doc, filter_doc))

    async def update_one(
        self,
        filter_doc: dict[str, Any],
        update_doc: dict[str, Any],
        upsert: bool = False,
    ) -> None:
        existing = await self.find_one(filter_doc)
        if existing:
            if "$set" in update_doc:
                existing.update(update_doc["$set"])
            if "$unset" in update_doc:
                for k in update_doc["$unset"]:
                    existing.pop(k, None)
            if "$inc" in update_doc:
                for k, v in update_doc["$inc"].items():
                    existing[k] = existing.get(k, 0) + v
            if "$addToSet" in update_doc:
                for k, v in update_doc["$addToSet"].items():
                    if k not in existing or not isinstance(existing[k], list):
                        existing[k] = []
                    if v not in existing[k]:
                        existing[k].append(v)
        elif upsert:
            new_doc = dict(filter_doc)
            if "_id" not in new_doc:
                new_doc["_id"] = str(uuid.uuid4())
            if "$setOnInsert" in update_doc:
                new_doc.update(update_doc["$setOnInsert"])
            if "$set" in update_doc:
                new_doc.update(update_doc["$set"])
            if "$addToSet" in update_doc:
                for k, v in update_doc["$addToSet"].items():
                    new_doc[k] = [v]
            if "$inc" in update_doc:
                for k, v in update_doc["$inc"].items():
                    new_doc[k] = v
            self.docs.append(new_doc)

    async def update_many(
        self, filter_doc: dict[str, Any], update_doc: dict[str, Any]
    ) -> None:
        for doc in self.docs:
            if _matches_filter(doc, filter_doc):
                if "$set" in update_doc:
                    doc.update(update_doc["$set"])
                if "$pull" in update_doc:
                    for pk, pv in update_doc["$pull"].items():
                        if pk in doc and isinstance(doc[pk], list):
                            doc[pk] = [item for item in doc[pk] if item != pv]
                if "$inc" in update_doc:
                    for ik, iv in update_doc["$inc"].items():
                        doc[ik] = doc.get(ik, 0) + iv

    async def delete_one(self, filter_doc: dict[str, Any]):
        class Result:
            deleted_count = 0

        res = Result()
        for idx, doc in enumerate(self.docs):
            if _matches_filter(doc, filter_doc):
                del self.docs[idx]
                res.deleted_count = 1
                break
        return res

    async def delete_many(self, filter_doc: dict[str, Any]):
        class Result:
            deleted_count = 0

        res = Result()
        new_docs = []
        for doc in self.docs:
            if _matches_filter(doc, filter_doc):
                res.deleted_count += 1
            else:
                new_docs.append(doc)
        self.docs = new_docs
        return res

    async def insert_one(self, doc: dict[str, Any]):
        class Result:
            inserted_id = None

        res = Result()
        item = dict(doc)
        if "_id" not in item:
            item["_id"] = str(uuid.uuid4())
        self.docs.append(item)
        res.inserted_id = item["_id"]
        return res

    async def insert_many(self, docs: list[dict[str, Any]]):
        class Result:
            def __init__(self):
                self.inserted_ids: list[Any] = []

        res = Result()
        for d in docs:
            item = dict(d)
            if "_id" not in item:
                item["_id"] = str(uuid.uuid4())
            self.docs.append(item)
            res.inserted_ids.append(item["_id"])
        return res

    def aggregate(self, pipeline: list[dict[str, Any]]) -> AsyncCursorMock:
        current_docs = [dict(d) for d in self.docs]
        for stage in pipeline:
            if "$search" in stage or "$vectorSearch" in stage:
                raise OperationFailure(
                    "Mock Atlas index not simulated; fallback required"
                )
            if "$match" in stage:
                current_docs = [
                    d for d in current_docs if _matches_filter(d, stage["$match"])
                ]
            if "$graphLookup" in stage:
                gl = stage["$graphLookup"]
                start_field = gl["startWith"].lstrip("$")
                connect_to = gl["connectToField"]
                as_field = gl["as"]
                for doc in current_docs:
                    network = []
                    start_val = doc.get(start_field)
                    for other in self.docs:
                        if other.get(connect_to) == start_val:
                            network.append(dict(other))
                    doc[as_field] = network
            if "$limit" in stage:
                current_docs = current_docs[: stage["$limit"]]
            if "$project" in stage:
                proj = stage["$project"]
                current_docs = [
                    {k: v for k, v in d.items() if proj.get(k) == 1}
                    for d in current_docs
                ]
        return AsyncCursorMock(current_docs)


class MockDBManager:
    def __init__(self):
        self.documents = AsyncCollectionMock()
        self.chunks = AsyncCollectionMock()
        self.edges = AsyncCollectionMock()
        self.entities = AsyncCollectionMock()
        self.communities = AsyncCollectionMock()


@pytest.mark.asyncio
async def test_lifecycle_archiving_and_restoration():
    db = MockDBManager()
    mgr = DocumentLifecycleManager(db, tenant_field="tenant_id")

    tenant_id = "tenant_test"
    doc_id = "doc_payment"

    # Seed initial active state
    await mgr.record_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        file_name="payment.md",
        content_hash="hash_v1",
        chunk_count=2,
        entity_count=2,
        edge_count=1,
        version=1,
        status="active",
    )

    db.chunks.docs.extend(
        [
            {
                "tenant_id": tenant_id,
                "document_id": doc_id,
                "chunk_id": "c1",
                "is_archived": False,
                "entities": ["PaymentGateway"],
            },
            {
                "tenant_id": tenant_id,
                "document_id": doc_id,
                "chunk_id": "c2",
                "is_archived": False,
                "entities": ["LedgerService"],
            },
        ]
    )
    db.edges.docs.append(
        {
            "tenant_id": tenant_id,
            "document_id": doc_id,
            "source": "PaymentGateway",
            "target": "LedgerService",
            "is_archived": False,
        }
    )
    db.entities.docs.extend(
        [
            {
                "tenant_id": tenant_id,
                "canonical_name": "PaymentGateway",
                "degree": 1,
                "mentions": 1,
            },
            {
                "tenant_id": tenant_id,
                "canonical_name": "LedgerService",
                "degree": 1,
                "mentions": 1,
            },
        ]
    )

    # 1. Archive document
    archived = await mgr.archive_document(tenant_id, doc_id, reason="superseded by v2")
    assert archived is True

    doc_record = await mgr.get_document(tenant_id, doc_id)
    assert doc_record["status"] == "archived"
    assert "archived_at" in doc_record

    # Verify chunks and edges are marked archived
    assert all(c["is_archived"] is True for c in db.chunks.docs)
    assert all(e["is_archived"] is True for e in db.edges.docs)

    # Verify active entity degrees were recalculated to 0
    ent_pg = await db.entities.find_one(
        {"tenant_id": tenant_id, "canonical_name": "PaymentGateway"}
    )
    assert ent_pg["degree"] == 0

    # 2. Check tenant stats
    stats = await mgr.get_tenant_stats(tenant_id)
    assert stats.total_documents == 1
    assert stats.archived_documents == 1
    assert stats.active_documents == 0
    assert stats.active_chunks == 0
    assert stats.active_edges == 0

    # 3. Restore document
    restored = await mgr.restore_document(tenant_id, doc_id)
    assert restored is True

    doc_record_restored = await mgr.get_document(tenant_id, doc_id)
    assert doc_record_restored["status"] == "active"
    assert "archived_at" not in doc_record_restored

    assert all(c["is_archived"] is False for c in db.chunks.docs)
    assert all(e["is_archived"] is False for e in db.edges.docs)

    ent_pg_restored = await db.entities.find_one(
        {"tenant_id": tenant_id, "canonical_name": "PaymentGateway"}
    )
    assert ent_pg_restored["degree"] == 1


@pytest.mark.asyncio
async def test_lifecycle_reingest_versioning():
    db = MockDBManager()
    mgr = DocumentLifecycleManager(db, tenant_field="tenant_id")
    tenant_id = "tenant_test"
    doc_id = "doc_architecture"

    # Ingest v1
    await mgr.record_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        file_name="arch.md",
        content_hash="hash_111",
        chunk_count=3,
        entity_count=4,
        edge_count=2,
        version=1,
    )

    doc_v1 = await mgr.get_document(tenant_id, doc_id)
    assert doc_v1["version"] == 1
    assert doc_v1["content_hash"] == "hash_111"

    # Re-ingest with updated content hash -> v2
    await mgr.record_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        file_name="arch.md",
        content_hash="hash_222",
        chunk_count=4,
        entity_count=5,
        edge_count=3,
        version=2,
    )

    doc_v2 = await mgr.get_document(tenant_id, doc_id)
    assert doc_v2["version"] == 2
    assert doc_v2["content_hash"] == "hash_222"


@pytest.mark.asyncio
async def test_lifecycle_list_documents():
    db = MockDBManager()
    mgr = DocumentLifecycleManager(db, tenant_field="tenant_id")
    tenant_id = "tenant_list"

    # Seed 3 documents
    for i in range(1, 4):
        status = "archived" if i == 3 else "active"
        await mgr.record_document(
            tenant_id=tenant_id,
            document_id=f"doc_{i}",
            file_name=f"file_{i}.md",
            content_hash=f"hash_{i}",
            chunk_count=2,
            entity_count=2,
            edge_count=1,
            version=1,
            status=status,
        )

    # 1. Filter by active
    active_docs = await mgr.list_documents(tenant_id, status="active")
    assert len(active_docs) == 2
    assert all(d.status == "active" for d in active_docs)

    # 2. Filter by archived
    archived_docs = await mgr.list_documents(tenant_id, status="archived")
    assert len(archived_docs) == 1
    assert archived_docs[0].document_id == "doc_3"

    # 3. Pagination with limit and skip on status="all"
    page_1 = await mgr.list_documents(tenant_id, status="all", limit=2, skip=0)
    assert len(page_1) == 2

    page_2 = await mgr.list_documents(tenant_id, status="all", limit=2, skip=2)
    assert len(page_2) == 1
    assert page_2[0].document_id not in [d.document_id for d in page_1]


@pytest.mark.asyncio
async def test_lifecycle_shared_edges():
    db = MockDBManager()
    mgr = DocumentLifecycleManager(db, tenant_field="tenant_id")
    tenant_id = "tenant_shared"
    doc_a = "doc_A"
    doc_b = "doc_B"

    # Document records
    await mgr.record_document(tenant_id, doc_a, "a.md", "h1", 1, 2, 1, version=1)
    await mgr.record_document(tenant_id, doc_b, "b.md", "h2", 1, 2, 1, version=1)

    # Chunks
    await db.chunks.insert_one(
        {
            "tenant_id": tenant_id,
            "document_id": doc_a,
            "chunk_id": "c_a",
            "entities": ["ServiceA", "ServiceB"],
            "is_archived": False,
        }
    )
    await db.chunks.insert_one(
        {
            "tenant_id": tenant_id,
            "document_id": doc_b,
            "chunk_id": "c_b",
            "entities": ["ServiceA", "ServiceB"],
            "is_archived": False,
        }
    )

    # Shared Edge corroborated by both doc_A and doc_B
    edge_id = "shared_edge_1"
    await db.edges.insert_one(
        {
            "_id": edge_id,
            "tenant_id": tenant_id,
            "source": "ServiceA",
            "relation": "CALLS",
            "target": "ServiceB",
            "weight": 2.0,
            "document_ids": [doc_a, doc_b],
            "is_archived": False,
        }
    )

    # Entities
    await db.entities.insert_one(
        {
            "tenant_id": tenant_id,
            "canonical_name": "ServiceA",
            "degree": 1,
            "mentions": 2,
        }
    )
    await db.entities.insert_one(
        {
            "tenant_id": tenant_id,
            "canonical_name": "ServiceB",
            "degree": 1,
            "mentions": 2,
        }
    )

    # Step 1: Delete doc_a -> Shared edge must NOT be deleted, doc_a must be pulled
    res_a = await mgr.delete_document(tenant_id, doc_a)
    assert res_a["edges_deleted"] == 0
    assert res_a["chunks_deleted"] == 1
    assert res_a["orphans_pruned"] == 0

    edge_after_a = await db.edges.find_one({"_id": edge_id})
    assert edge_after_a is not None
    assert edge_after_a["document_ids"] == [doc_b]
    assert edge_after_a["weight"] == 1.0

    # Step 2: Delete doc_b -> Edge is now exclusive to doc_b, so it should be deleted
    res_b = await mgr.delete_document(tenant_id, doc_b)
    assert res_b["edges_deleted"] == 1
    assert res_b["chunks_deleted"] == 1
    assert res_b["orphans_pruned"] == 2  # ServiceA and ServiceB pruned

    edge_after_b = await db.edges.find_one({"_id": edge_id})
    assert edge_after_b is None
    assert len(db.entities.docs) == 0


@pytest.mark.asyncio
async def test_chunk_level_diffing():
    rag = AtlasGraphRAG(mongo_uri="mongodb://localhost:27017", provider="mock")
    mock_db = MockDBManager()
    rag._db = mock_db
    rag._lifecycle = DocumentLifecycleManager(mock_db, tenant_field=rag._tenant_field)
    rag._chunker = HeadingAwareChunker(target_chunk_size=50)

    tenant_id = "tenant_diff"
    doc_id = "doc_diff"

    # Track how many texts were embedded
    embedded_texts: list[str] = []
    original_embed_batch = rag._embedder.embed_batch

    async def tracking_embed(texts):
        embedded_texts.extend(texts)
        return await original_embed_batch(texts)

    rag._embedder.embed_batch = tracking_embed

    doc_text_v1 = (
        "# Overview\n"
        "This is the stable unchanging section of the documentation.\n\n"
        "# Feature X\n"
        "Initial implementation details for Feature X."
    )

    # Initial ingest
    initial_chunks = rag._chunker.chunk_document(doc_text_v1, doc_id, tenant_id)
    await rag.ingest(tenant_id=tenant_id, document_id=doc_id, text=doc_text_v1)
    assert len(embedded_texts) == len(initial_chunks)
    embedded_texts.clear()

    # Re-ingest with Section 1 unchanged and Section 2 modified
    doc_text_v2 = (
        "# Overview\n"
        "This is the stable unchanging section of the documentation.\n\n"
        "# Feature X\n"
        "Updated and enhanced implementation details for Feature X."
    )
    reingest_chunks = rag._chunker.chunk_document(doc_text_v2, doc_id, tenant_id)

    await rag.ingest(tenant_id=tenant_id, document_id=doc_id, text=doc_text_v2)
    # Only the modified chunk should be sent to the embedder, strictly less than total
    assert 0 < len(embedded_texts) < len(reingest_chunks)
    assert any("Updated and enhanced" in t for t in embedded_texts)

    doc_meta = await rag.get_document(tenant_id, doc_id)
    assert doc_meta["version"] == 2


@pytest.mark.asyncio
async def test_entity_token_seed_fallback():
    mock_db = MockDBManager()
    tenant_id = "tenant_seed_fallback"

    # Seed an entity in db.entities
    await mock_db.entities.insert_one(
        {
            "tenant_id": tenant_id,
            "canonical_name": "ReconciliationWorker",
            "degree": 3,
            "mentions": 2,
        }
    )

    # Seed a chunk referencing this entity
    await mock_db.chunks.insert_one(
        {
            "tenant_id": tenant_id,
            "chunk_id": "chunk_rec_1",
            "text": "The ReconciliationWorker processes nocturnal settlement batches.",
            "entities": ["ReconciliationWorker"],
            "is_archived": False,
        }
    )

    engine = HybridRetrievalEngine(
        db_manager=mock_db,
        embedder=MockEmbedder(),
        domain_pack=GeneralDomainPack(),
        config=AtlasConfig(hub_degree_threshold=200),
        tenant_field="tenant_id",
    )

    query = "How does ReconciliationWorker operate in the system?"
    resp = await engine.search(query=query, tenant_id=tenant_id, top_k=5)

    assert resp is not None
    # Verify the chunk linked to ReconciliationWorker was discovered via entity token fallback
    assert any("ReconciliationWorker" in p.text for p in resp.passages)


@pytest.mark.asyncio
async def test_two_tier_tenancy_isolation():
    rag = AtlasGraphRAG(mongo_uri="mongodb://localhost:27017", provider="mock")
    mock_db = MockDBManager()
    rag._db = mock_db
    rag._lifecycle = DocumentLifecycleManager(mock_db, tenant_field=rag._tenant_field)
    rag._chunker = HeadingAwareChunker(target_chunk_size=100)
    rag._retrieval = HybridRetrievalEngine(
        db_manager=mock_db,
        embedder=rag._embedder,
        domain_pack=rag._domain,
        config=rag._config,
        reranker=rag._reranker,
        tenant_field=rag._tenant_field,
    )

    # Ingest Doc A (Org1, Proj A)
    await rag.ingest(
        tenant_id="org1",
        project_id="proj_a",
        document_id="doc_a",
        text="Alpha project confidential revenue report.",
    )

    # Ingest Doc B (Org1, Proj B)
    await rag.ingest(
        tenant_id="org1",
        project_id="proj_b",
        document_id="doc_b",
        text="Beta project confidential engineering specifications.",
    )

    # Ingest Doc C (Org2, Proj A)
    await rag.ingest(
        tenant_id="org2",
        project_id="proj_a",
        document_id="doc_c",
        text="External company confidential financial records.",
    )

    # 1. Scoped search for Org1 / Proj A -> Should match ONLY doc_a
    res_a = await rag.search(
        query="confidential",
        tenant_id="org1",
        project_id="proj_a",
        top_k=5,
    )
    assert res_a.tenant_id == "org1"
    assert res_a.project_id == "proj_a"
    assert len(res_a.passages) > 0
    assert any("Alpha" in p.text for p in res_a.passages)
    assert not any("Beta" in p.text for p in res_a.passages)
    assert not any("External" in p.text for p in res_a.passages)

    # 2. Scoped search for Org1 / Proj B -> Should match ONLY doc_b
    res_b = await rag.search(
        query="confidential",
        tenant_id="org1",
        project_id="proj_b",
        top_k=5,
    )
    assert res_b.tenant_id == "org1"
    assert res_b.project_id == "proj_b"
    assert len(res_b.passages) > 0
    assert any("Beta" in p.text for p in res_b.passages)
    assert not any("Alpha" in p.text for p in res_b.passages)
    assert not any("External" in p.text for p in res_b.passages)

    # 3. Cross-project search for Org1 (project_id=None) -> Matches doc_a AND doc_b, ZERO from Org2
    res_org1 = await rag.search(
        query="confidential",
        tenant_id="org1",
        project_id=None,
        top_k=5,
    )
    assert res_org1.tenant_id == "org1"
    assert res_org1.project_id is None
    assert any("Alpha" in p.text for p in res_org1.passages)
    assert any("Beta" in p.text for p in res_org1.passages)
    assert not any("External" in p.text for p in res_org1.passages)


@pytest.mark.asyncio
async def test_two_tier_lifecycle_scoping():
    db = MockDBManager()
    mgr = DocumentLifecycleManager(db, tenant_field="tenant_id")

    # Record documents across tenants and projects
    await mgr.record_document(
        tenant_id="org1",
        document_id="doc_1",
        project_id="p1",
        file_name="p1_doc1.md",
        content_hash="hash_p1_1",
        chunk_count=2,
        entity_count=2,
        edge_count=1,
        version=1,
        status="active",
    )
    await mgr.record_document(
        tenant_id="org1",
        document_id="doc_2",
        project_id="p2",
        file_name="p2_doc2.md",
        content_hash="hash_p2_2",
        chunk_count=3,
        entity_count=3,
        edge_count=2,
        version=1,
        status="active",
    )
    await mgr.record_document(
        tenant_id="org1",
        document_id="doc_3",
        project_id="p1",
        file_name="p1_doc3.md",
        content_hash="hash_p1_3",
        chunk_count=1,
        entity_count=1,
        edge_count=0,
        version=1,
        status="archived",
    )
    await mgr.record_document(
        tenant_id="org2",
        document_id="doc_4",
        project_id="p1",
        file_name="org2_doc4.md",
        content_hash="hash_org2_4",
        chunk_count=2,
        entity_count=2,
        edge_count=1,
        version=1,
        status="active",
    )

    # 1. List documents scoped by project
    docs_p1 = await mgr.list_documents("org1", project_id="p1", status="active")
    assert len(docs_p1) == 1
    assert docs_p1[0].document_id == "doc_1"
    assert docs_p1[0].project_id == "p1"

    docs_p1_all = await mgr.list_documents("org1", project_id="p1", status="all")
    assert len(docs_p1_all) == 2
    assert {d.document_id for d in docs_p1_all} == {"doc_1", "doc_3"}

    # 2. List documents cross-project (tenant-wide)
    docs_org1 = await mgr.list_documents("org1", project_id=None, status="active")
    assert len(docs_org1) == 2
    assert {d.document_id for d in docs_org1} == {"doc_1", "doc_2"}

    # 3. Stats scoped by project vs tenant-wide
    stats_p1 = await mgr.get_tenant_stats("org1", project_id="p1")
    assert stats_p1.project_id == "p1"
    assert stats_p1.total_documents == 2
    assert stats_p1.active_documents == 1
    assert stats_p1.archived_documents == 1

    stats_org1 = await mgr.get_tenant_stats("org1", project_id=None)
    assert stats_org1.project_id is None
    assert stats_org1.total_documents == 3
    assert stats_org1.active_documents == 2
    assert stats_org1.archived_documents == 1

    # 4. Scoped deletion
    del_res = await mgr.delete_document("org1", "doc_1", project_id="p1")
    assert del_res["documents_deleted"] == 1

    # Doc 1 is gone in p1
    doc_1_check = await mgr.get_document("org1", "doc_1", project_id="p1")
    assert doc_1_check is None

    # Doc 2 in p2 is untouched
    doc_2_check = await mgr.get_document("org1", "doc_2", project_id="p2")
    assert doc_2_check is not None

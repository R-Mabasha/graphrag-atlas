from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal

from graphrag_atlas.core.database import DatabaseManager
from graphrag_atlas.models import DocumentInfo, TenantStats


class DocumentLifecycleManager:
    """Manages document deduplication, content-hashing, cascading deletion, and orphan cleanup."""

    def __init__(
        self, db_manager: DatabaseManager, tenant_field: str = "tenant_id"
    ) -> None:
        self._db = db_manager
        self._tenant = tenant_field

    @staticmethod
    def hash_content(data: bytes) -> str:
        """Compute SHA-256 hex digest of file or text content."""
        return hashlib.sha256(data).hexdigest()

    async def get_document(
        self, tenant_id: str, document_id: str, project_id: str | None = None
    ) -> dict | None:
        """Retrieve existing document record."""
        query = {self._tenant: tenant_id, "document_id": document_id}
        if project_id is not None:
            query["project_id"] = project_id
        return await self._db.documents.find_one(query)

    async def list_documents(
        self,
        tenant_id: str,
        status: Literal["active", "archived", "all"] = "active",
        limit: int = 50,
        skip: int = 0,
        project_id: str | None = None,
    ) -> list[DocumentInfo]:
        """List documents for a tenant with optional project filtering, status filtering, and pagination."""
        query: dict[str, Any] = {self._tenant: tenant_id}
        if project_id is not None:
            query["project_id"] = project_id
        if status in ("active", "archived"):
            query["status"] = status

        cursor = (
            self._db.documents.find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs: list[DocumentInfo] = []
        async for doc in cursor:
            doc.pop("_id", None)
            docs.append(DocumentInfo(**doc))
        return docs

    async def record_document(
        self,
        tenant_id: str,
        document_id: str,
        file_name: str,
        content_hash: str,
        chunk_count: int,
        entity_count: int,
        edge_count: int,
        version: int = 1,
        status: str = "active",
        project_id: str | None = None,
    ) -> None:
        """Upsert document metadata."""
        now = datetime.now(timezone.utc).isoformat()
        filter_doc = {self._tenant: tenant_id, "document_id": document_id}
        if project_id is not None:
            filter_doc["project_id"] = project_id

        await self._db.documents.update_one(
            filter_doc,
            {
                "$set": {
                    self._tenant: tenant_id,
                    "project_id": project_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "content_hash": content_hash,
                    "chunk_count": chunk_count,
                    "entity_count": entity_count,
                    "edge_count": edge_count,
                    "version": version,
                    "status": status,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def delete_document(
        self, tenant_id: str, document_id: str, project_id: str | None = None
    ) -> dict[str, int]:
        """
        Cascading delete: Removes document metadata, chunks, and edges.
        Performs orphan cleanup on entities with zero remaining edges or mentions.
        """
        filter_doc = {self._tenant: tenant_id, "document_id": document_id}
        if project_id is not None:
            filter_doc["project_id"] = project_id

        # 1. Collect entity names mentioned in edges and chunks before deletion
        edge_filter = {
            self._tenant: tenant_id,
            "$or": [{"document_id": document_id}, {"document_ids": document_id}],
        }
        if project_id is not None:
            edge_filter["project_id"] = project_id
        cursor = self._db.edges.find(edge_filter)
        impacted_entities: set[str] = set()
        edges_to_delete: list[Any] = []
        edges_to_pull: list[Any] = []

        async for edge in cursor:
            impacted_entities.add(edge["source"])
            impacted_entities.add(edge["target"])
            doc_ids = edge.get("document_ids", [])
            if not doc_ids or doc_ids == [document_id]:
                edges_to_delete.append(edge.get("_id"))
            else:
                edges_to_pull.append(edge.get("_id"))

        chunk_cursor = self._db.chunks.find(filter_doc, {"entities": 1})
        async for chunk in chunk_cursor:
            for ent in chunk.get("entities", []):
                impacted_entities.add(ent)

        # 2. Delete chunks
        chunk_res = await self._db.chunks.delete_many(filter_doc)

        # 3. Delete exclusive edges or pull document provenance from shared edges
        edges_deleted_count = 0
        if edges_to_delete:
            edge_res = await self._db.edges.delete_many(
                {"_id": {"$in": edges_to_delete}}
            )
            edges_deleted_count = edge_res.deleted_count

        if edges_to_pull:
            await self._db.edges.update_many(
                {"_id": {"$in": edges_to_pull}},
                {"$pull": {"document_ids": document_id}, "$inc": {"weight": -1.0}},
            )

        # 4. Delete document record
        doc_res = await self._db.documents.delete_one(filter_doc)

        # 5. Re-evaluate impacted entities (orphan cleanup)
        orphans_pruned = 0
        for entity_name in impacted_entities:
            # Count remaining edges and chunks for this entity
            edge_count = await self._db.edges.count_documents(
                {
                    self._tenant: tenant_id,
                    "$or": [{"source": entity_name}, {"target": entity_name}],
                }
            )
            chunk_count = await self._db.chunks.count_documents(
                {self._tenant: tenant_id, "entities": entity_name}
            )
            if edge_count == 0 and chunk_count == 0:
                await self._db.entities.delete_one(
                    {self._tenant: tenant_id, "canonical_name": entity_name}
                )
                orphans_pruned += 1
            else:
                # Update accurate degree and mentions count
                await self._db.entities.update_one(
                    {self._tenant: tenant_id, "canonical_name": entity_name},
                    {"$set": {"degree": edge_count, "mentions": chunk_count}},
                )

        return {
            "chunks_deleted": chunk_res.deleted_count,
            "edges_deleted": edges_deleted_count,
            "documents_deleted": doc_res.deleted_count,
            "orphans_pruned": orphans_pruned,
        }

    async def archive_document(
        self,
        tenant_id: str,
        document_id: str,
        reason: str = "",
        project_id: str | None = None,
    ) -> bool:
        """
        Soft-delete / Archive a document.
        Marks document, chunks, and edges as archived, and recalibrates active entity degrees
        so archived data is omitted from standard retrieval and hub calculations.
        """
        filter_doc = {self._tenant: tenant_id, "document_id": document_id}
        if project_id is not None:
            filter_doc["project_id"] = project_id
        doc = await self._db.documents.find_one(filter_doc)
        if not doc:
            return False

        now = datetime.now(timezone.utc).isoformat()

        # 1. Update document record
        await self._db.documents.update_one(
            filter_doc,
            {
                "$set": {
                    "status": "archived",
                    "archived_at": now,
                    "archive_reason": reason,
                }
            },
        )

        # 2. Mark chunks and edges as archived
        await self._db.chunks.update_many(filter_doc, {"$set": {"is_archived": True}})
        await self._db.edges.update_many(filter_doc, {"$set": {"is_archived": True}})

        # 3. Recalculate degrees for impacted entities considering only active edges
        cursor = self._db.edges.find(filter_doc, {"source": 1, "target": 1})
        impacted: set[str] = set()
        async for e in cursor:
            impacted.add(e["source"])
            impacted.add(e["target"])

        for ent in impacted:
            edge_match: dict[str, Any] = {
                self._tenant: tenant_id,
                "is_archived": {"$ne": True},
                "$or": [{"source": ent}, {"target": ent}],
            }
            chunk_match: dict[str, Any] = {
                self._tenant: tenant_id,
                "is_archived": {"$ne": True},
                "entities": ent,
            }
            ent_match: dict[str, Any] = {
                self._tenant: tenant_id,
                "canonical_name": ent,
            }
            if project_id is not None:
                edge_match["project_id"] = project_id
                chunk_match["project_id"] = project_id
                ent_match["project_id"] = project_id

            active_edge_count = await self._db.edges.count_documents(edge_match)
            active_chunk_count = await self._db.chunks.count_documents(chunk_match)
            await self._db.entities.update_one(
                ent_match,
                {"$set": {"degree": active_edge_count, "mentions": active_chunk_count}},
            )

        return True

    async def restore_document(
        self, tenant_id: str, document_id: str, project_id: str | None = None
    ) -> bool:
        """
        Restore a previously archived document back to active retrieval state.
        Re-activates chunks, edges, and restores entity degrees.
        """
        filter_doc = {self._tenant: tenant_id, "document_id": document_id}
        if project_id is not None:
            filter_doc["project_id"] = project_id
        doc = await self._db.documents.find_one(filter_doc)
        if not doc or doc.get("status") != "archived":
            return False

        now = datetime.now(timezone.utc).isoformat()

        # 1. Update document record
        await self._db.documents.update_one(
            filter_doc,
            {
                "$set": {"status": "active", "updated_at": now},
                "$unset": {"archived_at": "", "archive_reason": ""},
            },
        )

        # 2. Un-archive chunks and edges
        await self._db.chunks.update_many(filter_doc, {"$set": {"is_archived": False}})
        await self._db.edges.update_many(filter_doc, {"$set": {"is_archived": False}})

        # 3. Recalculate degrees for impacted entities
        cursor = self._db.edges.find(filter_doc, {"source": 1, "target": 1})
        impacted: set[str] = set()
        async for e in cursor:
            impacted.add(e["source"])
            impacted.add(e["target"])

        for ent in impacted:
            edge_match = {
                self._tenant: tenant_id,
                "is_archived": {"$ne": True},
                "$or": [{"source": ent}, {"target": ent}],
            }
            chunk_match = {
                self._tenant: tenant_id,
                "is_archived": {"$ne": True},
                "entities": ent,
            }
            ent_match = {
                self._tenant: tenant_id,
                "canonical_name": ent,
            }
            if project_id is not None:
                edge_match["project_id"] = project_id
                chunk_match["project_id"] = project_id
                ent_match["project_id"] = project_id

            active_edge_count = await self._db.edges.count_documents(edge_match)
            active_chunk_count = await self._db.chunks.count_documents(chunk_match)
            await self._db.entities.update_one(
                ent_match,
                {"$set": {"degree": active_edge_count, "mentions": active_chunk_count}},
            )

        return True

    async def get_tenant_stats(
        self,
        tenant_id: str,
        hub_threshold: int = 200,
        project_id: str | None = None,
    ) -> TenantStats:
        """Compute full storage, document status, and graph diagnostics for a tenant or project."""
        tenant_match: dict[str, Any] = {self._tenant: tenant_id}
        if project_id is not None:
            tenant_match["project_id"] = project_id

        total_docs = await self._db.documents.count_documents(tenant_match)
        archived_match = dict(tenant_match)
        archived_match["status"] = "archived"
        archived_docs = await self._db.documents.count_documents(archived_match)
        active_docs = total_docs - archived_docs

        total_chunks = await self._db.chunks.count_documents(tenant_match)
        active_chunks_match = dict(tenant_match)
        active_chunks_match["is_archived"] = {"$ne": True}
        active_chunks = await self._db.chunks.count_documents(active_chunks_match)

        total_edges = await self._db.edges.count_documents(tenant_match)
        active_edges_match = dict(tenant_match)
        active_edges_match["is_archived"] = {"$ne": True}
        active_edges = await self._db.edges.count_documents(active_edges_match)

        total_entities = await self._db.entities.count_documents(tenant_match)

        # Find hub nodes
        hub_match = dict(tenant_match)
        hub_match["degree"] = {"$gt": hub_threshold}
        hub_cursor = self._db.entities.find(hub_match, {"canonical_name": 1})
        hub_entities = [doc["canonical_name"] async for doc in hub_cursor]

        return TenantStats(
            tenant_id=tenant_id,
            project_id=project_id,
            total_documents=total_docs,
            active_documents=active_docs,
            archived_documents=archived_docs,
            total_chunks=total_chunks,
            active_chunks=active_chunks,
            total_edges=total_edges,
            active_edges=active_edges,
            total_entities=total_entities,
            hub_entities=hub_entities,
        )

    async def vacuum(
        self, tenant_id: str, project_id: str | None = None
    ) -> dict[str, int]:
        """
        Integrity repair & graph vacuuming:
        1. Prunes dangling edges whose parent document no longer exists.
        2. Recalculates exact degrees and mentions for all entities.
        3. Prunes orphan entities with 0 active mentions and 0 active edges.
        """
        scope_match: dict[str, Any] = {self._tenant: tenant_id}
        if project_id is not None:
            scope_match["project_id"] = project_id

        # 1. Fetch valid document IDs
        valid_docs = set()
        async for doc in self._db.documents.find(scope_match, {"document_id": 1}):
            valid_docs.add(doc["document_id"])

        # 2. Prune dangling edges
        del_edge_match = dict(scope_match)
        del_edge_match["document_id"] = {"$nin": list(valid_docs)}
        dangling_edges_res = await self._db.edges.delete_many(del_edge_match)

        # 3. Prune dangling chunks
        del_chunk_match = dict(scope_match)
        del_chunk_match["document_id"] = {"$nin": list(valid_docs)}
        dangling_chunks_res = await self._db.chunks.delete_many(del_chunk_match)

        # 4. Repair entity degrees and mentions
        recalibrated_entities = 0
        orphans_pruned = 0

        async for ent_doc in self._db.entities.find(scope_match):
            name = ent_doc["canonical_name"]
            edge_match = {
                self._tenant: tenant_id,
                "is_archived": {"$ne": True},
                "$or": [{"source": name}, {"target": name}],
            }
            chunk_match = {
                self._tenant: tenant_id,
                "is_archived": {"$ne": True},
                "entities": name,
            }
            ent_query = {self._tenant: tenant_id, "canonical_name": name}
            if project_id is not None:
                edge_match["project_id"] = project_id
                chunk_match["project_id"] = project_id
                ent_query["project_id"] = project_id

            edge_count = await self._db.edges.count_documents(edge_match)
            chunk_count = await self._db.chunks.count_documents(chunk_match)

            if edge_count == 0 and chunk_count == 0:
                await self._db.entities.delete_one(ent_query)
                orphans_pruned += 1
            else:
                await self._db.entities.update_one(
                    ent_query,
                    {"$set": {"degree": edge_count, "mentions": chunk_count}},
                )
                recalibrated_entities += 1

        return {
            "dangling_edges_pruned": dangling_edges_res.deleted_count,
            "dangling_chunks_pruned": dangling_chunks_res.deleted_count,
            "orphans_pruned": orphans_pruned,
            "entities_recalibrated": recalibrated_entities,
        }

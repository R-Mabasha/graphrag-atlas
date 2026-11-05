from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

from graphrag_atlas.core.database import DatabaseManager


class IndexManager:
    """Manages B-Tree, Multikey, and compound tenant index creation on MongoDB Atlas collections."""

    def __init__(
        self, db_manager: DatabaseManager, tenant_field: str = "tenant_id"
    ) -> None:
        self._db = db_manager
        self._tenant = tenant_field

    async def ensure_indexes(self) -> dict[str, list[str]]:
        """Create all required compound indexes for multi-tenant hybrid graph search."""
        results: dict[str, list[str]] = {}

        # 1. Chunks Collection Indexes
        chunk_indexes = [
            IndexModel(
                [(self._tenant, ASCENDING), ("document_id", ASCENDING)],
                name="tenant_doc_idx",
            ),
            IndexModel(
                [
                    (self._tenant, ASCENDING),
                    ("document_id", ASCENDING),
                    ("content_hash", ASCENDING),
                ],
                name="tenant_doc_chunk_hash_idx",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("entities", ASCENDING)],
                name="tenant_entities_multikey_idx",
            ),
            IndexModel(
                [
                    (self._tenant, ASCENDING),
                    ("is_archived", ASCENDING),
                    ("entities", ASCENDING),
                ],
                name="tenant_active_entities_multikey_idx",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("chunk_index", ASCENDING)],
                name="tenant_chunk_idx",
            ),
        ]
        created_chunks = await self._db.chunks.create_indexes(chunk_indexes)
        results["chunks"] = created_chunks

        # 2. Edges Collection Indexes
        edge_indexes = [
            IndexModel(
                [(self._tenant, ASCENDING), ("source", ASCENDING)],
                name="tenant_source_idx",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("target", ASCENDING)],
                name="tenant_target_idx",
            ),
            IndexModel(
                [
                    (self._tenant, ASCENDING),
                    ("is_archived", ASCENDING),
                    ("source", ASCENDING),
                ],
                name="tenant_active_source_idx",
            ),
            IndexModel(
                [
                    (self._tenant, ASCENDING),
                    ("is_archived", ASCENDING),
                    ("target", ASCENDING),
                ],
                name="tenant_active_target_idx",
            ),
            IndexModel(
                [
                    (self._tenant, ASCENDING),
                    ("source", ASCENDING),
                    ("relation", ASCENDING),
                    ("target", ASCENDING),
                ],
                unique=True,
                name="tenant_edge_unique",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("document_ids", ASCENDING)],
                name="tenant_doc_ids_edge_idx",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("document_id", ASCENDING)],
                name="tenant_doc_edge_idx",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("weight", DESCENDING)],
                name="tenant_weight_idx",
            ),
        ]
        created_edges = await self._db.edges.create_indexes(edge_indexes)
        results["edges"] = created_edges

        # 3. Entities Collection Indexes
        entity_indexes = [
            IndexModel(
                [(self._tenant, ASCENDING), ("canonical_name", ASCENDING)],
                unique=True,
                name="tenant_canonical_unique",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("degree", DESCENDING)],
                name="tenant_degree_idx",
            ),
        ]
        created_entities = await self._db.entities.create_indexes(entity_indexes)
        results["entities"] = created_entities

        # 4. Documents Collection Indexes
        doc_indexes = [
            IndexModel(
                [(self._tenant, ASCENDING), ("document_id", ASCENDING)],
                unique=True,
                name="tenant_doc_unique",
            ),
            IndexModel(
                [(self._tenant, ASCENDING), ("content_hash", ASCENDING)],
                name="tenant_content_hash_idx",
            ),
        ]
        created_docs = await self._db.documents.create_indexes(doc_indexes)
        results["documents"] = created_docs

        return results

    @staticmethod
    def get_atlas_search_index_spec(
        index_name: str = "default_search",
    ) -> dict[str, Any]:
        """
        Returns JSON specification for Atlas Search (BM25) index with multi-analyzer support.
        Deploy via Atlas UI or mongocli under the chunks collection.
        """
        return {
            "name": index_name,
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "text": {"type": "string", "analyzer": "lucene.standard"},
                        "tenant_id": {"type": "stringFacet"},
                    },
                }
            },
        }

    @staticmethod
    def get_atlas_vector_index_spec(
        dimension: int = 1536,
        index_name: str = "default_vector",
        path: str = "embedding",
        similarity: str = "cosine",
    ) -> dict[str, Any]:
        """
        Returns JSON specification for Atlas Vector Search ($vectorSearch) index.
        Deploy via Atlas UI or mongocli under the chunks collection.
        """
        return {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": path,
                        "numDimensions": dimension,
                        "similarity": similarity,
                    },
                    {"type": "filter", "path": "tenant_id"},
                ]
            },
        }

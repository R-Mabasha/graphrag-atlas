from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Triple(BaseModel):
    """Represents a knowledge graph relationship edge."""

    source: str = Field(..., description="Source entity name")
    relation: str = Field(..., description="Relationship type (e.g. DEPENDS_ON, CALLS)")
    target: str = Field(..., description="Target entity name")
    weight: float = Field(
        default=1.0, description="Edge confidence or co-occurrence weight"
    )
    is_archived: bool = Field(
        default=False, description="Whether this edge belongs to an archived document"
    )
    document_ids: list[str] = Field(
        default_factory=list,
        description="IDs of documents corroborating this relationship",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary provenance metadata"
    )


class Entity(BaseModel):
    """Represents a discovered entity node in the knowledge graph."""

    name: str = Field(..., description="Original extracted entity mention")
    canonical_name: str = Field(..., description="Resolved canonical identity")
    entity_type: str = Field(default="CONCEPT", description="Entity category or class")
    degree: int = Field(default=0, description="Total in-degree and out-degree count")
    mentions: int = Field(
        default=1, description="Number of text passages mentioning this entity"
    )


class Chunk(BaseModel):
    """Represents a passage chunk extracted from a source document."""

    chunk_id: str = Field(
        ..., description="Unique chunk identifier (e.g. doc_id:chunk_idx)"
    )
    document_id: str = Field(..., description="Parent document identifier")
    tenant_id: str = Field(..., description="Tenant partition identifier")
    project_id: str | None = Field(
        default=None, description="Optional sub-partition project identifier"
    )
    text: str = Field(..., description="Raw text content of the chunk")
    heading_breadcrumbs: list[str] = Field(
        default_factory=list, description="Section heading hierarchy"
    )
    entities: list[str] = Field(
        default_factory=list, description="Extracted entity names in this chunk"
    )
    embedding: list[float] | None = Field(
        default=None, description="Dense vector embedding"
    )
    chunk_index: int = Field(
        default=0, description="0-based sequence index in document"
    )
    is_archived: bool = Field(
        default=False, description="Whether this chunk belongs to an archived document"
    )
    content_hash: str | None = Field(
        default=None, description="SHA-256 hash of chunk content for granular diffing"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="File and layout attributes"
    )


class Passage(BaseModel):
    """Represents a scored passage returned in retrieval results."""

    chunk_id: str = Field(..., description="Source chunk ID")
    document_id: str = Field(..., description="Source document ID")
    text: str = Field(..., description="Passage text")
    score: float = Field(..., description="Relevance score (RRF or reranker score)")
    source_type: Literal["bm25", "vector", "graph", "hybrid"] = Field(
        default="hybrid", description="Primary discovery channel"
    )
    heading_breadcrumbs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Community(BaseModel):
    """Represents a Leiden-clustered graph community / subsystem."""

    community_id: str = Field(..., description="Unique community ID")
    tenant_id: str = Field(..., description="Tenant identifier")
    project_id: str | None = Field(
        default=None, description="Optional sub-partition project identifier"
    )
    title: str = Field(..., description="High-level topic or subsystem title")
    summary: str = Field(..., description="LLM synthesis of community responsibilities")
    entities: list[str] = Field(default_factory=list, description="Member entities")
    level: int = Field(default=0, description="Hierarchy depth level")


class SearchResponse(BaseModel):
    """Structured response object returned by AtlasGraphRAG.search()."""

    query: str
    tenant_id: str
    project_id: str | None = Field(
        default=None, description="Optional sub-partition project identifier"
    )
    passages: list[Passage] = Field(default_factory=list)
    graph_entities: list[str] = Field(default_factory=list)
    graph_edges: list[Triple] = Field(default_factory=list)
    community_summaries: list[str] = Field(default_factory=list)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """Metadata tracking document state and lifecycle."""

    document_id: str
    tenant_id: str
    project_id: str | None = Field(
        default=None, description="Optional sub-partition project identifier"
    )
    file_name: str
    content_hash: str
    chunk_count: int
    entity_count: int
    edge_count: int
    version: int = Field(default=1, description="Sequential document version")
    status: Literal["active", "archived"] = Field(
        default="active", description="Lifecycle state of the document"
    )
    archived_at: str | None = Field(
        default=None, description="ISO timestamp of archiving"
    )
    created_at: str
    updated_at: str | None = None


class TenantStats(BaseModel):
    """Comprehensive storage and lifecycle diagnostics for a tenant partition."""

    tenant_id: str
    project_id: str | None = Field(
        default=None, description="Optional sub-partition project identifier"
    )
    total_documents: int
    active_documents: int
    archived_documents: int
    total_chunks: int
    active_chunks: int
    total_edges: int
    active_edges: int
    total_entities: int
    hub_entities: list[str] = Field(default_factory=list)

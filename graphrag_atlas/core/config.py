from __future__ import annotations

from pydantic import BaseModel, Field


class AtlasConfig(BaseModel):
    """Configuration settings for AtlasGraphRAG storage and retrieval."""

    # Collection names
    chunks_collection: str = Field(
        default="atlas_chunks", description="Passage chunks collection"
    )
    edges_collection: str = Field(
        default="atlas_edges", description="Knowledge graph edges collection"
    )
    entities_collection: str = Field(
        default="atlas_entities", description="Knowledge graph entities collection"
    )
    documents_collection: str = Field(
        default="atlas_documents", description="Document metadata collection"
    )
    communities_collection: str = Field(
        default="atlas_communities", description="Leiden communities collection"
    )

    # Atlas Search & Vector Index names
    search_index_name: str = Field(
        default="default_search", description="Atlas Search (BM25) index name"
    )
    vector_index_name: str = Field(
        default="default_vector", description="Atlas Vector Search index name"
    )

    # Hub Guard circuit breaker
    hub_degree_threshold: int = Field(
        default=200,
        description="Nodes with degree higher than this are treated as super-nodes and not expanded",
    )

    # Retrieval parameters
    max_graph_depth: int = Field(
        default=2, description="Maximum traversal hops in $graphLookup"
    )
    max_graph_seeds: int = Field(
        default=25, description="Maximum seed entities from query"
    )
    max_branching_per_hop: int = Field(
        default=20, description="Maximum edges to follow per node direction"
    )
    rrf_k: int = Field(
        default=60, description="Reciprocal Rank Fusion constant (standard k=60)"
    )

    # Ingestion concurrency
    ingest_concurrency: int = Field(
        default=8, description="Maximum concurrent chunk embeddings during ingestion"
    )

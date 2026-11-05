# `graphrag-atlas`

[![PyPI version](https://img.shields.io/pypi/v/graphrag-atlas.svg)](https://pypi.org/project/graphrag-atlas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Production-grade Hybrid GraphRAG in a single MongoDB Atlas cluster.**

`graphrag-atlas` unifies BM25 lexical search, dense vector embeddings, dynamic document hierarchies, and bounded 2-hop knowledge graph expansion using native MongoDB Atlas features (`$search`, `$vectorSearch`, `$graphLookup`).

---

## Why `graphrag-atlas`?

Most GraphRAG implementations force difficult trade-offs:
1. **Multi-Database Sprawl:** Running separate vector databases (Qdrant, Pinecone), graph databases (Neo4j), and relational metadata stores.
2. **Offline Batch Only:** Generating massive offline parquets without interactive updates or multi-tenant query isolation.
3. **Naive Traversal:** Using basic `$graphLookup` without hub node circuit breakers or multi-modal fusion.

`graphrag-atlas` solves this by delivering an **end-to-end production retrieval pipeline in a single MongoDB Atlas database**:

```
User Query
    │
    ├── Concurrent Search ────────────────────────┐
    │   ├── Atlas BM25 (Lucene Analyzers)         │
    │   └── Atlas Vector Search (HNSW)            │
    │                                             ▼
    ├── Reciprocal Rank Fusion (RRF, k=60) ◄──────┘
    │
    ├── Graph Candidate Seed Selection
    │   └── Bounded 2-Hop $graphLookup (Tenant Filtered)
    │       └── Hub Guard Circuit Breaker (degree > 200)
    │
    ├── Cross-Encoder Reranking (Fair evaluation without bypass)
    │
    └── Structured Context Output (Passages + Graph Entities + Communities)
```

---

## Feature Comparison

| Capability | `langchain-mongodb` (`MongoDBGraphStore`) | `Cognee` | `graphrag-atlas` |
|---|---|---|---|
| **Storage Backend** | MongoDB | Pluggable multi-store | **Single MongoDB Atlas Cluster** |
| **Search Modalities** | Vector + Flat Graph | Vector + Graph | **BM25 + Vector + 2-hop Graph** |
| **Fusion Algorithm** | Naive list concatenation | LLM stitching | **Reciprocal Rank Fusion (RRF)** |
| **Super-node Protection** | None (risks exponential fan-out) | None | **Hub Guard Circuit Breaker (`degree > 200`)** |
| **Document Structure** | Flat text | Flat text | **AST Section Hierarchy + Semantic Triples** |
| **Community Detection** | None | Hierarchical | **Dynamic Leiden Clustering with Atomic Versioning** |
| **Lifecycle & Pruning** | Manual | Manual | **Content-hashed deduplication, cascading deletes & orphan cleanup** |
| **Multi-Tenancy** | Application-level | Application-level | **Compound index & aggregation root enforcement** |

---

## Installation

```bash
# Minimal installation (core engine)
pip install graphrag-atlas

# With OpenAI support
pip install "graphrag-atlas[openai]"

# With AWS Bedrock support
pip install "graphrag-atlas[aws]"

# With all providers and document parsers
pip install "graphrag-atlas[all]"
```

---

## Quickstart

```python
import asyncio
from graphrag_atlas import AtlasGraphRAG, GeneralDomainPack

async def main():
    # 1. Initialize RAG with unified provider resolver
    # Automatically reads OPENAI_API_KEY from environment
    rag = AtlasGraphRAG(
        mongo_uri="mongodb+srv://cluster.mongodb.net/",
        database="knowledge_base",
        tenant_field="tenant_id",
        provider="openai",  # or "aws"
        domain_pack=GeneralDomainPack()
    )

    # 2. Deploy Atlas indexes (B-Tree, Multikey, Compound Tenant)
    await rag.setup_indexes()

    # 3. Ingest documents in-process with automated AST and entity extraction
    doc_id = await rag.ingest(
        file_path="./system_architecture.pdf",
        tenant_id="client_alpha"
    )
    print(f"Ingested document: {doc_id}")

    # 4. Hybrid Search across BM25 + Vector + Graph
    response = await rag.search(
        query="Which services communicate with the payment processing pipeline?",
        tenant_id="client_alpha",
        top_k=5
    )

    for passage in response.passages:
        print(f"[{passage.score:.3f}] {passage.text}\n")

    print("Discovered Graph Entities:", response.graph_entities)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Domain Packs

Domain logic is completely decoupled from the retrieval engine. You can use the built-in `GeneralDomainPack` (zero-shot LLM extraction) or define custom domain rules:

```python
from graphrag_atlas.domain import DomainPack, RelationSpec

finance_pack = DomainPack(
    name="finance",
    ontology=[
        RelationSpec("SETTLES_VIA", direction="directed"),
        RelationSpec("POSTS_TO", direction="directed"),
        RelationSpec("RECONCILES_WITH", direction="undirected"),
    ],
    acronyms={"GL": "General Ledger", "AP": "Accounts Payable"},
    domain_vocabulary={"settlement", "clearing", "ledger", "journal"}
)
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

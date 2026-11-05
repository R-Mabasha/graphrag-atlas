from graphrag_atlas.extractor.ast_graph import ASTGraphBuilder


def test_ast_graph_structural_edges():
    sample_text = """# Core System
Overview of the core.

## Database Layer
Information about database schema.

### Partitioning Strategy
Details on sharding and hash distribution.

## API Gateway
Routing endpoints.
"""
    builder = ASTGraphBuilder()
    edges = builder.extract_structural_edges(sample_text, document_id="doc_sys")

    # Expected edges:
    # Core System -> CONTAINS_SECTION -> Database Layer
    # Database Layer -> CONTAINS_SECTION -> Partitioning Strategy
    # Core System -> CONTAINS_SECTION -> API Gateway
    triples_map = {(e.source, e.relation, e.target) for e in edges}

    assert ("Core System", "CONTAINS_SECTION", "Database Layer") in triples_map
    assert (
        "Database Layer",
        "CONTAINS_SECTION",
        "Partitioning Strategy",
    ) in triples_map
    assert ("Core System", "CONTAINS_SECTION", "API Gateway") in triples_map
    assert ("Database Layer", "CONTAINS_SECTION", "API Gateway") not in triples_map

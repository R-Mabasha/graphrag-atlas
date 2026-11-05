from graphrag_atlas.extractor.chunker import HeadingAwareChunker


def test_chunker_breadcrumbs_preservation():
    sample_text = """# Architecture Overview
This is the root architecture introduction.

## Payment Subsystem
Details about payment processing pipeline.

### Settlement Engine
The settlement engine handles ledger reconciliation and transaction journals.
"""
    chunker = HeadingAwareChunker(target_chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_document(
        sample_text, document_id="doc_arch", tenant_id="tenant_1"
    )

    assert len(chunks) >= 2
    # Verify that the last chunk preserves nested breadcrumbs
    last_chunk = chunks[-1]
    assert "Architecture Overview" in last_chunk.heading_breadcrumbs
    assert "Payment Subsystem" in last_chunk.heading_breadcrumbs
    assert "Settlement Engine" in last_chunk.heading_breadcrumbs

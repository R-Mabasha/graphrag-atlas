from graphrag_atlas.engine.rrf import reciprocal_rank_fusion


def test_rrf_scoring_order():
    # List 1: docA (rank 0), docB (rank 1)
    # List 2: docB (rank 0), docC (rank 1)
    list1 = ["docA", "docB"]
    list2 = ["docB", "docC"]

    # docB appears in both at rank 1 and rank 0
    fused = reciprocal_rank_fusion([list1, list2], k=60)

    # docB must have highest RRF score because it appears near top of both lists
    top_doc, _top_score = fused[0]
    assert top_doc == "docB"

    # docA and docC each appear once
    remaining_docs = [doc for doc, _ in fused[1:]]
    assert "docA" in remaining_docs
    assert "docC" in remaining_docs


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

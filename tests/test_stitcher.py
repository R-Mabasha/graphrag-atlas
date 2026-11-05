from graphrag_atlas.engine.stitcher import PassageStitcher
from graphrag_atlas.models import Passage


def test_passage_stitcher():
    p1 = Passage(
        chunk_id="doc1:0",
        document_id="doc1",
        text="First sentence of paragraph.",
        score=0.8,
        source_type="vector",
        metadata={"chunk_index": 0},
    )
    p2 = Passage(
        chunk_id="doc1:1",
        document_id="doc1",
        text="Second sentence of paragraph.",
        score=0.7,
        source_type="vector",
        metadata={"chunk_index": 1},
    )
    p3 = Passage(
        chunk_id="doc2:0",
        document_id="doc2",
        text="Unrelated document passage.",
        score=0.6,
        source_type="bm25",
        metadata={"chunk_index": 0},
    )

    stitched = PassageStitcher.stitch([p1, p2, p3])

    # p1 and p2 should be merged because chunk_index is sequential (0 then 1) in doc1
    assert len(stitched) == 2
    assert (
        "First sentence of paragraph.\n\nSecond sentence of paragraph."
        in stitched[0].text
    )
    assert stitched[0].score == 0.8
    assert stitched[1].document_id == "doc2"

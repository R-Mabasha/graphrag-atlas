from __future__ import annotations

from graphrag_atlas.models import Passage


class PassageStitcher:
    """Stitches adjacent sequential chunks from the same document into unified passages."""

    @staticmethod
    def stitch(passages: list[Passage]) -> list[Passage]:
        """Merge passages from same document whose chunk indices are sequential."""
        if len(passages) <= 1:
            return passages

        # Group by document_id
        stitched: list[Passage] = []
        visited: set[str] = set()

        for i, current in enumerate(passages):
            if current.chunk_id in visited:
                continue

            current_doc = current.document_id
            current_idx = current.metadata.get("chunk_index", -99)

            # Check if any other unvisited passage is immediately adjacent (current_idx + 1)
            merged_text = current.text
            merged_chunk_id = current.chunk_id
            best_score = current.score
            visited.add(current.chunk_id)

            for j, candidate in enumerate(passages):
                if candidate.chunk_id in visited:
                    continue
                cand_doc = candidate.document_id
                cand_idx = candidate.metadata.get("chunk_index", -99)

                if cand_doc == current_doc and cand_idx == current_idx + 1:
                    merged_text += "\n\n" + candidate.text
                    merged_chunk_id += f"+{candidate.chunk_id}"
                    best_score = max(best_score, candidate.score)
                    visited.add(candidate.chunk_id)
                    current_idx = cand_idx

            stitched.append(
                Passage(
                    chunk_id=merged_chunk_id,
                    document_id=current_doc,
                    text=merged_text,
                    score=best_score,
                    source_type=current.source_type,
                    heading_breadcrumbs=current.heading_breadcrumbs,
                    metadata=current.metadata,
                )
            )

        return stitched

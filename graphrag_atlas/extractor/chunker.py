from __future__ import annotations

import hashlib
import re

from graphrag_atlas.models import Chunk


class HeadingAwareChunker:
    """
    Splits text into chunks preserving markdown heading breadcrumbs and structural hierarchy.
    """

    def __init__(self, target_chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._target_size = target_chunk_size
        self._overlap = chunk_overlap
        self._heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk_document(
        self,
        text: str,
        document_id: str,
        tenant_id: str,
        project_id: str | None = None,
    ) -> list[Chunk]:
        """Split document text into chunks with breadcrumb metadata."""
        lines = text.splitlines()
        chunks: list[Chunk] = []

        current_breadcrumbs: list[str] = []
        current_buffer: list[str] = []
        current_length = 0
        chunk_idx = 0

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Adjust breadcrumbs depth
                if level <= len(current_breadcrumbs):
                    current_breadcrumbs = current_breadcrumbs[: level - 1]
                current_breadcrumbs.append(title)

            line_len = len(line) + 1
            if current_length + line_len > self._target_size and current_buffer:
                chunk_text = "\n".join(current_buffer).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document_id}:{chunk_idx}",
                            document_id=document_id,
                            tenant_id=tenant_id,
                            project_id=project_id,
                            text=chunk_text,
                            content_hash=hashlib.sha256(
                                chunk_text.encode("utf-8")
                            ).hexdigest(),
                            heading_breadcrumbs=list(current_breadcrumbs),
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1

                # Keep overlap from previous buffer
                overlap_lines: list[str] = []
                overlap_len = 0
                for prev_line in reversed(current_buffer):
                    if overlap_len + len(prev_line) > self._overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_len += len(prev_line) + 1

                current_buffer = list(overlap_lines)
                current_length = overlap_len

            current_buffer.append(line)
            current_length += line_len

        # Flush final buffer
        if current_buffer:
            chunk_text = "\n".join(current_buffer).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}:{chunk_idx}",
                        document_id=document_id,
                        tenant_id=tenant_id,
                        project_id=project_id,
                        text=chunk_text,
                        content_hash=hashlib.sha256(
                            chunk_text.encode("utf-8")
                        ).hexdigest(),
                        heading_breadcrumbs=list(current_breadcrumbs),
                        chunk_index=chunk_idx,
                    )
                )

        return chunks

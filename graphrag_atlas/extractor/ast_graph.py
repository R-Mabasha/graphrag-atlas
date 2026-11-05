from __future__ import annotations

import re

from graphrag_atlas.models import Triple


class ASTGraphBuilder:
    """Builds structural hierarchy knowledge graph edges from document headings."""

    def __init__(self) -> None:
        self._heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def extract_structural_edges(
        self, markdown_text: str, document_id: str
    ) -> list[Triple]:
        """Extract CONTAINS_SECTION relationships from markdown heading nesting."""
        lines = markdown_text.splitlines()
        stack: list[tuple[int, str]] = []  # (level, heading_title)
        edges: list[Triple] = []

        for line in lines:
            match = self._heading_re.match(line.strip())
            if not match:
                continue

            level = len(match.group(1))
            title = match.group(2).strip()

            # Pop higher or equal levels
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                parent_title = stack[-1][1]
                edges.append(
                    Triple(
                        source=parent_title,
                        relation="CONTAINS_SECTION",
                        target=title,
                        weight=1.0,
                        metadata={"document_id": document_id, "structural": True},
                    )
                )

            stack.append((level, title))

        return edges

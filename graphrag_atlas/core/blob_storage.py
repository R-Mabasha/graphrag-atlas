from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BlobStorage(ABC):
    """Abstract interface for storing raw document files and visual artifacts."""

    @abstractmethod
    async def save(self, tenant_id: str, file_name: str, data: bytes) -> str:
        """Store bytes and return a URI or path."""
        ...

    @abstractmethod
    async def load(self, uri: str) -> bytes:
        """Read bytes from storage URI."""
        ...

    @abstractmethod
    async def delete(self, uri: str) -> bool:
        """Delete file at storage URI."""
        ...


class LocalFileStorage(BlobStorage):
    """Stores files on the local filesystem."""

    def __init__(self, base_dir: str = ".graphrag_storage") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, tenant_id: str, file_name: str, data: bytes) -> str:
        tenant_path = self._base_dir / tenant_id
        tenant_path.mkdir(parents=True, exist_ok=True)
        target_file = tenant_path / file_name
        target_file.write_bytes(data)
        return str(target_file.resolve())

    async def load(self, uri: str) -> bytes:
        return Path(uri).read_bytes()

    async def delete(self, uri: str) -> bool:
        p = Path(uri)
        if p.exists():
            p.unlink()
            return True
        return False

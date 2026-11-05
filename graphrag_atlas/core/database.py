from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from graphrag_atlas.core.config import AtlasConfig


class DatabaseManager:
    """Manages AsyncIOMotorClient connection lifecycle and collections."""

    def __init__(self, mongo_uri: str, database_name: str, config: AtlasConfig) -> None:
        self._uri = mongo_uri
        self._db_name = database_name
        self._config = config
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(self._uri)
        self._db: AsyncIOMotorDatabase = self._client[self._db_name]

    @property
    def client(self) -> AsyncIOMotorClient:
        return self._client

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self._db

    @property
    def chunks(self):
        return self._db[self._config.chunks_collection]

    @property
    def edges(self):
        return self._db[self._config.edges_collection]

    @property
    def entities(self):
        return self._db[self._config.entities_collection]

    @property
    def documents(self):
        return self._db[self._config.documents_collection]

    @property
    def communities(self):
        return self._db[self._config.communities_collection]

    async def ping(self) -> bool:
        """Check MongoDB connectivity."""
        res = await self._db.command("ping")
        return res.get("ok") == 1.0

    async def close(self) -> None:
        """Close database connection."""
        self._client.close()

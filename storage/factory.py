"""Storage backend factory."""
from __future__ import annotations

import logging

from config import settings
from .base import StorageBase

logger = logging.getLogger(__name__)


def get_storage() -> StorageBase:
    """Return an initialized StorageBase for the configured provider."""
    from .sqlite_storage import SQLiteStorage

    provider = settings.storage.provider
    match provider:
        case "sqlite":
            logger.info("Using SQLite storage: %s", settings.storage.sqlite_path)
            storage = SQLiteStorage(settings.storage.sqlite_path)
        case "postgres":
            # PostgreSQL support for scale
            try:
                from .postgres_storage import PostgresStorage
                logger.info("Using PostgreSQL storage")
                storage = PostgresStorage(settings.storage.postgres_url)
            except ImportError:
                raise RuntimeError("asyncpg is not installed. Run: pip install asyncpg")
        case _:
            raise ValueError(
                f"Unknown storage provider: '{provider}'. Valid options: sqlite, postgres"
            )

    storage.initialize()
    return storage

"""Abstract base class for all news ingestors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .models import RawArticle


class IngestorBase(ABC):
    """All ingestors must implement fetch() and return a normalized list of RawArticle."""

    @abstractmethod
    async def fetch(self) -> List[RawArticle]:
        """Fetch articles from the source and return them as RawArticle objects.

        - Must handle rate limits, retries, and partial failures internally.
        - Must NOT clean or analyze text — return raw content only.
        - Must normalize all provider-specific fields into the RawArticle schema.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this ingestor (used in logging)."""
        raise NotImplementedError

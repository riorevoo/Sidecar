"""Abstract base class for storage backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from clustering.models import Cluster
from ingestion.models import RawArticle
from .models import RunRecord, VideoRecord


class StorageBase(ABC):
    """Persistent storage interface. Implementations: SQLite, PostgreSQL."""

    @abstractmethod
    def initialize(self) -> None:
        """Create tables if they don't exist."""
        raise NotImplementedError

    @abstractmethod
    def create_run(self) -> int:
        """Insert a new RunRecord and return its ID."""
        raise NotImplementedError

    @abstractmethod
    def complete_run(self, run_id: int, status: str, **counts) -> None:
        """Mark a run as complete and record stage counts."""
        raise NotImplementedError

    @abstractmethod
    def get_seen_hashes(self) -> set[str]:
        """Return all content_hash values already in the DB (for dedup)."""
        raise NotImplementedError

    @abstractmethod
    def save_articles(self, articles: list, run_id: Optional[int] = None) -> None:
        """Persist cleaned articles."""
        raise NotImplementedError

    @abstractmethod
    def save_clusters(self, clusters: List[Cluster], run_id: int) -> None:
        """Persist cluster metadata and link articles."""
        raise NotImplementedError

    @abstractmethod
    def save_video(self, cluster_id: str, video_result) -> int:
        """Persist a VideoRecord and return its ID."""
        raise NotImplementedError

    @abstractmethod
    def get_unpublished_videos(self) -> List[VideoRecord]:
        """Return videos not yet published on any platform."""
        raise NotImplementedError

    @abstractmethod
    def mark_published(
        self,
        video_id: int,
        platform: str,
        platform_video_id: str,
        platform_url: str,
    ) -> None:
        """Record a successful publish."""
        raise NotImplementedError

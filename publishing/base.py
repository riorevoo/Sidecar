"""Abstract base class for publishing platforms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PublishResult:
    platform: str
    success: bool
    platform_video_id: str = ""
    platform_url: str = ""
    error: str = ""


class PublisherBase(ABC):
    """Upload a video to a social media platform."""

    @abstractmethod
    def publish(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        thumbnail_path: Optional[str] = None,
    ) -> PublishResult:
        """Upload video and return the result.

        Implementations must be idempotent — safe to retry on failure.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def platform_name(self) -> str:
        raise NotImplementedError

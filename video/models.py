"""Data models for video generation output."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoResult:
    """Result of a video generation call."""
    output_path: str         # Path to the final MP4 file
    duration_seconds: float
    width: int
    height: int
    thumbnail_path: str = ""  # Path to extracted thumbnail image
    story_id: str = ""
    provider: str = ""

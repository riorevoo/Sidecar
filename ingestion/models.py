"""Shared data models for the ingestion stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RawArticle:
    """A single raw article as fetched from a news source.
    No analysis has been performed — this is pure ingestion output.
    """
    url: str
    title: str
    content: str
    source: str
    timestamp: datetime
    language: str = "en"
    raw_html: str = ""
    author: str = ""
    image_url: str = ""
    # Stable hash used for deduplication before DB insertion
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            import hashlib
            self.content_hash = hashlib.sha256(
                (self.url + self.title).encode("utf-8")
            ).hexdigest()

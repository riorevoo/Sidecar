"""Data models for the cleaning stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ingestion.models import RawArticle


@dataclass
class CleanArticle:
    """A RawArticle that has been through the cleaning pipeline."""
    # Original fields preserved for traceability
    url: str
    title: str
    source: str
    timestamp: datetime
    content_hash: str

    # Cleaned output — this is what gets embedded
    cleaned_text: str

    # Metadata added during cleaning
    language: str = "en"
    word_count: int = 0
    author: str = ""
    image_url: str = ""

    @classmethod
    def from_raw(cls, raw: RawArticle, cleaned_text: str) -> "CleanArticle":
        return cls(
            url=raw.url,
            title=raw.title,
            source=raw.source,
            timestamp=raw.timestamp,
            content_hash=raw.content_hash,
            cleaned_text=cleaned_text,
            language=raw.language,
            word_count=len(cleaned_text.split()),
            author=raw.author,
            image_url=raw.image_url,
        )

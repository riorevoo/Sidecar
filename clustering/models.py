"""Data models for the clustering stage."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from cleaning.models import CleanArticle


@dataclass
class Cluster:
    """A group of topically related articles that together form a story."""
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    articles: List[CleanArticle] = field(default_factory=list)

    # Computed during clustering
    centroid: Optional[object] = None  # np.ndarray, set after clustering
    coherence_score: float = 0.0       # Average pairwise cosine similarity

    # Computed during ranking
    importance_score: float = 0.0

    # Set by enrichment
    entities: List[str] = field(default_factory=list)
    topic: str = ""
    summary: str = ""

    # Set by script generation
    script: Optional[object] = None    # script.models.Script

    @property
    def size(self) -> int:
        return len(self.articles)

    @property
    def unique_sources(self) -> set[str]:
        return {a.source for a in self.articles}

    @property
    def latest_timestamp(self) -> Optional[datetime]:
        if not self.articles:
            return None
        return max(a.timestamp for a in self.articles)

    @property
    def oldest_timestamp(self) -> Optional[datetime]:
        if not self.articles:
            return None
        return min(a.timestamp for a in self.articles)

    @property
    def representative_title(self) -> str:
        """Return the title of the most recent article as representative headline."""
        if not self.articles:
            return ""
        latest = max(self.articles, key=lambda a: a.timestamp)
        return latest.title

    @property
    def all_text(self) -> str:
        """Concatenate all cleaned article text for LLM summarization."""
        return "\n\n".join(
            f"[{a.source}] {a.cleaned_text}" for a in self.articles[:5]
        )

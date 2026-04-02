"""PipelineContext — the data carrier that flows through all pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from ingestion.models import RawArticle
from cleaning.models import CleanArticle
from clustering.models import Cluster
from video.models import VideoResult


@dataclass
class PipelineContext:
    """Holds all intermediate data across a single pipeline run."""
    run_id: int = 0
    run_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Stage outputs — each stage reads the previous and writes the next
    raw_articles: List[RawArticle] = field(default_factory=list)
    clean_articles: List[CleanArticle] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)
    ranked_clusters: List[Cluster] = field(default_factory=list)

    # Final outputs
    videos: List[VideoResult] = field(default_factory=list)

    # Run metadata
    errors: List[str] = field(default_factory=list)

    def record_error(self, stage: str, error: Exception) -> None:
        msg = f"[{stage}] {type(error).__name__}: {error}"
        self.errors.append(msg)

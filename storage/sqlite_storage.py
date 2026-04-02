"""SQLite storage backend (MVP default).

Uses SQLAlchemy with a synchronous engine for simplicity.
WAL mode is enabled for safe concurrent reads during scheduling.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from clustering.models import Cluster
from video.models import VideoResult
from .base import StorageBase
from .models import (
    Base, ArticleRecord, ClusterRecord, RunRecord, VideoRecord, PublishRecord
)

logger = logging.getLogger(__name__)


class SQLiteStorage(StorageBase):
    """SQLite-backed storage using SQLAlchemy."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for concurrent readers
        @event.listens_for(self._engine, "connect")
        def set_wal_mode(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")

    def initialize(self) -> None:
        Base.metadata.create_all(self._engine)
        logger.info("SQLite storage initialized")

    def create_run(self) -> int:
        with Session(self._engine) as session:
            run = RunRecord(started_at=datetime.now(tz=timezone.utc))
            session.add(run)
            session.commit()
            session.refresh(run)
            return run.id

    def complete_run(self, run_id: int, status: str = "success", **counts) -> None:
        with Session(self._engine) as session:
            run = session.get(RunRecord, run_id)
            if run:
                run.completed_at = datetime.now(tz=timezone.utc)
                run.status = status
                for key, val in counts.items():
                    if hasattr(run, key):
                        setattr(run, key, val)
                session.commit()

    def get_seen_hashes(self) -> set[str]:
        with Session(self._engine) as session:
            rows = session.query(ArticleRecord.content_hash).all()
            return {r[0] for r in rows}

    def save_articles(self, articles: list, run_id: Optional[int] = None) -> None:
        with Session(self._engine) as session:
            saved = 0
            for article in articles:
                # Skip if already exists (dedup)
                existing = session.query(ArticleRecord).filter_by(
                    content_hash=article.content_hash
                ).first()
                if existing:
                    continue
                record = ArticleRecord(
                    url=article.url,
                    title=article.title,
                    source=article.source,
                    published_at=article.timestamp,
                    content_hash=article.content_hash,
                    cleaned_text=article.cleaned_text,
                    language=article.language,
                    word_count=article.word_count,
                )
                session.add(record)
                saved += 1
            session.commit()
            logger.info("Saved %d new articles to storage", saved)

    def save_clusters(self, clusters: List[Cluster], run_id: int) -> None:
        with Session(self._engine) as session:
            for cluster in clusters:
                article_urls = [
                    a.url for a in cluster.articles if getattr(a, "url", None)
                ]
                record = ClusterRecord(
                    cluster_id=cluster.cluster_id,
                    run_id=run_id,
                    topic=cluster.topic or "",
                    summary=cluster.summary or "",
                    entities=json.dumps(cluster.entities),
                    importance_score=cluster.importance_score,
                    article_count=cluster.size,
                    source_count=len(cluster.unique_sources),
                    script_text=(
                        cluster.script.full_text if cluster.script else None
                    ),
                    article_urls=json.dumps(article_urls),
                    latest_article_at=cluster.latest_timestamp,
                )
                session.merge(record)

                # Link articles to this cluster
                for article in cluster.articles:
                    row = session.query(ArticleRecord).filter_by(
                        content_hash=article.content_hash
                    ).first()
                    if row:
                        row.cluster_id = cluster.cluster_id

            session.commit()
            logger.info("Saved %d clusters to storage", len(clusters))

    def save_video(self, cluster_id: str, video_result: VideoResult) -> int:
        with Session(self._engine) as session:
            record = VideoRecord(
                cluster_id=cluster_id,
                video_path=video_result.output_path,
                thumbnail_path=video_result.thumbnail_path,
                duration_seconds=video_result.duration_seconds,
                provider=video_result.provider,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def get_unpublished_videos(self) -> List[VideoRecord]:
        with Session(self._engine) as session:
            return (
                session.query(VideoRecord)
                .filter(VideoRecord.published == False)
                .all()
            )

    def mark_published(
        self,
        video_id: int,
        platform: str,
        platform_video_id: str,
        platform_url: str,
    ) -> None:
        with Session(self._engine) as session:
            record = PublishRecord(
                video_id=video_id,
                platform=platform,
                platform_video_id=platform_video_id,
                platform_url=platform_url,
                status="success",
            )
            session.add(record)
            video = session.get(VideoRecord, video_id)
            if video:
                video.published = True
            session.commit()

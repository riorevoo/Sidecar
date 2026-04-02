"""SQLAlchemy ORM models shared between SQLite and PostgreSQL backends."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, Boolean,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class ArticleRecord(Base):
    """Stores each ingested article."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), nullable=False)
    title = Column(Text, nullable=False)
    source = Column(String(256), nullable=False)
    published_at = Column(DateTime, nullable=True)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    cleaned_text = Column(Text, nullable=True)
    language = Column(String(8), default="en")
    word_count = Column(Integer, default=0)
    cluster_id = Column(String(36), ForeignKey("clusters.cluster_id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ClusterRecord(Base):
    """Stores each story cluster."""
    __tablename__ = "clusters"

    cluster_id = Column(String(36), primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=True)
    topic = Column(String(64), nullable=True)
    summary = Column(Text, nullable=True)
    entities = Column(Text, nullable=True)   # JSON array stored as text
    importance_score = Column(Float, default=0.0)
    article_count = Column(Integer, default=0)
    source_count = Column(Integer, default=0)
    script_text = Column(Text, nullable=True)
    article_urls = Column(Text, nullable=True)  # JSON array of source URLs
    latest_article_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    articles = relationship("ArticleRecord", foreign_keys=[ArticleRecord.cluster_id],
                            primaryjoin="ClusterRecord.cluster_id == ArticleRecord.cluster_id")
    videos = relationship("VideoRecord", back_populates="cluster")


class RunRecord(Base):
    """Audit log — one row per pipeline execution."""
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(32), default="running")  # running, success, failed

    # Stage-level counts for funnel analysis
    articles_ingested = Column(Integer, default=0)
    articles_cleaned = Column(Integer, default=0)
    clusters_formed = Column(Integer, default=0)
    clusters_ranked = Column(Integer, default=0)
    scripts_generated = Column(Integer, default=0)
    videos_generated = Column(Integer, default=0)

    error_message = Column(Text, nullable=True)
    clusters = relationship("ClusterRecord", back_populates="run")


# Fix circular relationship
ClusterRecord.run = relationship("RunRecord", back_populates="clusters")


class VideoRecord(Base):
    """Tracks generated videos and their publish status."""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(String(36), ForeignKey("clusters.cluster_id"), nullable=False)
    video_path = Column(String(1024), nullable=False)
    thumbnail_path = Column(String(1024), nullable=True)
    duration_seconds = Column(Float, default=0.0)
    provider = Column(String(32), nullable=True)
    published = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())

    cluster = relationship("ClusterRecord", back_populates="videos")
    publish_records = relationship("PublishRecord", back_populates="video")


class PublishRecord(Base):
    """Tracks publish attempts per platform."""
    __tablename__ = "publish_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    platform = Column(String(32), nullable=False)  # youtube, instagram, tiktok
    platform_video_id = Column(String(256), nullable=True)
    platform_url = Column(String(1024), nullable=True)
    status = Column(String(32), default="pending")  # pending, success, failed
    error_message = Column(Text, nullable=True)
    attempted_at = Column(DateTime, server_default=func.now())

    video = relationship("VideoRecord", back_populates="publish_records")

"""GDELT 2.0 DOC API ingestor.

Uses the gdeltdoc library to query the GDELT Document API.
GDELT is free, global, and requires no API key.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List

from config import settings
from .base import IngestorBase
from .models import RawArticle

logger = logging.getLogger(__name__)


class GDELTIngestor(IngestorBase):
    """Fetches news articles from GDELT 2.0 using the gdeltdoc library."""

    name = "gdelt"

    def __init__(self) -> None:
        self._cfg = settings.ingestion

    async def fetch(self) -> List[RawArticle]:
        """Run GDELT queries in a thread pool to avoid blocking the event loop."""
        return await asyncio.get_event_loop().run_in_executor(None, self._fetch_sync)

    def _fetch_sync(self) -> List[RawArticle]:
        try:
            from gdeltdoc import GdeltDoc, Filters
        except ImportError:
            logger.error("gdeltdoc is not installed. Run: pip install gdeltdoc")
            return []

        gd = GdeltDoc()
        articles: List[RawArticle] = []

        # Build a broad filter — English language, recent articles
        f = Filters(
            timespan=f"{self._cfg.ingest_lookback_hours}h",
            num_records=min(self._cfg.ingest_max_articles, 250),  # GDELT max per call
        )

        try:
            result = gd.article_search(f)
        except Exception as exc:
            logger.warning("GDELT article_search failed: %s", exc)
            return []

        if result is None or result.empty:
            logger.info("GDELT returned no articles")
            return []

        for _, row in result.iterrows():
            try:
                pub_ts = self._parse_timestamp(row.get("seendate", ""))
                article = RawArticle(
                    url=str(row.get("url", "")),
                    title=str(row.get("title", "")),
                    content=str(row.get("title", "")),  # GDELT doesn't provide body text
                    source=str(row.get("domain", "gdelt")),
                    timestamp=pub_ts,
                    language=str(row.get("language", "English")).lower()[:2],
                    image_url=str(row.get("socialimage", "")),
                )
                if article.url and article.title:
                    articles.append(article)
            except Exception as exc:
                logger.debug("Skipping GDELT row due to parse error: %s", exc)
                continue

        logger.info("GDELT fetched %d articles", len(articles))
        return articles[: self._cfg.ingest_max_articles]

    @staticmethod
    def _parse_timestamp(raw: str) -> datetime:
        """Parse GDELT's seendate format: YYYYMMDDTHHMMSSZ"""
        try:
            return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return datetime.now(tz=timezone.utc)

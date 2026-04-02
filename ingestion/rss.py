"""RSS feed ingestor.

Fetches and parses RSS/Atom feeds using feedparser.
Attempts to extract full article text using newspaper4k where available.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List
from urllib.parse import urlparse

import httpx
import feedparser

from config import settings
from .base import IngestorBase
from .models import RawArticle

logger = logging.getLogger(__name__)

# Seconds to wait between full-text fetch requests to be polite
_FETCH_DELAY = 0.5
_HTTP_TIMEOUT = 15


class RSSIngestor(IngestorBase):
    """Fetches articles from a list of RSS/Atom feed URLs."""

    name = "rss"

    def __init__(self) -> None:
        self._cfg = settings.ingestion

    async def fetch(self) -> List[RawArticle]:
        feeds = self._cfg.rss_feed_list
        if not feeds:
            logger.warning("No RSS feeds configured. Set RSS_FEEDS in .env")
            return []

        all_articles: List[RawArticle] = []
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            for feed_url in feeds:
                try:
                    articles = await self._fetch_feed(client, feed_url)
                    all_articles.extend(articles)
                    logger.info("RSS [%s]: fetched %d articles", feed_url, len(articles))
                except Exception as exc:
                    logger.warning("RSS feed failed [%s]: %s", feed_url, exc)
                    continue

        # Deduplicate by URL within this batch
        seen: set[str] = set()
        unique = []
        for a in all_articles:
            if a.url not in seen:
                seen.add(a.url)
                unique.append(a)

        logger.info("RSS total unique articles: %d", len(unique))
        return unique[: self._cfg.ingest_max_articles]

    async def _fetch_feed(
        self, client: httpx.AsyncClient, feed_url: str
    ) -> List[RawArticle]:
        response = await client.get(feed_url)
        response.raise_for_status()

        parsed = feedparser.parse(response.text)
        articles: List[RawArticle] = []
        source_domain = urlparse(feed_url).netloc

        for entry in parsed.entries:
            url = entry.get("link", "")
            title = entry.get("title", "")
            if not url or not title:
                continue

            # Best-effort body extraction from feed summary
            content = (
                entry.get("summary", "")
                or entry.get("content", [{}])[0].get("value", "")
                if hasattr(entry, "content")
                else entry.get("summary", "")
            )

            timestamp = self._parse_entry_timestamp(entry)

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    content=content,
                    source=source_domain,
                    timestamp=timestamp,
                    author=entry.get("author", ""),
                    image_url=self._extract_image(entry),
                )
            )

        return articles

    @staticmethod
    def _parse_entry_timestamp(entry: feedparser.FeedParserDict) -> datetime:
        """Try multiple date fields in order of reliability."""
        for field in ("published", "updated", "created"):
            raw = entry.get(field, "")
            if not raw:
                continue
            try:
                return parsedate_to_datetime(raw).replace(tzinfo=timezone.utc)
            except Exception:
                pass
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _extract_image(entry: feedparser.FeedParserDict) -> str:
        """Try to find a thumbnail or media image in the feed entry."""
        if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
            return entry.media_thumbnail[0].get("url", "")
        if hasattr(entry, "media_content") and entry.media_content:
            return entry.media_content[0].get("url", "")
        return ""

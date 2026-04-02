"""NewsAPI ingestor.

Uses the newsapi-python client to fetch top headlines.
Free tier: 100 requests/day, 100 articles/request.
Requires NEWSAPI_KEY in .env
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


class NewsAPIIngestor(IngestorBase):
    """Fetches English top headlines from NewsAPI."""

    name = "newsapi"

    def __init__(self) -> None:
        self._cfg = settings.ingestion
        self._api_key = settings.ingestion.newsapi_key

    async def fetch(self) -> List[RawArticle]:
        if not self._api_key:
            logger.warning("NEWSAPI_KEY not set — skipping NewsAPI ingestor")
            return []
        return await asyncio.get_event_loop().run_in_executor(None, self._fetch_sync)

    def _fetch_sync(self) -> List[RawArticle]:
        try:
            from newsapi import NewsApiClient
        except ImportError:
            logger.error("newsapi-python is not installed. Run: pip install newsapi-python")
            return []

        client = NewsApiClient(api_key=self._api_key)
        articles: List[RawArticle] = []

        page = 1
        page_size = 100  # max allowed by free tier

        while len(articles) < self._cfg.ingest_max_articles:
            try:
                response = client.get_top_headlines(
                    language="en",
                    page=page,
                    page_size=page_size,
                )
            except Exception as exc:
                logger.warning("NewsAPI request failed (page %d): %s", page, exc)
                break

            batch = response.get("articles", [])
            if not batch:
                break

            for item in batch:
                url = item.get("url", "")
                title = item.get("title", "")
                if not url or not title or title == "[Removed]":
                    continue

                articles.append(
                    RawArticle(
                        url=url,
                        title=title,
                        content=item.get("content", "") or item.get("description", ""),
                        source=item.get("source", {}).get("name", "newsapi"),
                        timestamp=self._parse_ts(item.get("publishedAt", "")),
                        author=item.get("author", ""),
                        image_url=item.get("urlToImage", ""),
                    )
                )

            if len(batch) < page_size:
                break  # No more pages
            page += 1

        logger.info("NewsAPI fetched %d articles", len(articles))
        return articles[: self._cfg.ingest_max_articles]

    @staticmethod
    def _parse_ts(raw: str) -> datetime:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now(tz=timezone.utc)

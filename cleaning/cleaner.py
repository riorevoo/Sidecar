"""Main text cleaning orchestrator."""
from __future__ import annotations

import logging
from typing import List

from ingestion.models import RawArticle
from .models import CleanArticle
from .normalizer import normalize

logger = logging.getLogger(__name__)

# Minimum word count to keep an article after cleaning
_MIN_WORD_COUNT = 10
# Target languages (ISO 639-1 codes)
_ALLOWED_LANGUAGES = {"en"}


class TextCleaner:
    """Cleans a batch of RawArticle objects and returns CleanArticle objects.

    Dropped articles are logged but not raised as errors — partial success is
    the expected outcome of a broad news fetch.
    """

    def clean(self, articles: List[RawArticle]) -> List[CleanArticle]:
        """Clean all articles. Returns only articles that pass quality filters."""
        seen_hashes: set[str] = set()
        results: List[CleanArticle] = []

        for article in articles:
            try:
                cleaned = self._clean_one(article, seen_hashes)
                if cleaned is not None:
                    results.append(cleaned)
                    seen_hashes.add(article.content_hash)
            except Exception as exc:
                logger.debug("Cleaning failed for %s: %s", article.url, exc)
                continue

        logger.info(
            "Cleaning: %d in → %d out (dropped %d)",
            len(articles),
            len(results),
            len(articles) - len(results),
        )
        return results

    def _clean_one(
        self, article: RawArticle, seen_hashes: set[str]
    ) -> CleanArticle | None:
        # Deduplicate by content hash
        if article.content_hash in seen_hashes:
            return None

        # Combine title and content for embedding (title carries high signal)
        raw_text = f"{article.title}. {article.content}"
        cleaned_text = normalize(raw_text)

        # Drop if too short after cleaning
        word_count = len(cleaned_text.split())
        if word_count < _MIN_WORD_COUNT:
            return None

        # Basic language filter
        lang = article.language.lower()[:2] if article.language else "en"
        if lang not in _ALLOWED_LANGUAGES:
            # Attempt langdetect as fallback
            try:
                from langdetect import detect
                lang = detect(cleaned_text)
            except Exception:
                pass
            if lang not in _ALLOWED_LANGUAGES:
                return None

        clean = CleanArticle.from_raw(article, cleaned_text)
        clean.language = lang
        clean.word_count = word_count
        return clean

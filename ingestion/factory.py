"""Factory for building ingestor instances from config."""
from __future__ import annotations

import logging
from typing import List

from config import settings
from .base import IngestorBase

logger = logging.getLogger(__name__)


def build_ingestors() -> List[IngestorBase]:
    """Return one IngestorBase instance per configured provider."""
    from .gdelt import GDELTIngestor
    from .rss import RSSIngestor
    from .newsapi import NewsAPIIngestor

    _registry: dict[str, type[IngestorBase]] = {
        "gdelt": GDELTIngestor,
        "rss": RSSIngestor,
        "newsapi": NewsAPIIngestor,
    }

    ingestors: List[IngestorBase] = []
    for name in settings.ingestion.ingestor_list:
        cls = _registry.get(name)
        if cls is None:
            logger.warning("Unknown ingestor '%s' — skipping", name)
            continue
        ingestors.append(cls())
        logger.debug("Registered ingestor: %s", name)

    return ingestors

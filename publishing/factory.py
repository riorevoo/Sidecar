"""Publishing factory."""
from __future__ import annotations

import logging
from typing import List

from config import settings
from .base import PublisherBase

logger = logging.getLogger(__name__)


def build_publishers() -> List[PublisherBase]:
    """Build publisher instances for each configured platform."""
    from .youtube import YouTubePublisher
    from .instagram import InstagramPublisher

    registry: dict[str, type[PublisherBase]] = {
        "youtube": YouTubePublisher,
        "instagram": InstagramPublisher,
    }

    publishers: List[PublisherBase] = []
    for name in settings.publishing.publisher_list:
        cls = registry.get(name)
        if cls is None:
            logger.warning("Unknown publisher '%s' — skipping", name)
            continue
        publishers.append(cls())
        logger.info("Registered publisher: %s", name)

    return publishers

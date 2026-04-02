"""Factory for embedding provider selection."""
from __future__ import annotations

import logging

from config import settings
from .base import EmbedderBase

logger = logging.getLogger(__name__)


def get_embedder() -> EmbedderBase:
    """Return an EmbedderBase instance for the configured provider."""
    from .local_embedder import LocalEmbedder
    from .openai_embedder import OpenAIEmbedder

    provider = settings.embedding.provider
    match provider:
        case "local":
            logger.info("Using local sentence-transformers embedder")
            return LocalEmbedder()
        case "openai":
            logger.info("Using OpenAI embedder")
            return OpenAIEmbedder()
        case _:
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                "Valid options: local, openai"
            )

"""Local sentence-transformers embedding provider.

Uses the sentence-transformers library with a model that runs on GPU if available.
The model is loaded lazily on first call and stays resident in memory to amortize
the large load cost across many batches.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from config import settings
from utils.hardware import get_device
from .base import EmbedderBase

logger = logging.getLogger(__name__)


class LocalEmbedder(EmbedderBase):
    """Embeds text using a local sentence-transformers model."""

    def __init__(self) -> None:
        self._cfg = settings.embedding
        self._model_name = self._cfg.local_model
        self._model = None  # Lazy load
        self._dim: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        device = get_device()
        logger.info("Loading embedding model '%s' on %s", self._model_name, device)
        self._model = SentenceTransformer(self._model_name, device=device)
        # Probe dimension with a dummy encode
        probe = self._model.encode(["probe"], normalize_embeddings=True)
        self._dim = probe.shape[1]
        logger.info("Embedding model loaded. Dimension: %d", self._dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        self._load()
        if not texts:
            return np.empty((0, self._dim or 384), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,  # L2 norm → cosine sim = dot product
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        self._load()
        return self._dim or 384

"""OpenAI embeddings provider.

Uses the OpenAI text-embedding API. Batches requests to respect
the per-request token limit and rate limits.
Requires OPENAI_API_KEY in .env.
"""
from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

from config import settings
from .base import EmbedderBase

logger = logging.getLogger(__name__)

_OPENAI_EMBED_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(EmbedderBase):
    """Embeds text using the OpenAI embeddings API."""

    def __init__(self) -> None:
        self._cfg = settings.embedding
        self._api_key = settings.openai_api_key
        self._client = None  # Lazy init

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("openai is not installed. Run: pip install openai")
            if not self._api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to .env or set as environment variable."
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        client = self._get_client()
        batch_size = self._cfg.openai_batch_size
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    response = client.embeddings.create(
                        model=self._cfg.openai_model,
                        input=batch,
                    )
                    # API returns embeddings in input order
                    batch_vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                    all_embeddings.extend(batch_vecs)
                    break
                except Exception as exc:
                    if attempt < 2:
                        wait = 2 ** attempt
                        logger.warning(
                            "OpenAI embed attempt %d failed: %s. Retrying in %ds",
                            attempt + 1, exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        raise

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-10)
        logger.info("OpenAI embedded %d texts → shape %s", len(texts), arr.shape)
        return arr

    @property
    def embedding_dim(self) -> int:
        return _OPENAI_EMBED_DIM.get(self._cfg.openai_model, 1536)

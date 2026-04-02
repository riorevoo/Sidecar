"""Abstract base class for embedding providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbedderBase(ABC):
    """Convert a list of text strings into a 2D numpy array of float vectors.

    All implementations must:
    - Return shape (N, D) where N = len(texts) and D is the model's embedding dim.
    - Use L2-normalized vectors (unit length) so cosine similarity = dot product.
    - Handle batching internally.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of cleaned text strings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output vectors."""
        raise NotImplementedError

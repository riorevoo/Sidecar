"""Zero-shot topic classification using HuggingFace transformers."""
from __future__ import annotations

import logging
from typing import List

from clustering.models import Cluster

logger = logging.getLogger(__name__)

# Coarse topic labels for news classification
TOPIC_LABELS = [
    "politics",
    "business",
    "technology",
    "science",
    "health",
    "sports",
    "entertainment",
    "environment",
    "world",
    "crime",
    "education",
    "economy",
]


class TopicClassifier:
    """Classifies clusters into coarse topic categories using zero-shot classification."""

    def __init__(self) -> None:
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError(
                "transformers is not installed. Run: pip install transformers"
            )
        logger.info("Loading zero-shot classification pipeline...")
        self._pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self._use_gpu() else -1,
        )
        logger.info("Topic classifier loaded")

    def _use_gpu(self) -> bool:
        from utils.hardware import get_device
        return get_device() != "cpu"

    def classify(self, cluster: Cluster) -> str:
        """Return the most likely topic label for the cluster."""
        self._load()

        # Use representative title + summary for classification
        text = f"{cluster.representative_title}. {cluster.summary or ''}"
        text = text[:512]  # Keep within model limits

        result = self._pipe(text, candidate_labels=TOPIC_LABELS, multi_label=False)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        logger.debug(
            "Topic classification: '%s' → %s (%.2f)",
            cluster.representative_title[:50],
            top_label,
            top_score,
        )
        return top_label

    def unload(self) -> None:
        """Release model from VRAM when no longer needed."""
        import gc
        self._pipe = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

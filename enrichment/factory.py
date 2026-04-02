"""Build the enrichment pipeline."""
from __future__ import annotations

import logging
from typing import List

from clustering.models import Cluster
from config import settings
from .base import EnricherBase
from .ner import SpacyNER
from .topic_classifier import TopicClassifier
from .summarizer import LLMSummarizer

logger = logging.getLogger(__name__)


class FullEnricher(EnricherBase):
    """Runs NER, topic classification, and LLM summarization on each cluster."""

    def __init__(self) -> None:
        self._ner = SpacyNER()
        self._topic = TopicClassifier()
        self._summarizer = LLMSummarizer()

    def enrich(self, clusters: List[Cluster]) -> List[Cluster]:
        for cluster in clusters:
            try:
                # Order matters: summarize first so topic classifier can use it
                cluster.summary = self._summarizer.summarize(cluster)
                cluster.entities = self._ner.extract_entities(cluster)
                cluster.topic = self._topic.classify(cluster)
                logger.debug(
                    "Enriched cluster %s: topic=%s entities=%d",
                    cluster.cluster_id[:8],
                    cluster.topic,
                    len(cluster.entities),
                )
            except Exception as exc:
                logger.warning(
                    "Enrichment failed for cluster %s: %s",
                    cluster.cluster_id[:8],
                    exc,
                )
        return clusters


class MinimalEnricher(EnricherBase):
    """Lightweight enricher for MVP mode — summarization only, no heavy models."""

    def __init__(self) -> None:
        self._summarizer = LLMSummarizer()

    def enrich(self, clusters: List[Cluster]) -> List[Cluster]:
        for cluster in clusters:
            try:
                if not cluster.summary:
                    cluster.summary = self._summarizer.summarize(cluster)
            except Exception as exc:
                logger.warning(
                    "MVP enrichment failed for cluster %s: %s",
                    cluster.cluster_id[:8],
                    exc,
                )
                if not cluster.summary:
                    cluster.summary = cluster.representative_title
        return clusters


def build_enricher() -> EnricherBase:
    if settings.enrichment.enabled:
        logger.info("Using full enricher (NER + topic + summarization)")
        return FullEnricher()
    logger.info("Using minimal enricher (summarization only)")
    return MinimalEnricher()

"""Cluster ranker — selects and orders top stories by importance."""
from __future__ import annotations

import logging
from typing import List

from clustering.models import Cluster
from config import settings
from .scoring import compute_importance_score

logger = logging.getLogger(__name__)


class ClusterRanker:
    """Ranks clusters by importance and returns the top N."""

    def rank(self, clusters: List[Cluster]) -> List[Cluster]:
        """Score all clusters and return top N sorted by score descending.

        Scores are written back to cluster.importance_score for downstream use.
        """
        if not clusters:
            return []

        for cluster in clusters:
            cluster.importance_score = compute_importance_score(cluster)

        ranked = sorted(clusters, key=lambda c: c.importance_score, reverse=True)
        top_n = settings.ranking.top_n
        result = ranked[:top_n]

        logger.info(
            "Ranking: %d clusters → top %d selected",
            len(clusters),
            len(result),
        )
        for i, c in enumerate(result):
            logger.info(
                "  #%d [%.3f] %s (%d articles, %d sources)",
                i + 1,
                c.importance_score,
                c.representative_title[:60],
                c.size,
                len(c.unique_sources),
            )

        return result

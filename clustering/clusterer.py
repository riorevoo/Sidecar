"""Cosine similarity clustering for news articles.

Groups articles into stories using agglomerative clustering with
a cosine distance threshold. Articles within the same cluster cover
the same underlying news event.
"""
from __future__ import annotations

import logging
import uuid
from typing import List

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from cleaning.models import CleanArticle
from config import settings
from .models import Cluster

logger = logging.getLogger(__name__)


class CosineClustering:
    """Groups articles into story clusters using cosine similarity thresholding."""

    def __init__(self) -> None:
        self._cfg = settings.clustering

    def cluster(
        self,
        articles: List[CleanArticle],
        embeddings: np.ndarray,
    ) -> List[Cluster]:
        """Cluster articles into stories.

        Args:
            articles: List of CleanArticle objects in the same order as embeddings.
            embeddings: np.ndarray of shape (N, D), L2-normalized float32.

        Returns:
            List of Cluster objects, sorted by size (largest first).
        """
        if not articles:
            return []

        if len(articles) == 1:
            return [self._make_cluster([articles[0]], embeddings[0:1])]

        # Convert cosine similarity threshold to distance threshold
        # cosine_distance = 1 - cosine_similarity
        distance_threshold = 1.0 - self._cfg.threshold

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )

        labels = model.fit_predict(embeddings)
        n_clusters = model.n_clusters_
        logger.info(
            "Clustering: %d articles → %d raw clusters (threshold=%.2f)",
            len(articles),
            n_clusters,
            self._cfg.threshold,
        )

        # Group articles by label
        groups: dict[int, list[tuple[CleanArticle, np.ndarray]]] = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append((articles[i], embeddings[i]))

        clusters: List[Cluster] = []
        for label, members in groups.items():
            member_articles = [m[0] for m in members]
            member_embeddings = np.stack([m[1] for m in members])

            if len(member_articles) < self._cfg.min_cluster_size:
                continue

            cluster = self._make_cluster(member_articles, member_embeddings)
            clusters.append(cluster)

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)

        # Cap at max_clusters
        clusters = clusters[: self._cfg.max_clusters]

        logger.info(
            "Clustering output: %d clusters (min_size=%d, max=%d)",
            len(clusters),
            self._cfg.min_cluster_size,
            self._cfg.max_clusters,
        )
        return clusters

    def _make_cluster(
        self, articles: List[CleanArticle], embeddings: np.ndarray
    ) -> Cluster:
        centroid = embeddings.mean(axis=0)
        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm

        # Compute coherence as mean pairwise cosine similarity
        if len(embeddings) > 1:
            sim_matrix = cosine_similarity(embeddings)
            n = len(embeddings)
            # Exclude diagonal (self-similarity = 1.0)
            mask = ~np.eye(n, dtype=bool)
            coherence = float(sim_matrix[mask].mean())
        else:
            coherence = 1.0

        return Cluster(
            cluster_id=str(uuid.uuid4()),
            articles=articles,
            centroid=centroid,
            coherence_score=coherence,
        )

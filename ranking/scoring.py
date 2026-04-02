"""Weighted importance scoring for news clusters.

Scoring is fully deterministic — no LLM calls.
All weights are configurable via .env.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List

from clustering.models import Cluster
from config import settings


def compute_importance_score(cluster: Cluster) -> float:
    """Compute a weighted importance score for a cluster.

    Formula:
        score = w_coverage * coverage
              + w_diversity * diversity
              + w_velocity * velocity
              + w_entity * entity_weight
              + w_llm * llm_score

    All sub-scores are normalized to [0, 1] before weighting.
    Time decay is applied to the final score.
    """
    cfg = settings.ranking

    coverage = _coverage_score(cluster)
    diversity = _diversity_score(cluster)
    velocity = _velocity_score(cluster)
    entity_weight = _entity_weight_score(cluster)
    # LLM score placeholder — 0.5 when not precomputed
    llm_score = getattr(cluster, "llm_score", 0.5)

    raw_score = (
        cfg.weight_coverage * coverage
        + cfg.weight_diversity * diversity
        + cfg.weight_velocity * velocity
        + cfg.weight_entity * entity_weight
        + cfg.weight_llm * llm_score
    )

    # Apply time decay based on cluster's newest article
    decay = _time_decay(cluster, cfg.decay_half_life_hours)
    return raw_score * decay


def _coverage_score(cluster: Cluster) -> float:
    """Normalize cluster size to [0, 1]. Cap at 20 articles = 1.0."""
    return min(cluster.size / 20.0, 1.0)


def _diversity_score(cluster: Cluster) -> float:
    """Normalize unique source count to [0, 1]. Cap at 10 sources = 1.0."""
    return min(len(cluster.unique_sources) / 10.0, 1.0)


def _velocity_score(cluster: Cluster) -> float:
    """Rate of article arrival within the cluster's time window.

    Higher velocity = more articles arriving recently = higher score.
    """
    if cluster.size < 2:
        return 0.0
    oldest = cluster.oldest_timestamp
    newest = cluster.latest_timestamp
    if oldest is None or newest is None:
        return 0.0

    window_hours = max((newest - oldest).total_seconds() / 3600.0, 0.01)
    articles_per_hour = cluster.size / window_hours
    # Normalize: 5+ articles/hour = score 1.0
    return min(articles_per_hour / 5.0, 1.0)


def _entity_weight_score(cluster: Cluster) -> float:
    """Score based on presence of named entities (more = more newsworthy).

    Cap at 10 entities = 1.0.
    """
    return min(len(cluster.entities) / 10.0, 1.0)


def _time_decay(cluster: Cluster, half_life_hours: float) -> float:
    """Exponential time decay: score halves every half_life_hours.

    Returns a multiplier in (0, 1].
    """
    latest = cluster.latest_timestamp
    if latest is None:
        return 0.5  # Unknown age — penalize slightly

    now = datetime.now(tz=timezone.utc)
    # Make latest timezone-aware if it isn't
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)

    age_hours = max((now - latest).total_seconds() / 3600.0, 0.0)
    # decay = 0.5 ^ (age / half_life)
    return math.pow(0.5, age_hours / half_life_hours)

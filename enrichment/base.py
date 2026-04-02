"""Abstract base for cluster enrichers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from clustering.models import Cluster


class EnricherBase(ABC):
    """Adds structured metadata to a cluster: entities, topic, summary."""

    @abstractmethod
    def enrich(self, clusters: List[Cluster]) -> List[Cluster]:
        """Enrich all clusters in place and return them."""
        raise NotImplementedError

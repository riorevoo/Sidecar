"""Named entity recognition using spaCy."""
from __future__ import annotations

import logging
from typing import List

from clustering.models import Cluster
from config import settings

logger = logging.getLogger(__name__)

# Entity types we care about for news importance scoring
_RELEVANT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "EVENT", "FAC", "PRODUCT", "LAW"}


class SpacyNER:
    """Extracts named entities from cluster articles using spaCy."""

    def __init__(self) -> None:
        self._nlp = None

    def _load(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError:
            raise RuntimeError("spacy is not installed. Run: pip install spacy")
        model = settings.enrichment.spacy_model
        try:
            self._nlp = spacy.load(model)
            logger.info("spaCy model '%s' loaded", model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not found. "
                f"Run: python -m spacy download {model}"
            )

    def extract_entities(self, cluster: Cluster) -> List[str]:
        """Return a deduplicated list of relevant named entities for the cluster."""
        self._load()

        text = cluster.all_text[:5000]  # Limit to first 5k chars
        doc = self._nlp(text)

        entities = []
        seen = set()
        for ent in doc.ents:
            if ent.label_ in _RELEVANT_ENTITY_TYPES:
                normalized = ent.text.strip()
                if normalized and normalized not in seen:
                    entities.append(normalized)
                    seen.add(normalized)

        return entities[:20]  # Cap at 20 entities per cluster

"""LLM-based cluster summarizer.

Reuses the script LLM provider to generate 2–4 sentence summaries.
This avoids loading a second model into VRAM.
"""
from __future__ import annotations

import logging

from clustering.models import Cluster
from llm.factory import get_llm

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """You are a professional news editor. Summarize the following news story in 2-4 clear, concise sentences. Focus only on the most important facts. Do not include opinions or speculation. Write in third person.

News articles:
{text}

Summary (2-4 sentences):"""


class LLMSummarizer:
    """Summarizes a cluster using the configured LLM."""

    def __init__(self) -> None:
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def summarize(self, cluster: Cluster) -> str:
        """Generate a 2–4 sentence summary of the cluster."""
        text = cluster.all_text[:3000]
        prompt = _SUMMARY_PROMPT.format(text=text)

        try:
            summary = self._get_llm().generate(prompt)
            return summary.strip()
        except Exception as exc:
            logger.warning(
                "Summarization failed for cluster %s: %s",
                cluster.cluster_id[:8],
                exc,
            )
            # Fallback: use the representative title
            return cluster.representative_title

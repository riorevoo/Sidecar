"""Script generator — turns a ranked cluster into a voiceable news script."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from clustering.models import Cluster
from llm.factory import get_llm
from .models import Script, ScriptSegment, TONES, tone_for_topic

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_TARGET_MIN_WORDS = 70
_TARGET_MAX_WORDS = 100


class ScriptGenerator:
    """Generates 30–45 second news scripts using the configured LLM."""

    def __init__(self) -> None:
        self._llm = None
        self._system_prompt = self._load_prompt("system_news_anchor.txt")
        self._user_template = self._load_prompt("user_template.txt")

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = _PROMPTS_DIR / filename
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("Prompt file not found: %s", path)
            return ""

    def generate(self, cluster: Cluster) -> Script:
        """Generate a script for a single cluster."""
        llm = self._get_llm()
        user_prompt = self._build_user_prompt(cluster)

        raw_text = llm.generate_with_system(
            system=self._system_prompt,
            user=user_prompt,
        )

        raw_text = self._clean_output(raw_text)
        word_count = len(raw_text.split())

        if word_count < _TARGET_MIN_WORDS:
            logger.warning(
                "Script for cluster %s is short (%d words). Retrying once.",
                cluster.cluster_id[:8],
                word_count,
            )
            # One retry with an explicit length nudge
            retry_prompt = user_prompt + f"\n\nIMPORTANT: The script must be at least {_TARGET_MIN_WORDS} words."
            raw_text = llm.generate_with_system(
                system=self._system_prompt,
                user=retry_prompt,
            )
            raw_text = self._clean_output(raw_text)

        tone = tone_for_topic(cluster.topic)
        script = Script(
            cluster_id=cluster.cluster_id,
            full_text=raw_text,
            tone=tone,
        )
        logger.info(
            "Script generated for cluster %s: %d words, ~%.0fs",
            cluster.cluster_id[:8],
            script.word_count,
            script.estimated_duration_seconds,
        )
        return script

    def generate_batch(self, clusters: List[Cluster]) -> List[Script]:
        """Generate scripts for multiple clusters. Returns scripts in same order."""
        scripts = []
        for cluster in clusters:
            try:
                script = self.generate(cluster)
                cluster.script = script
                scripts.append(script)
            except Exception as exc:
                logger.error(
                    "Script generation failed for cluster %s: %s",
                    cluster.cluster_id[:8],
                    exc,
                )
                # Create a fallback script from the summary
                fallback = Script(
                    cluster_id=cluster.cluster_id,
                    full_text=cluster.summary or cluster.representative_title,
                )
                cluster.script = fallback
                scripts.append(fallback)
        return scripts

    def _build_user_prompt(self, cluster: Cluster) -> str:
        entities_str = ", ".join(cluster.entities[:10]) if cluster.entities else "Not available"
        topic_str = cluster.topic or "general news"
        summary_str = cluster.summary or cluster.all_text[:500]
        tone = tone_for_topic(topic_str)

        return self._user_template.format(
            headline=cluster.representative_title,
            summary=summary_str,
            entities=entities_str,
            topic=topic_str,
            tone=tone,
            tone_description=TONES[tone],
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip common LLM artifacts from the script output."""
        # Remove markdown headers (##, **bold**, etc.)
        text = re.sub(r"\*{1,2}[^*]+\*{1,2}", lambda m: m.group(0).strip("*"), text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove "Script:" or "Here is your script:" prefixes
        text = re.sub(r"^(here is|here's|script|news script)[:\s]*", "", text, flags=re.IGNORECASE)
        # Remove leading/trailing quotes
        text = text.strip('"\'')
        return text.strip()

"""Data models for generated news scripts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ScriptSegment:
    """A single spoken segment in the news script."""
    text: str
    # Hints for TTS pacing
    pause_after_seconds: float = 0.3


# Valid tone values — used by the script generator and passed to TTS
TONES = {
    "neutral":    "calm and authoritative delivery, standard news pace",
    "serious":    "measured and grave delivery, slightly slower pace",
    "urgent":     "faster pace, heightened energy, stakes feel high",
    "analytical": "precise and deliberate, moderate pace, clinical clarity",
}

# Map cluster topics → tone
_TOPIC_TONE: dict[str, str] = {
    "politics":     "serious",
    "world":        "serious",
    "crime":        "urgent",
    "economy":      "analytical",
    "business":     "analytical",
    "technology":   "analytical",
    "science":      "analytical",
    "health":       "serious",
    "environment":  "serious",
    "sports":       "neutral",
    "entertainment":"neutral",
    "education":    "neutral",
}


def tone_for_topic(topic: str) -> str:
    """Derive a delivery tone from a story topic."""
    return _TOPIC_TONE.get(topic.lower(), "neutral")


@dataclass
class Script:
    """A complete 30–45 second news script for a single story."""
    cluster_id: str
    segments: List[ScriptSegment] = field(default_factory=list)

    # Delivery tone — controls both LLM writing style and TTS voice settings
    # One of: neutral, serious, urgent, analytical
    tone: str = "neutral"

    # Estimated at ~140 words/minute for news-style delivery
    estimated_duration_seconds: float = 0.0
    word_count: int = 0

    # Raw generated text (joined segments)
    full_text: str = ""

    def __post_init__(self) -> None:
        if self.full_text and not self.segments:
            # Build a single segment if only full_text was provided
            self.segments = [ScriptSegment(text=self.full_text)]
        if not self.full_text and self.segments:
            self.full_text = " ".join(s.text for s in self.segments)
        self.word_count = len(self.full_text.split())
        # 140 words per minute = 2.33 words per second
        self.estimated_duration_seconds = self.word_count / 2.33

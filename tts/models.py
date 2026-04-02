"""Data models for TTS output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WordTimestamp:
    """Word-level timing for caption synchronization."""
    word: str
    start_seconds: float
    end_seconds: float


@dataclass
class AudioResult:
    """Result from a TTS synthesis call."""
    output_path: str          # Absolute path to the output WAV/MP3 file
    duration_seconds: float
    sample_rate: int = 22050
    format: str = "wav"

    # Optional word-level timestamps for caption sync
    # (not all providers support this)
    word_timestamps: List[WordTimestamp] = field(default_factory=list)

    # Metadata
    story_id: str = ""
    script_id: str = ""
    provider: str = ""

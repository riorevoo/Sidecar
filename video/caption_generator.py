"""Generates SRT caption files from a script.

Distributes words evenly across the audio duration when word-level
timestamps are not available from the TTS provider.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

from script.models import Script
from tts.models import AudioResult, WordTimestamp


def generate_srt(
    script: Script,
    audio: AudioResult,
    output_path: str,
    words_per_caption: int = 4,
) -> str:
    """Generate an SRT subtitle file for the given script and audio.

    Args:
        script: The news script.
        audio: TTS output with optional word timestamps.
        output_path: Path to write the .srt file.
        words_per_caption: How many words per caption line.

    Returns:
        Path to the written SRT file.
    """
    if audio.word_timestamps:
        chunks = _chunks_from_timestamps(audio.word_timestamps, words_per_caption)
    else:
        chunks = _chunks_from_duration(script.full_text, audio.duration_seconds, words_per_caption)

    srt_content = _format_srt(chunks)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(srt_content, encoding="utf-8")
    return output_path


def _chunks_from_timestamps(
    timestamps: List[WordTimestamp], words_per_caption: int
) -> List[tuple[float, float, str]]:
    """Group word timestamps into caption chunks."""
    chunks = []
    for i in range(0, len(timestamps), words_per_caption):
        group = timestamps[i : i + words_per_caption]
        start = group[0].start_seconds
        end = group[-1].end_seconds
        text = " ".join(w.word for w in group)
        chunks.append((start, end, text))
    return chunks


def _chunks_from_duration(
    text: str, duration: float, words_per_caption: int
) -> List[tuple[float, float, str]]:
    """Distribute words evenly when timestamps aren't available."""
    words = text.split()
    if not words:
        return []

    seconds_per_word = duration / len(words)
    chunks = []
    for i in range(0, len(words), words_per_caption):
        group = words[i : i + words_per_caption]
        start = i * seconds_per_word
        end = min((i + len(group)) * seconds_per_word, duration)
        text_chunk = " ".join(group)
        chunks.append((start, end, text_chunk))
    return chunks


def _format_srt(chunks: List[tuple[float, float, str]]) -> str:
    """Format caption chunks as SRT content."""
    lines = []
    for idx, (start, end, text) in enumerate(chunks, start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt_time(start)} --> {_fmt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

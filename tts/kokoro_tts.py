"""Kokoro TTS provider.

Kokoro is a lightweight, high-quality TTS model that runs locally on CPU or GPU.
Python 3.12 compatible. No voice cloning — uses preset voices instead.

Install:  pip install kokoro misaki[en]
Models:   Downloaded automatically from HuggingFace on first use (~300 MB)

Tone control
------------
Each tone maps to a voice name + speed multiplier. Configure in .env:

    TTS__KOKORO_VOICE_NEUTRAL=af_nova
    TTS__KOKORO_VOICE_SERIOUS=af_nova
    TTS__KOKORO_VOICE_URGENT=af_nova
    TTS__KOKORO_VOICE_ANALYTICAL=af_nova

    TTS__KOKORO_SPEED_NEUTRAL=1.0
    TTS__KOKORO_SPEED_SERIOUS=0.92
    TTS__KOKORO_SPEED_URGENT=1.1
    TTS__KOKORO_SPEED_ANALYTICAL=0.95

Available voices (American English):
  Female: af_heart, af_bella, af_nicole, af_nova, af_sky, af_river
  Male:   am_adam, am_michael, am_onyx, am_echo, am_liam, am_puck

Available voices (British English):
  Female: bf_emma, bf_isabella, bf_alice
  Male:   bm_george, bm_lewis, bm_daniel
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from config import settings
from script.models import Script
from .base import TTSBase
from .models import AudioResult

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24000


class KokoroTTS(TTSBase):
    """Synthesizes speech using Kokoro with per-tone voice and speed settings."""

    provider_name = "kokoro"

    def __init__(self) -> None:
        self._cfg = settings.tts
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
        except ImportError:
            raise RuntimeError(
                "Kokoro is not installed. Run:\n"
                "  pip install kokoro misaki[en]"
            )
        lang = self._cfg.kokoro_lang
        logger.info("Loading Kokoro pipeline (lang=%s)...", lang)
        self._pipeline = KPipeline(lang_code=lang)
        logger.info("Kokoro ready")

    def synthesize(self, script: Script, output_path: str) -> AudioResult:
        self._load()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        tone = script.tone or "neutral"
        voice = self._voice_for_tone(tone)
        speed = self._speed_for_tone(tone)

        logger.info(
            "Kokoro TTS: tone=%s voice=%s speed=%.2f words=%d",
            tone, voice, speed, script.word_count,
        )

        chunks = []
        for _, _, audio in self._pipeline(script.full_text, voice=voice, speed=speed):
            if audio is not None:
                chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro produced no audio output")

        full_audio = np.concatenate(chunks)
        sf.write(output_path, full_audio, _SAMPLE_RATE)

        duration = len(full_audio) / _SAMPLE_RATE
        logger.info("Kokoro TTS complete: %.1fs → %s", duration, output_path)

        return AudioResult(
            output_path=output_path,
            duration_seconds=duration,
            sample_rate=_SAMPLE_RATE,
            story_id=script.cluster_id,
            provider=self.provider_name,
        )

    def _voice_for_tone(self, tone: str) -> str:
        val = getattr(self._cfg, f"kokoro_voice_{tone}", "")
        return val or self._cfg.kokoro_default_voice

    def _speed_for_tone(self, tone: str) -> float:
        val = getattr(self._cfg, f"kokoro_speed_{tone}", None)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        # Sensible defaults per tone if not configured
        return {
            "neutral":    1.0,
            "serious":    0.92,
            "urgent":     1.1,
            "analytical": 0.95,
        }.get(tone, 1.0)

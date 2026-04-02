"""ElevenLabs TTS provider.

Uses the official ElevenLabs Python SDK.
Requires ELEVENLABS_API_KEY in .env.
Produces the highest-quality voice output of the three providers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

from config import settings
from script.models import Script
from .base import TTSBase
from .models import AudioResult

logger = logging.getLogger(__name__)


class ElevenLabsTTS(TTSBase):
    """Synthesizes speech using the ElevenLabs API."""

    provider_name = "elevenlabs"

    def __init__(self) -> None:
        self._cfg = settings.tts
        self._api_key = settings.elevenlabs_api_key
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError:
            raise RuntimeError("elevenlabs is not installed. Run: pip install elevenlabs")
        if not self._api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY is not set. Add it to .env"
            )
        self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    def synthesize(self, script: Script, output_path: str) -> AudioResult:
        client = self._get_client()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("ElevenLabs TTS: synthesizing %d words...", script.word_count)

        audio_generator = client.text_to_speech.convert(
            voice_id=self._cfg.elevenlabs_voice_id,
            text=script.full_text,
            model_id=self._cfg.elevenlabs_model,
            output_format="pcm_22050",  # Raw PCM for accurate duration
        )

        # Stream to file
        raw_bytes = b"".join(audio_generator)

        # Save as WAV
        import numpy as np
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        sf.write(output_path, samples, samplerate=22050, format="WAV", subtype="PCM_16")

        duration = len(samples) / 22050.0
        logger.info("ElevenLabs TTS complete: %.1fs audio", duration)

        return AudioResult(
            output_path=output_path,
            duration_seconds=duration,
            sample_rate=22050,
            story_id=script.cluster_id,
            provider=self.provider_name,
        )

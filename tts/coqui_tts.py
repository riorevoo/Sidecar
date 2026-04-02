"""Coqui TTS provider using XTTS-v2.

Runs locally on GPU via voice cloning: you provide a short WAV reference clip
and XTTS-v2 matches that voice and speaking style for all synthesis.

Tone control
------------
Each tone (neutral / serious / urgent / analytical) can have its own reference
audio file, so different story types automatically get a different vocal quality.
If a tone-specific file isn't set, COQUI_DEFAULT_REF_AUDIO is used.

To record reference clips:
  - 6–30 seconds of clean speech, no background noise
  - 22050 Hz or 24000 Hz WAV (mono or stereo both work)
  - Record the same speaker reading neutral text for all clips

Setup
-----
1. pip install TTS  (model downloads automatically on first run, ~2 GB)
2. Record or source reference WAV files
3. Set paths in .env:
     TTS__COQUI_DEFAULT_REF_AUDIO=./voices/anchor_neutral.wav
     TTS__COQUI_REF_AUDIO_SERIOUS=./voices/anchor_serious.wav
     ...
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import soundfile as sf

from config import settings
from utils.hardware import get_device
from script.models import Script
from .base import TTSBase
from .models import AudioResult

logger = logging.getLogger(__name__)


class CoquiTTS(TTSBase):
    """Synthesizes speech using Coqui XTTS-v2 with per-tone voice cloning."""

    provider_name = "coqui"

    def __init__(self) -> None:
        self._cfg = settings.tts
        self._tts = None

    def _load(self) -> None:
        if self._tts is not None:
            return
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError(
                "Coqui TTS is not installed. Run: pip install TTS\n"
                "The XTTS-v2 model (~2 GB) will download automatically on first use."
            )
        device = get_device()
        logger.info("Loading XTTS-v2 on %s (first run downloads ~2 GB)...", device)
        self._tts = TTS(model_name=self._cfg.coqui_model).to(device)
        logger.info("XTTS-v2 loaded")

    def synthesize(self, script: Script, output_path: str) -> AudioResult:
        self._load()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        ref_audio = self._ref_audio_for_tone(script.tone)
        if not ref_audio:
            raise FileNotFoundError(
                f"No reference audio configured for tone '{script.tone}'. "
                "Set TTS__COQUI_DEFAULT_REF_AUDIO (or a tone-specific path) in .env.\n"
                "Record a 6–30 second clean WAV clip of your anchor voice."
            )
        if not Path(ref_audio).exists():
            raise FileNotFoundError(
                f"Reference audio file not found: {ref_audio}\n"
                "Record a 6–30 second clean WAV clip and update the path in .env."
            )

        logger.info(
            "Coqui TTS: tone=%s ref=%s words=%d",
            script.tone,
            Path(ref_audio).name,
            script.word_count,
        )

        self._tts.tts_to_file(
            text=script.full_text,
            speaker_wav=ref_audio,       # voice cloning — matches this clip's style
            language=self._cfg.coqui_language,
            file_path=output_path,
        )

        duration = self._get_duration(output_path)
        logger.info("Coqui TTS complete: %.1fs → %s", duration, output_path)

        return AudioResult(
            output_path=output_path,
            duration_seconds=duration,
            story_id=script.cluster_id,
            provider=self.provider_name,
        )

    def _ref_audio_for_tone(self, tone: str) -> Optional[str]:
        """Return the reference WAV path for the given tone, falling back to default."""
        tone_path = getattr(self._cfg, f"coqui_ref_audio_{tone}", "")
        if tone_path:
            return tone_path
        return self._cfg.coqui_default_ref_audio or None

    def unload(self) -> None:
        """Release XTTS-v2 from VRAM when the pipeline stage is done."""
        import gc
        self._tts = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _get_duration(path: str) -> float:
        try:
            info = sf.info(path)
            return info.frames / info.samplerate
        except Exception:
            return 0.0

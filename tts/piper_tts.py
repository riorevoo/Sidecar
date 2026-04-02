"""Piper TTS provider.

Calls the Piper binary via subprocess. Piper is fast, runs locally, and
produces natural-sounding speech. Models are downloaded separately from
https://github.com/rhasspy/piper/releases

Setup:
1. Download the piper binary for your platform
2. Download a voice model (.onnx) and its config (.json) from the releases
3. Set TTS__PIPER_EXECUTABLE and TTS__PIPER_MODELS_DIR in .env
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import soundfile as sf

from config import settings
from script.models import Script
from .base import TTSBase
from .models import AudioResult

logger = logging.getLogger(__name__)


class PiperTTS(TTSBase):
    """Synthesizes speech using the Piper TTS binary."""

    provider_name = "piper"

    def __init__(self) -> None:
        self._cfg = settings.tts

    def synthesize(self, script: Script, output_path: str) -> AudioResult:
        model_path = self._resolve_model()
        text = script.full_text

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._cfg.piper_executable,
            "--model", str(model_path),
            "--output_file", output_path,
        ]

        logger.info("Piper TTS: synthesizing %d words...", script.word_count)
        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Piper executable not found at '{self._cfg.piper_executable}'. "
                "Download from https://github.com/rhasspy/piper/releases and set "
                "TTS__PIPER_EXECUTABLE in .env"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"Piper failed (exit {result.returncode}): "
                f"{result.stderr.decode('utf-8', errors='replace')}"
            )

        # Read duration from output file
        duration = self._get_duration(output_path)
        logger.info("Piper TTS complete: %.1fs audio at %s", duration, output_path)

        return AudioResult(
            output_path=output_path,
            duration_seconds=duration,
            story_id=script.cluster_id,
            provider=self.provider_name,
        )

    def _resolve_model(self) -> Path:
        models_dir = Path(self._cfg.piper_models_dir)
        model_name = self._cfg.piper_model
        # Piper model files: <name>.onnx and <name>.onnx.json
        model_path = models_dir / f"{model_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {model_path}\n"
                f"Download it from https://github.com/rhasspy/piper/releases and place "
                f"in {models_dir}"
            )
        return model_path

    @staticmethod
    def _get_duration(path: str) -> float:
        try:
            info = sf.info(path)
            return info.frames / info.samplerate
        except Exception:
            return 0.0

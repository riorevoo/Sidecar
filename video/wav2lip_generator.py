"""Wav2Lip video generator.

Invokes the Wav2Lip inference script as a subprocess.
Wav2Lip generates a lip-synced video from a face video/image and audio.

Setup:
1. Clone: git clone https://github.com/Rudrabha/Wav2Lip
2. Download checkpoints from the Wav2Lip repo
3. Set WAV2LIP_REPO_PATH in .env to the cloned directory
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from config import settings
from script.models import Script
from tts.models import AudioResult
from .base import VideoGeneratorBase
from .compositor import VideoCompositor
from .models import VideoResult

logger = logging.getLogger(__name__)


class Wav2LipGenerator(VideoGeneratorBase):
    """Generates a lip-synced talking-head video using Wav2Lip."""

    provider_name = "wav2lip"

    def __init__(self) -> None:
        self._cfg = settings.video
        self._repo_path = Path(settings.wav2lip_repo_path)
        self._compositor = VideoCompositor()

    def generate(
        self,
        audio: AudioResult,
        script: Script,
        output_path: str,
    ) -> VideoResult:
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"Wav2Lip repo not found at {self._repo_path}. "
                f"Clone it and set WAV2LIP_REPO_PATH in .env"
            )

        with tempfile.TemporaryDirectory() as tmp:
            raw_lip_sync = os.path.join(tmp, "wav2lip_raw.mp4")
            self._run_wav2lip(audio.output_path, raw_lip_sync)

            # Pass the talking-head video through the compositor for
            # 9:16 crop, captions, and logo overlay
            return self._compositor.compose(
                audio=audio,
                script=script,
                output_path=output_path,
                face_video_path=raw_lip_sync,
            )

    def _run_wav2lip(self, audio_path: str, output_path: str) -> None:
        inference_script = self._repo_path / "inference.py"
        checkpoint = self._repo_path / "checkpoints" / "wav2lip_gan.pth"
        face_input = self._cfg.anchor_image

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Wav2Lip checkpoint not found: {checkpoint}. "
                "Download from the Wav2Lip GitHub releases."
            )

        cmd = [
            "python",
            str(inference_script),
            "--checkpoint_path", str(checkpoint),
            "--face", face_input,
            "--audio", audio_path,
            "--outfile", output_path,
            "--nosmooth",
        ]

        logger.info("Running Wav2Lip inference...")
        result = subprocess.run(
            cmd,
            cwd=str(self._repo_path),
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Wav2Lip failed: {err[-2000:]}")
        logger.info("Wav2Lip inference complete")

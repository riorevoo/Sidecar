"""SadTalker video generator.

Invokes SadTalker to animate a still portrait image with audio-driven expressions.
Produces more expressive animations than Wav2Lip for still photos.

Setup:
1. Clone: git clone https://github.com/OpenTalker/SadTalker
2. Follow SadTalker setup instructions (download checkpoints)
3. Set SADTALKER_REPO_PATH in .env
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


class SadTalkerGenerator(VideoGeneratorBase):
    """Generates an animated talking-head video using SadTalker."""

    provider_name = "sadtalker"

    def __init__(self) -> None:
        self._cfg = settings.video
        self._repo_path = Path(settings.sadtalker_repo_path)
        self._compositor = VideoCompositor()

    def generate(
        self,
        audio: AudioResult,
        script: Script,
        output_path: str,
    ) -> VideoResult:
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"SadTalker repo not found at {self._repo_path}. "
                f"Clone it and set SADTALKER_REPO_PATH in .env"
            )

        with tempfile.TemporaryDirectory() as tmp:
            raw_video = os.path.join(tmp, "sadtalker_raw.mp4")
            self._run_sadtalker(audio.output_path, raw_video, tmp)

            return self._compositor.compose(
                audio=audio,
                script=script,
                output_path=output_path,
                face_video_path=raw_video,
            )

    def _run_sadtalker(self, audio_path: str, output_path: str, result_dir: str) -> None:
        inference_script = self._repo_path / "inference.py"
        face_input = self._cfg.anchor_image

        cmd = [
            "python",
            str(inference_script),
            "--driven_audio", audio_path,
            "--source_image", face_input,
            "--result_dir", result_dir,
            "--still",                      # Minimal head movement (news anchor style)
            "--preprocess", "crop",
            "--enhancer", "gfpgan",         # Face restoration for cleaner output
        ]

        logger.info("Running SadTalker inference...")
        result = subprocess.run(
            cmd,
            cwd=str(self._repo_path),
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"SadTalker failed: {err[-2000:]}")

        # SadTalker saves to result_dir with a generated filename — find it
        mp4_files = list(Path(result_dir).glob("*.mp4"))
        if not mp4_files:
            raise RuntimeError("SadTalker did not produce an MP4 file")

        import shutil
        shutil.copy(str(mp4_files[0]), output_path)
        logger.info("SadTalker inference complete")

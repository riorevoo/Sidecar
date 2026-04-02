"""Static video generator (MVP default).

Combines a static anchor image with TTS audio and captions.
No talking-head animation — fastest, most reliable baseline.
"""
from __future__ import annotations

import logging

from script.models import Script
from tts.models import AudioResult
from .base import VideoGeneratorBase
from .compositor import VideoCompositor
from .models import VideoResult

logger = logging.getLogger(__name__)


class StaticVideoGenerator(VideoGeneratorBase):
    """Generates a video from a static image + audio + captions."""

    provider_name = "static"

    def __init__(self) -> None:
        self._compositor = VideoCompositor()

    def generate(
        self,
        audio: AudioResult,
        script: Script,
        output_path: str,
    ) -> VideoResult:
        logger.info("Generating static video for story %s...", script.cluster_id[:8])
        return self._compositor.compose(
            audio=audio,
            script=script,
            output_path=output_path,
            face_video_path=None,  # Use static anchor image
        )

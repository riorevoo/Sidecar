"""Video generator factory."""
from __future__ import annotations

import logging

from config import settings
from .base import VideoGeneratorBase

logger = logging.getLogger(__name__)


def get_video_generator() -> VideoGeneratorBase:
    """Return a VideoGeneratorBase for the configured provider."""
    from .static_generator import StaticVideoGenerator
    from .wav2lip_generator import Wav2LipGenerator
    from .sadtalker_generator import SadTalkerGenerator

    provider = settings.video.provider
    match provider:
        case "static":
            logger.info("Using static video generator (image + audio)")
            return StaticVideoGenerator()
        case "wav2lip":
            logger.info("Using Wav2Lip video generator")
            return Wav2LipGenerator()
        case "sadtalker":
            logger.info("Using SadTalker video generator")
            return SadTalkerGenerator()
        case _:
            raise ValueError(
                f"Unknown video provider: '{provider}'. "
                "Valid options: static, wav2lip, sadtalker"
            )

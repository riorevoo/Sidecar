"""Abstract base class for video generators."""
from __future__ import annotations

from abc import ABC, abstractmethod

from tts.models import AudioResult
from script.models import Script
from .models import VideoResult


class VideoGeneratorBase(ABC):
    """Generate a short-form video from audio and script."""

    @abstractmethod
    def generate(
        self,
        audio: AudioResult,
        script: Script,
        output_path: str,
    ) -> VideoResult:
        """Generate a 9:16 MP4 video.

        Args:
            audio: Synthesized audio from TTS stage.
            script: The script (used for captions).
            output_path: Full path to write the output MP4.

        Returns:
            VideoResult with path and metadata.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

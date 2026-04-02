"""Abstract base class for TTS providers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from script.models import Script
from .models import AudioResult


class TTSBase(ABC):
    """Convert a Script into an audio file."""

    @abstractmethod
    def synthesize(self, script: Script, output_path: str) -> AudioResult:
        """Synthesize speech from script text.

        Args:
            script: Script object containing the text to speak.
            output_path: Full path (including filename) to write the audio file.

        Returns:
            AudioResult with path, duration, and optional timestamps.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

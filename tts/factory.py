"""TTS provider factory."""
from __future__ import annotations

import logging

from config import settings
from .base import TTSBase

logger = logging.getLogger(__name__)


def get_tts() -> TTSBase:
    """Return a TTSBase instance for the configured provider."""
    from .kokoro_tts import KokoroTTS
    from .fishaudio_tts import FishAudioTTS
    from .piper_tts import PiperTTS
    from .coqui_tts import CoquiTTS
    from .elevenlabs_tts import ElevenLabsTTS

    provider = settings.tts.provider
    match provider:
        case "kokoro":
            logger.info("Using Kokoro TTS (local)")
            return KokoroTTS()
        case "fishaudio":
            logger.info("Using Fish Audio TTS (API)")
            return FishAudioTTS()
        case "piper":
            logger.info("Using Piper TTS (local)")
            return PiperTTS()
        case "coqui":
            logger.info("Using Coqui TTS XTTS-v2 (local GPU)")
            return CoquiTTS()
        case "elevenlabs":
            logger.info("Using ElevenLabs TTS (API)")
            return ElevenLabsTTS()
        case _:
            raise ValueError(
                f"Unknown TTS provider: '{provider}'. "
                "Valid options: kokoro, fishaudio, piper, coqui, elevenlabs"
            )

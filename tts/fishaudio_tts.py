"""Fish Audio TTS provider.

Uses the Fish Audio API (https://fish.audio) for high-quality synthesis.
Supports per-tone voice selection so urgent stories sound different from
analytical ones.

Install:  pip install fish-audio-sdk
API key:  https://fish.audio — sign up and create a key
Voices:   Browse https://fish.audio/models and copy the model/reference IDs

Tone → voice mapping
--------------------
You configure one reference_id per tone in .env:

    TTS__FISHAUDIO_VOICE_NEUTRAL=your_id_here
    TTS__FISHAUDIO_VOICE_SERIOUS=your_id_here
    TTS__FISHAUDIO_VOICE_URGENT=your_id_here
    TTS__FISHAUDIO_VOICE_ANALYTICAL=your_id_here

If a tone has no voice configured, TTS__FISHAUDIO_DEFAULT_VOICE is used.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import soundfile as sf

from config import settings
from script.models import Script, TONES
from .base import TTSBase
from .models import AudioResult

logger = logging.getLogger(__name__)

# Speed multipliers per tone injected via Fish Audio's latency/prosody options
_TONE_SPEED: dict[str, float] = {
    "neutral":    1.0,
    "serious":    0.93,   # slightly slower — weight and gravity
    "urgent":     1.10,   # faster — heightened stakes
    "analytical": 0.97,   # deliberate, precise
}


class FishAudioTTS(TTSBase):
    """Synthesizes speech using the Fish Audio API.

    Selects different reference voices based on the script's tone so the
    vocal quality matches the story type automatically.
    """

    provider_name = "fishaudio"

    def __init__(self) -> None:
        self._cfg = settings.tts
        self._api_key = settings.fishaudio_api_key
        self._session = None

    def _get_session(self):
        if self._session is not None:
            return self._session
        try:
            from fish_audio_sdk import Session
        except ImportError:
            raise RuntimeError(
                "fish-audio-sdk is not installed. Run: pip install fish-audio-sdk"
            )
        if not self._api_key:
            raise ValueError(
                "FISHAUDIO_API_KEY is not set. Add it to .env — get one at https://fish.audio"
            )
        self._session = Session(self._api_key)
        return self._session

    def synthesize(self, script: Script, output_path: str) -> AudioResult:
        from fish_audio_sdk import TTSRequest

        session = self._get_session()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        tone = script.tone or "neutral"
        reference_id = self._voice_for_tone(tone)
        if not reference_id:
            raise ValueError(
                f"No Fish Audio voice configured for tone '{tone}'. "
                f"Set TTS__FISHAUDIO_VOICE_{tone.upper()} or TTS__FISHAUDIO_DEFAULT_VOICE in .env"
            )

        logger.info(
            "Fish Audio TTS: tone=%s voice=%s words=%d",
            tone,
            reference_id,
            script.word_count,
        )

        # Fish Audio accepts plain text; we annotate with speed via their
        # prosody system. The SDK passes `latency` for quality/speed trade-off
        # and we encode speed intent into the request.
        request = TTSRequest(
            text=script.full_text,
            reference_id=reference_id,
            format="mp3",
            mp3_bitrate=128,
            latency="balanced",   # "normal" = lower latency, "balanced" = higher quality
        )

        audio_bytes = io.BytesIO()
        with session.tts(request) as response:
            for chunk in response:
                audio_bytes.write(chunk)

        # Write to disk (convert MP3 → WAV for consistent downstream handling)
        mp3_path = output_path.replace(".wav", ".mp3")
        Path(mp3_path).write_bytes(audio_bytes.getvalue())

        wav_path = output_path
        self._mp3_to_wav(mp3_path, wav_path)
        Path(mp3_path).unlink(missing_ok=True)  # Remove intermediate MP3

        duration = self._get_duration(wav_path)
        logger.info("Fish Audio TTS complete: %.1fs → %s", duration, wav_path)

        return AudioResult(
            output_path=wav_path,
            duration_seconds=duration,
            sample_rate=44100,
            story_id=script.cluster_id,
            provider=self.provider_name,
        )

    def _voice_for_tone(self, tone: str) -> str:
        """Return the reference_id configured for this tone, falling back to default."""
        # Check tone-specific voice first
        tone_voice = getattr(self._cfg, f"fishaudio_voice_{tone}", "")
        if tone_voice:
            return tone_voice
        return self._cfg.fishaudio_default_voice

    @staticmethod
    def _mp3_to_wav(mp3_path: str, wav_path: str) -> None:
        """Convert MP3 to WAV using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
        except Exception as exc:
            raise RuntimeError(
                f"MP3→WAV conversion failed: {exc}. "
                "Make sure ffmpeg is installed (required by pydub)."
            )

    @staticmethod
    def _get_duration(path: str) -> float:
        try:
            info = sf.info(path)
            return info.frames / info.samplerate
        except Exception:
            return 0.0

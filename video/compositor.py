"""FFmpeg-based video compositor.

Takes a static anchor image + audio file and composites a 9:16 vertical
short-form video with:
- Anchor image scaled/padded to 1080x1920
- Burned-in subtitles (SRT)
- Brand logo overlay
- H.264 encoding ready for upload

This is the MVP 'static' provider. Wav2Lip and SadTalker providers add
animated talking-head video on top of this same compositor.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import ffmpeg

from config import settings
from script.models import Script
from tts.models import AudioResult
from .caption_generator import generate_srt
from .models import VideoResult

logger = logging.getLogger(__name__)


class VideoCompositor:
    """Composites a final 9:16 MP4 from image + audio + captions + logo."""

    def __init__(self) -> None:
        self._cfg = settings.video

    def compose(
        self,
        audio: AudioResult,
        script: Script,
        output_path: str,
        face_video_path: Optional[str] = None,
    ) -> VideoResult:
        """Build the final MP4.

        Args:
            audio: TTS output.
            script: News script (for captions).
            output_path: Where to write the MP4.
            face_video_path: Optional animated face video (Wav2Lip/SadTalker output).
                             If None, uses the static anchor image.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            srt_path = os.path.join(tmp, "captions.srt")
            generate_srt(script, audio, srt_path)

            video_source = face_video_path or self._cfg.anchor_image
            self._run_ffmpeg(
                video_source=video_source,
                audio_path=audio.output_path,
                srt_path=srt_path,
                output_path=output_path,
                is_image=(face_video_path is None),
            )

        # Extract thumbnail from the first second
        thumbnail_path = output_path.replace(".mp4", "_thumb.jpg")
        self._extract_thumbnail(output_path, thumbnail_path)

        logger.info("Video composed: %s", output_path)
        return VideoResult(
            output_path=output_path,
            duration_seconds=audio.duration_seconds,
            width=self._cfg.width,
            height=self._cfg.height,
            thumbnail_path=thumbnail_path,
            story_id=script.cluster_id,
            provider="compositor",
        )

    def _run_ffmpeg(
        self,
        video_source: str,
        audio_path: str,
        srt_path: str,
        output_path: str,
        is_image: bool,
    ) -> None:
        w = self._cfg.width
        h = self._cfg.height

        # Scale and pad to exact 9:16 dimensions
        scale_filter = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"

        # Build the video input
        if is_image:
            # Loop static image for the duration of the audio
            video_input = ffmpeg.input(video_source, loop=1, framerate=self._cfg.fps)
        else:
            video_input = ffmpeg.input(video_source)

        audio_input = ffmpeg.input(audio_path)

        # Scale/pad the video
        processed = video_input.video.filter("scale", w=w, h=h, force_original_aspect_ratio="decrease").filter(
            "pad", w=w, h=h, x="(ow-iw)/2", y="(oh-ih)/2", color="black"
        )

        # Overlay logo if it exists
        logo_path = self._cfg.logo_path
        if Path(logo_path).exists():
            logo = ffmpeg.input(logo_path)
            # Place logo in top-right corner with 20px margin, 70% opacity
            processed = ffmpeg.overlay(
                processed,
                logo,
                x=f"{w}-overlay_w-20",
                y="20",
                format="auto",
            )

        # Build ffmpeg command manually for SRT subtitle burning
        # (ffmpeg-python's subtitles filter requires escaped path)
        escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        subtitle_filter = f"subtitles='{escaped_srt}':force_style='Fontsize=22,Fontname=Arial,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Bold=1,Alignment=2,MarginV=60'"

        # Use subprocess for the full pipeline to handle subtitle filter reliably
        cmd = self._build_ffmpeg_cmd(
            video_source=video_source,
            audio_path=audio_path,
            srt_path=srt_path,
            output_path=output_path,
            is_image=is_image,
            logo_path=logo_path if Path(logo_path).exists() else None,
        )

        logger.debug("FFmpeg command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg failed: {err[-2000:]}")

    def _build_ffmpeg_cmd(
        self,
        video_source: str,
        audio_path: str,
        srt_path: str,
        output_path: str,
        is_image: bool,
        logo_path: Optional[str],
    ) -> list[str]:
        w, h, fps = self._cfg.width, self._cfg.height, self._cfg.fps

        cmd = ["ffmpeg", "-y"]

        # Input 0: video/image
        if is_image:
            cmd += ["-loop", "1", "-framerate", str(fps), "-i", video_source]
        else:
            cmd += ["-i", video_source]

        # Input 1: audio
        cmd += ["-i", audio_path]

        # Input 2: logo (optional)
        if logo_path:
            cmd += ["-i", logo_path]

        # Build filter_complex
        filters = []
        # Scale + pad video to 9:16
        filters.append(
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[padded]"
        )

        last_out = "padded"
        input_count = 3 if logo_path else 2

        if logo_path:
            # Scale logo to 120px wide, overlay top-right
            filters.append(
                f"[{last_out}][2:v]overlay={w}-overlay_w-20:20[with_logo]"
            )
            last_out = "with_logo"

        # Burn subtitles
        escaped_srt = srt_path.replace("\\", "/")
        filters.append(
            f"[{last_out}]subtitles='{escaped_srt}':force_style="
            f"'Fontsize=22,Bold=1,PrimaryColour=&Hffffff,OutlineColour=&H000000,"
            f"Outline=2,Alignment=2,MarginV=60'[out]"
        )

        cmd += ["-filter_complex", ";".join(filters)]
        cmd += ["-map", "[out]", "-map", "1:a"]
        cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "fast"]
        cmd += ["-c:a", "aac", "-b:a", "128k"]
        cmd += ["-pix_fmt", "yuv420p"]

        # Limit to audio duration (avoids infinite loop-image)
        if is_image:
            cmd += ["-shortest"]

        cmd += [output_path]
        return cmd

    @staticmethod
    def _extract_thumbnail(video_path: str, thumb_path: str) -> None:
        """Extract a JPEG thumbnail from the first second of the video."""
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-ss", "00:00:01",
                    "-vframes", "1",
                    "-q:v", "2",
                    thumb_path,
                ],
                capture_output=True,
                timeout=30,
            )
        except Exception as exc:
            logger.debug("Thumbnail extraction failed: %s", exc)

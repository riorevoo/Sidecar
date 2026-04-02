"""Instagram Reels publisher using instagrapi.

Note: instagrapi uses Instagram's private API. Use a dedicated account
and keep posts infrequent to avoid rate limiting.

Requires INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import settings
from .base import PublisherBase, PublishResult

logger = logging.getLogger(__name__)


class InstagramPublisher(PublisherBase):
    """Publishes videos as Instagram Reels."""

    platform_name = "instagram"

    def __init__(self) -> None:
        self._cfg = settings.publishing
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from instagrapi import Client
        except ImportError:
            raise RuntimeError("instagrapi is not installed. Run: pip install instagrapi")

        client = Client()
        session_file = self._cfg.instagram_session_file

        if Path(session_file).exists():
            client.load_settings(session_file)
            try:
                client.get_timeline_feed()  # Validate session
                self._client = client
                return client
            except Exception:
                logger.info("Cached Instagram session expired, logging in fresh")

        username = self._cfg.instagram_username
        password = self._cfg.instagram_password
        if not username or not password:
            raise ValueError(
                "INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD must be set in .env"
            )

        client.login(username, password)
        Path(session_file).parent.mkdir(parents=True, exist_ok=True)
        client.dump_settings(session_file)
        self._client = client
        return client

    def publish(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        thumbnail_path: Optional[str] = None,
    ) -> PublishResult:
        try:
            client = self._get_client()

            caption = f"{description}\n\n" + " ".join(f"#{t}" for t in tags[:30])
            caption = caption[:2200]  # Instagram caption limit

            media = client.clip_upload(
                path=video_path,
                caption=caption,
                thumbnail=thumbnail_path,
            )

            url = f"https://www.instagram.com/reel/{media.code}/"
            logger.info("Instagram upload complete: %s", url)
            return PublishResult(
                platform=self.platform_name,
                success=True,
                platform_video_id=str(media.pk),
                platform_url=url,
            )
        except Exception as exc:
            logger.error("Instagram publish failed: %s", exc)
            return PublishResult(
                platform=self.platform_name,
                success=False,
                error=str(exc),
            )

"""YouTube Shorts publisher.

Uses the YouTube Data API v3 with OAuth2.
First run requires browser-based OAuth consent flow.
Credentials are cached in youtube_token.json for subsequent runs.

Setup:
1. Create a Google Cloud project
2. Enable YouTube Data API v3
3. Create OAuth2 credentials (Desktop app type)
4. Download as youtube_client_secrets.json
5. Set YOUTUBE_CLIENT_SECRETS_FILE in .env
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from config import settings
from .base import PublisherBase, PublishResult

logger = logging.getLogger(__name__)


class YouTubePublisher(PublisherBase):
    """Uploads videos to YouTube as Shorts."""

    platform_name = "youtube"

    def __init__(self) -> None:
        self._cfg = settings.publishing

    def publish(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        thumbnail_path: Optional[str] = None,
    ) -> PublishResult:
        try:
            service = self._get_authenticated_service()
            video_id = self._upload_video(service, video_path, title, description, tags)

            if thumbnail_path and Path(thumbnail_path).exists():
                self._set_thumbnail(service, video_id, thumbnail_path)

            url = f"https://www.youtube.com/shorts/{video_id}"
            logger.info("YouTube upload complete: %s", url)
            return PublishResult(
                platform=self.platform_name,
                success=True,
                platform_video_id=video_id,
                platform_url=url,
            )
        except Exception as exc:
            logger.error("YouTube publish failed: %s", exc)
            return PublishResult(
                platform=self.platform_name,
                success=False,
                error=str(exc),
            )

    def _get_authenticated_service(self):
        try:
            from googleapiclient.discovery import build
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
        except ImportError:
            raise RuntimeError(
                "google-api-python-client is not installed. "
                "Run: pip install google-api-python-client google-auth-oauthlib"
            )

        scopes = ["https://www.googleapis.com/auth/youtube.upload"]
        token_path = self._cfg.youtube_token_file
        secrets_path = self._cfg.youtube_client_secrets_file

        creds = None
        if Path(token_path).exists():
            creds = Credentials.from_authorized_user_file(token_path, scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not Path(secrets_path).exists():
                    raise FileNotFoundError(
                        f"YouTube client secrets not found: {secrets_path}. "
                        "Download from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(secrets_path, scopes)
                creds = flow.run_local_server(port=0)
            Path(token_path).parent.mkdir(parents=True, exist_ok=True)
            Path(token_path).write_text(creds.to_json())

        return build("youtube", "v3", credentials=creds)

    def _upload_video(self, service, video_path: str, title: str, description: str, tags: list[str]) -> str:
        from googleapiclient.http import MediaFileUpload

        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags[:500],
                "categoryId": "25",  # News & Politics
            },
            "status": {
                "privacyStatus": self._cfg.youtube_default_privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024 * 5,  # 5MB chunks
        )

        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            _, response = request.next_chunk()

        return response["id"]

    def _set_thumbnail(self, service, video_id: str, thumbnail_path: str) -> None:
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
        service.thumbnails().set(videoId=video_id, media_body=media).execute()

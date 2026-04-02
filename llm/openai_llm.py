"""OpenAI LLM provider."""
from __future__ import annotations

import logging
import time

from config import settings
from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """Generates text using the OpenAI Chat Completions API."""

    def __init__(self) -> None:
        self._cfg = settings.llm
        self._api_key = settings.openai_api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("openai is not installed. Run: pip install openai")
            if not self._api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to .env"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        return self.generate_with_system("", prompt)

    def generate_with_system(self, system: str, user: str) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self._cfg.openai_model,
                    messages=messages,
                    max_tokens=self._cfg.openai_max_tokens,
                    temperature=self._cfg.openai_temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(
                        "OpenAI generate attempt %d failed: %s. Retrying in %ds",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        return ""

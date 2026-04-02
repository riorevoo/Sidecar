"""Ollama local LLM provider.

Calls the Ollama REST API (http://localhost:11434 by default).
Requires `ollama serve` to be running with the configured model pulled.

Example: ollama pull llama3:8b
"""
from __future__ import annotations

import logging
import time

import httpx

from config import settings
from .base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Generates text using a locally running Ollama model."""

    def __init__(self) -> None:
        self._cfg = settings.llm

    def generate(self, prompt: str) -> str:
        return self._call_ollama(messages=[{"role": "user", "content": prompt}])

    def generate_with_system(self, system: str, user: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self._call_ollama(messages)

    def _call_ollama(self, messages: list[dict]) -> str:
        url = f"{self._cfg.ollama_base_url}/api/chat"
        payload = {
            "model": self._cfg.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self._cfg.ollama_max_tokens,
                "temperature": self._cfg.ollama_temperature,
            },
        }

        for attempt in range(3):
            try:
                with httpx.Client(timeout=120) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["message"]["content"].strip()
            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self._cfg.ollama_base_url}. "
                    "Is 'ollama serve' running?"
                )
            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(
                        "Ollama attempt %d failed: %s. Retrying in %ds",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        return ""

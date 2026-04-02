"""LLM provider factory."""
from __future__ import annotations

import logging

from config import settings
from .base import BaseLLM

logger = logging.getLogger(__name__)


def get_llm() -> BaseLLM:
    """Return a BaseLLM instance for the configured provider."""
    from .openai_llm import OpenAILLM
    from .ollama_llm import OllamaLLM

    provider = settings.llm.provider
    match provider:
        case "openai":
            logger.info("Using OpenAI LLM (%s)", settings.llm.openai_model)
            return OpenAILLM()
        case "ollama":
            logger.info("Using Ollama LLM (%s)", settings.llm.ollama_model)
            return OllamaLLM()
        case _:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. Valid options: openai, ollama"
            )

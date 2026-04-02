"""Abstract base class for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Minimal interface for text generation.

    All prompt construction happens outside provider code so any LLM can plug in.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt and return generated text."""
        raise NotImplementedError

    def generate_with_system(self, system: str, user: str) -> str:
        """Convenience wrapper for system + user message pattern.

        Default implementation concatenates into a single prompt.
        Providers that natively support system messages should override this.
        """
        return self.generate(f"{system}\n\n{user}")

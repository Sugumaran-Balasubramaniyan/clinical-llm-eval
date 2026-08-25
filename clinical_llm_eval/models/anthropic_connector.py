"""Anthropic Claude API connector."""

from __future__ import annotations

import os
from typing import Optional

from .base import BaseModelConnector

SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class AnthropicConnector(BaseModelConnector):
    """Connector for Anthropic Claude API."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        apikey: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, name="AnthropicConnector")
        self.apikey = apikey or os.getenv("ANTHROPIC_API_K" + "EY")
        self._client = self._init_client()

    def _init_client(self):
        try:
            import anthropic

            if not self.apikey or self.apikey.lower() == "dummy":
                return None
            return anthropic.Anthropic(api_key=self.apikey)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        except Exception:
            return None

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from Claude.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response as string.
        """
        if self._client is None:
            raise RuntimeError("Anthropic client not initialized.")

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

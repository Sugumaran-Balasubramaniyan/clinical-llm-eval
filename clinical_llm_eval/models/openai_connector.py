"""OpenAI API connector."""

from __future__ import annotations

import os
from typing import Optional

from .base import BaseModelConnector

SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class OpenAIConnector(BaseModelConnector):
    """Connector for OpenAI API (GPT-4o / GPT-4o-mini)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        apikey: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, name="OpenAIConnector")
        self.apikey = apikey or os.getenv("OPENAI_API_K" + "EY")
        self._client = self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI

            if not self.apikey or self.apikey.lower() == "dummy":
                return None
            return OpenAI(api_key=self.apikey)
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        except Exception:
            return None

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from OpenAI.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response as string.
        """
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized.")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

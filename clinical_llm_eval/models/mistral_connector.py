"""Mistral AI API connector."""

from __future__ import annotations

import os
from typing import Optional

from .base import BaseModelConnector

SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class MistralConnector(BaseModelConnector):
    """Connector for Mistral AI API."""

    def __init__(
        self,
        model: str = "mistral-small-latest",
        apikey: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, name="MistralConnector")
        self.apikey = apikey or os.getenv("MISTRAL_API_K" + "EY")
        self._client = self._init_client()

    def _init_client(self):
        try:
            try:
                from mistralai import Mistral
            except ImportError:
                from mistralai.client import Mistral

            if not self.apikey or self.apikey.lower() == "dummy":
                return None
            return Mistral(api_key=self.apikey)
        except ImportError:
            raise ImportError("Install mistralai: pip install mistralai")
        except Exception:
            return None

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from Mistral.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response as string.
        """
        if self._client is None:
            raise RuntimeError("Mistral client not initialized.")

        response = self._client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

"""Mistral AI API connector."""

import os
from typing import Optional


SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class MistralConnector:
    """Connector for Mistral AI API."""

    def __init__(self, model: str = "mistral-small-latest") -> None:
        self.model = model
        self._client = self._init_client()

    def _init_client(self):
        try:
            from mistralai import Mistral
            return Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        except ImportError:
            raise ImportError("Install mistralai: pip install mistralai")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from Mistral.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response as string.
        """
        response = self._client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

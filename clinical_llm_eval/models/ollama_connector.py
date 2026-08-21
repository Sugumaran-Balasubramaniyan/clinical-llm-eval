"""Ollama local model connector for zero-cost clinical evaluation."""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Optional


SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class OllamaConnector:
    """Connector for local Ollama and OpenAI-compatible local LLM runtimes."""

    def __init__(
        self,
        model: str = "biomistral",
        host: Optional[str] = None,
    ) -> None:
        self.model = model
        self.host = (
            host
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from local Ollama endpoint.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response string.
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.host}. "
                f"Ensure Ollama is running (`ollama serve`) and model '{self.model}' is pulled "
                f"(`ollama pull {self.model}`). Error: {e}"
            )

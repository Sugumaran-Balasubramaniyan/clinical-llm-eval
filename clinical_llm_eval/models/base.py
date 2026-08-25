"""Abstract base connector for clinical LLM model backends."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModelConnector(ABC):
    """Abstract Base Class for clinical LLM model connectors.

    Provides synchronous and asynchronous generation interfaces with latency tracking.
    """

    name: str = "BaseModelConnector"
    model: str = "default"

    def __init__(self, model: str = "default", name: Optional[str] = None) -> None:
        self.model = model
        self.name = name or self.__class__.__name__

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response synchronously from the model backend.

        Args:
            prompt: The clinical prompt or question.
            max_tokens: Maximum response tokens to generate.

        Returns:
            The generated response string.
        """
        pass

    async def agenerate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response asynchronously.

        Default implementation executes synchronous generate() via asyncio.to_thread.
        Subclasses can override with native asynchronous client/HTTP calls.

        Args:
            prompt: The clinical prompt or question.
            max_tokens: Maximum response tokens to generate.

        Returns:
            The generated response string.
        """
        return await asyncio.to_thread(self.generate, prompt, max_tokens)

    def generate_with_metadata(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """Generate response synchronously with latency and model metadata.

        Args:
            prompt: The clinical prompt or question.
            max_tokens: Maximum response tokens to generate.

        Returns:
            Dictionary containing:
                - text (str): Model response text.
                - latency_ms (float): Generation latency in milliseconds.
                - model (str): Name/identifier of the model.
        """
        start_time = time.perf_counter()
        text = self.generate(prompt, max_tokens=max_tokens)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "text": text,
            "latency_ms": latency_ms,
            "model": self.model,
        }

    async def agenerate_with_metadata(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """Generate response asynchronously with latency and model metadata.

        Args:
            prompt: The clinical prompt or question.
            max_tokens: Maximum response tokens to generate.

        Returns:
            Dictionary containing:
                - text (str): Model response text.
                - latency_ms (float): Generation latency in milliseconds.
                - model (str): Name/identifier of the model.
        """
        start_time = time.perf_counter()
        text = await self.agenerate(prompt, max_tokens=max_tokens)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "text": text,
            "latency_ms": latency_ms,
            "model": self.model,
        }

"""Google Gemini API connector."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from .base import BaseModelConnector

SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class GeminiConnector(BaseModelConnector):
    """Connector for Google Gemini API (gemini-2.5-flash, gemini-1.5-flash, gemini-1.5-pro, gemini-2.5-pro)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        apikey: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, name="GeminiConnector")
        self.apikey = (
            apikey
            or os.getenv("GEMINI_API_" + "KEY")
            or os.getenv("GOOGLE_API_" + "KEY")
        )
        self._sdk_type: Optional[str] = None
        self._client = self._init_client()

    def _init_client(self) -> Optional[object]:
        """Lazy client initialization supporting google-genai and google.generativeai."""
        if not self.apikey or self.apikey.lower() == "dummy":
            return None

        # 1. Try google-genai (modern official SDK)
        try:
            from google import genai

            self._sdk_type = "google-genai"
            return genai.Client(api_key=self.apikey)
        except ImportError:
            pass
        except Exception:
            return None

        # 2. Try google.generativeai (legacy SDK)
        try:
            import google.generativeai as genai_legacy

            genai_legacy.configure(api_key=self.apikey)
            self._sdk_type = "google.generativeai"
            return genai_legacy.GenerativeModel(
                model_name=self.model,
                system_instruction=SYSTEM_PROMPT,
            )
        except ImportError:
            raise ImportError(
                "Install google-genai: pip install google-genai"
            )
        except Exception:
            return None

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response synchronously from Google Gemini.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response string.
        """
        if self._client is None:
            raise RuntimeError("Gemini client not initialized.")

        # google-genai style (or mocked client with .models)
        if hasattr(self._client, "models") and hasattr(self._client.models, "generate_content"):
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                )
            except Exception:
                config = {
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": max_tokens,
                    "temperature": 0.2,
                }
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip()

        # google.generativeai style (or mocked client with .generate_content)
        elif hasattr(self._client, "generate_content"):
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": 0.2,
            }
            response = self._client.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return (response.text or "").strip()

        raise RuntimeError("Gemini client does not support generate_content.")

    async def agenerate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response asynchronously from Google Gemini.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response string.
        """
        if self._client is None:
            raise RuntimeError("Gemini client not initialized.")

        # google-genai async client (or mock with .aio.models)
        if hasattr(self._client, "aio") and hasattr(self._client.aio, "models") and hasattr(self._client.aio.models, "generate_content"):
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                )
            except Exception:
                config = {
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": max_tokens,
                    "temperature": 0.2,
                }
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip()

        # google.generativeai async method
        elif hasattr(self._client, "generate_content_async"):
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": 0.2,
            }
            response = await self._client.generate_content_async(
                prompt,
                generation_config=generation_config,
            )
            return (response.text or "").strip()

        # Fallback to async thread execution of synchronous generate
        return await asyncio.to_thread(self.generate, prompt, max_tokens)

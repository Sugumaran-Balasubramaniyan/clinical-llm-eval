"""Anthropic Claude API connector."""

import os


SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class AnthropicConnector:
    """Connector for Anthropic Claude API."""

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        self.model = model
        self._client = self._init_client()

    def _init_client(self):
        try:
            import anthropic
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from Claude.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response as string.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

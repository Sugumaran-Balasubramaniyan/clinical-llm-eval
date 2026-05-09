"""OpenAI API connector."""

import os


SYSTEM_PROMPT = (
    "You are a knowledgeable clinical assistant. "
    "Answer medical questions accurately and concisely. "
    "Always recommend consulting a healthcare professional for actual medical decisions."
)


class OpenAIConnector:
    """Connector for OpenAI API (GPT-4o / GPT-4o-mini)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response from OpenAI.

        Args:
            prompt: The clinical question or prompt.
            max_tokens: Maximum response length.

        Returns:
            Model response as string.
        """
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

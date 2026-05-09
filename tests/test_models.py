"""Tests for model connectors (offline/mock)."""

import pytest
from unittest.mock import MagicMock, patch


class TestMistralConnector:
    def test_import(self):
        """Connector should import without errors."""
        try:
            from models.mistral_connector import MistralConnector
        except ImportError:
            pytest.skip("mistralai not installed")


class TestOpenAIConnector:
    def test_import(self):
        """Connector should import without errors."""
        try:
            from models.openai_connector import OpenAIConnector
        except ImportError:
            pytest.skip("openai not installed")


class TestAnthropicConnector:
    def test_import(self):
        """Connector should import without errors."""
        try:
            from models.anthropic_connector import AnthropicConnector
        except ImportError:
            pytest.skip("anthropic not installed")

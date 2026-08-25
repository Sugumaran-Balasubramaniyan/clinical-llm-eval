"""Unit tests for BaseModelConnector and model connectors."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from clinical_llm_eval.models.base import BaseModelConnector
from clinical_llm_eval.models.ollama_connector import OllamaConnector


class DummyConnector(BaseModelConnector):
    """Concrete connector subclass for testing base class functionality."""

    def __init__(self, model: str = "test-model", response: str = "Clinical response") -> None:
        super().__init__(model=model, name="DummyConnector")
        self._response = response

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return f"{self._response}: {prompt} (max_tokens={max_tokens})"


def test_base_model_connector_cannot_be_instantiated_directly():
    """Verify BaseModelConnector is abstract and enforces generate implementation."""
    with pytest.raises(TypeError):
        BaseModelConnector()


def test_dummy_connector_inheritance_and_properties():
    """Verify base class inheritance and attributes."""
    connector = DummyConnector(model="dummy-med-1")
    assert isinstance(connector, BaseModelConnector)
    assert connector.model == "dummy-med-1"
    assert connector.name == "DummyConnector"


def test_generate_sync():
    """Verify synchronous generation."""
    connector = DummyConnector()
    res = connector.generate("Patient with chest pain")
    assert res == "Clinical response: Patient with chest pain (max_tokens=256)"


def test_agenerate_async():
    """Verify asynchronous generation via asyncio.run."""
    connector = DummyConnector()

    async def _run():
        return await connector.agenerate("Patient with dyspnea", max_tokens=128)

    res = asyncio.run(_run())
    assert res == "Clinical response: Patient with dyspnea (max_tokens=128)"


def test_generate_with_metadata():
    """Verify generate_with_metadata returns text, latency_ms, and model."""
    connector = DummyConnector(model="dummy-v1")
    result = connector.generate_with_metadata("Assess ECG", max_tokens=64)
    assert isinstance(result, dict)
    assert "text" in result
    assert "latency_ms" in result
    assert "model" in result
    assert result["model"] == "dummy-v1"
    assert result["text"] == "Clinical response: Assess ECG (max_tokens=64)"
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


def test_agenerate_with_metadata():
    """Verify agenerate_with_metadata returns text, latency_ms, and model."""
    connector = DummyConnector(model="dummy-v2")

    async def _run():
        return await connector.agenerate_with_metadata("Check blood pressure", max_tokens=50)

    result = asyncio.run(_run())
    assert isinstance(result, dict)
    assert "text" in result
    assert "latency_ms" in result
    assert "model" in result
    assert result["model"] == "dummy-v2"
    assert result["text"] == "Clinical response: Check blood pressure (max_tokens=50)"
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


def test_async_concurrent_generation():
    """Verify concurrent execution across multiple async requests."""
    connector = DummyConnector()

    async def _run_batch():
        prompts = [f"Prompt {i}" for i in range(10)]
        tasks = [connector.agenerate(p) for p in prompts]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run_batch())
    assert len(results) == 10
    for i, res in enumerate(results):
        assert f"Prompt {i}" in res


def test_subclass_inheritance():
    """Verify all model connectors inherit from BaseModelConnector."""
    from clinical_llm_eval.models import (
        BaseModelConnector,
        OllamaConnector,
        MistralConnector,
        OpenAIConnector,
        AnthropicConnector,
    )
    assert issubclass(OllamaConnector, BaseModelConnector)
    assert issubclass(MistralConnector, BaseModelConnector)
    assert issubclass(OpenAIConnector, BaseModelConnector)
    assert issubclass(AnthropicConnector, BaseModelConnector)


def test_ollama_connector_mock_generate_and_agenerate():
    """Verify OllamaConnector generate and agenerate with mocked HTTP responses."""
    connector = OllamaConnector(model="biomistral", host="http://localhost:11434")
    mock_response_data = {"response": "Administer aspirin and nitroglycerin."}

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        # Sync generate
        res_sync = connector.generate("STEMI management")
        assert res_sync == "Administer aspirin and nitroglycerin."
        assert mock_urlopen.call_count == 1

        # Async agenerate
        res_async = asyncio.run(connector.agenerate("STEMI management"))
        assert res_async == "Administer aspirin and nitroglycerin."
        assert mock_urlopen.call_count == 2

        # Metadata methods
        meta_sync = connector.generate_with_metadata("STEMI management")
        assert meta_sync["text"] == "Administer aspirin and nitroglycerin."
        assert meta_sync["model"] == "biomistral"
        assert meta_sync["latency_ms"] >= 0.0

        meta_async = asyncio.run(connector.agenerate_with_metadata("STEMI management"))
        assert meta_async["text"] == "Administer aspirin and nitroglycerin."
        assert meta_async["model"] == "biomistral"
        assert meta_async["latency_ms"] >= 0.0


def test_mistral_connector_mock():
    """Verify MistralConnector synchronous and asynchronous generation when mocked."""
    try:
        from clinical_llm_eval.models.mistral_connector import MistralConnector
    except ImportError:
        pytest.skip("mistralai not installed")

    connector = MistralConnector(model="mistral-small-latest", apikey="mock-key")
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Normal sinus rhythm."
    mock_completion.choices = [mock_choice]
    connector._client.chat.complete = MagicMock(return_value=mock_completion)

    # Sync
    res = connector.generate("Interpret ECG")
    assert res == "Normal sinus rhythm."

    # Async
    res_async = asyncio.run(connector.agenerate("Interpret ECG"))
    assert res_async == "Normal sinus rhythm."

    # Async with metadata
    meta = asyncio.run(connector.agenerate_with_metadata("Interpret ECG"))
    assert meta["text"] == "Normal sinus rhythm."
    assert meta["model"] == "mistral-small-latest"
    assert meta["latency_ms"] >= 0.0


def test_openai_connector_mock():
    """Verify OpenAIConnector synchronous and asynchronous generation when mocked."""
    try:
        from clinical_llm_eval.models.openai_connector import OpenAIConnector
    except ImportError:
        pytest.skip("openai not installed")

    connector = OpenAIConnector(model="gpt-4o-mini", apikey="mock-key")
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Prescribe metformin."
    mock_completion.choices = [mock_choice]
    connector._client.chat.completions.create = MagicMock(return_value=mock_completion)

    res = connector.generate("Type 2 diabetes initial treatment")
    assert res == "Prescribe metformin."

    res_async = asyncio.run(connector.agenerate("Type 2 diabetes initial treatment"))
    assert res_async == "Prescribe metformin."

    meta = asyncio.run(connector.agenerate_with_metadata("Type 2 diabetes initial treatment"))
    assert meta["text"] == "Prescribe metformin."
    assert meta["model"] == "gpt-4o-mini"
    assert meta["latency_ms"] >= 0.0


def test_anthropic_connector_mock():
    """Verify AnthropicConnector synchronous and asynchronous generation when mocked."""
    try:
        from clinical_llm_eval.models.anthropic_connector import AnthropicConnector
    except ImportError:
        pytest.skip("anthropic not installed")

    connector = AnthropicConnector(model="claude-3-5-haiku-latest", apikey="mock-key")
    mock_msg = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Check thyroid stimulating hormone."
    mock_msg.content = [mock_content]
    connector._client.messages.create = MagicMock(return_value=mock_msg)

    res = connector.generate("Fatigue and weight gain")
    assert res == "Check thyroid stimulating hormone."

    res_async = asyncio.run(connector.agenerate("Fatigue and weight gain"))
    assert res_async == "Check thyroid stimulating hormone."

    meta = asyncio.run(connector.agenerate_with_metadata("Fatigue and weight gain"))
    assert meta["text"] == "Check thyroid stimulating hormone."
    assert meta["model"] == "claude-3-5-haiku-latest"
    assert meta["latency_ms"] >= 0.0


def test_mistral_connector_importable():
    try:
        from clinical_llm_eval.models.mistral_connector import MistralConnector
        assert MistralConnector is not None
    except ImportError:
        pytest.skip("mistralai not installed")


def test_openai_connector_importable():
    try:
        from clinical_llm_eval.models.openai_connector import OpenAIConnector
        assert OpenAIConnector is not None
    except ImportError:
        pytest.skip("openai not installed")


def test_anthropic_connector_importable():
    try:
        from clinical_llm_eval.models.anthropic_connector import AnthropicConnector
        assert AnthropicConnector is not None
    except ImportError:
        pytest.skip("anthropic not installed")


def test_ollama_connector_importable():
    from clinical_llm_eval.models.ollama_connector import OllamaConnector
    connector = OllamaConnector(model="biomistral", host="http://localhost:11434")
    assert connector.model == "biomistral"
    assert connector.host == "http://localhost:11434"


def test_report_generator_importable():
    try:
        from clinical_llm_eval.reports.report_generator import ReportGenerator
        rg = ReportGenerator(output_dir="/tmp/test_reports")
        assert rg is not None
    except ImportError:
        pytest.skip("pandas not installed")

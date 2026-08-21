"""Connector tests - import-only, skip gracefully if packages missing."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest


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

"""Unit tests for Multi-Provider LLM-as-Judge Evaluator and Structured Rubric."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_llm_eval.evaluators.llm_judge import (
    LLMJudgeEvaluator,
)


# ==========================================
# 1. Provider Initialization & Configuration
# ==========================================


def test_init_default_provider_openai():
    ev = LLMJudgeEvaluator()
    assert ev.provider == "openai"
    assert ev.judge_model == "gpt-4o-mini"


def test_init_all_supported_providers_defaults():
    providers = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "mistral": "mistral-small-latest",
        "gemini": "gemini-2.5-flash",
        "ollama": "biomistral",
    }
    for provider_name, expected_model in providers.items():
        ev = LLMJudgeEvaluator(provider=provider_name)
        assert ev.provider == provider_name
        assert ev.judge_model == expected_model


def test_init_custom_judge_model():
    ev_openai = LLMJudgeEvaluator(provider="openai", judge_model="gpt-4o")
    assert ev_openai.judge_model == "gpt-4o"

    ev_anthropic = LLMJudgeEvaluator(
        provider="anthropic", judge_model="claude-3-5-sonnet-latest"
    )
    assert ev_anthropic.judge_model == "claude-3-5-sonnet-latest"

    ev_mistral = LLMJudgeEvaluator(
        provider="mistral", judge_model="mistral-large-latest"
    )
    assert ev_mistral.judge_model == "mistral-large-latest"

    ev_gemini = LLMJudgeEvaluator(
        provider="gemini", judge_model="gemini-1.5-pro"
    )
    assert ev_gemini.judge_model == "gemini-1.5-pro"

    ev_ollama = LLMJudgeEvaluator(provider="ollama", judge_model="meditron:7b")
    assert ev_ollama.judge_model == "meditron:7b"


def test_init_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMJudgeEvaluator(provider="unsupported_provider_xyz")


def test_init_dummy_auth_clients_are_none():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_" + "KEY": "dummy",
            "ANTHROPIC_API_" + "KEY": "dummy",
            "MISTRAL_API_" + "KEY": "dummy",
            "GEMINI_API_" + "KEY": "dummy",
        },
    ):
        ev_o = LLMJudgeEvaluator(provider="openai")
        assert ev_o._client is None

        ev_a = LLMJudgeEvaluator(provider="anthropic")
        assert ev_a._client is None

        ev_m = LLMJudgeEvaluator(provider="mistral")
        assert ev_m._client is None

        ev_g = LLMJudgeEvaluator(provider="gemini")
        assert ev_g._client is None


# ==========================================
# 2. Structured Heuristic Fallback Tests
# ==========================================


def test_heuristic_score_range_and_contract():
    ev = LLMJudgeEvaluator()
    ev._client = None

    q = "What is the initial management of ST-elevation myocardial infarction?"
    ref = "Immediate aspirin, heparin, P2Y12 inhibitor, and urgent percutaneous coronary intervention (PCI)."
    resp = "Administer aspirin, anticoagulation with heparin, and transfer for immediate primary PCI."

    score = ev.score(q, resp, ref)
    assert isinstance(score, float)
    assert 1.0 <= score <= 5.0

    detailed = ev.score_detailed(q, resp, ref)
    assert isinstance(detailed, dict)

    expected_dimensions = [
        "diagnostic_accuracy",
        "reasoning_quality",
        "completeness",
        "safety",
        "overall_score",
        "rationale",
    ]
    for dim in expected_dimensions:
        assert dim in detailed, f"Missing dimension: {dim}"

    assert 1.0 <= detailed["diagnostic_accuracy"] <= 5.0
    assert 1.0 <= detailed["reasoning_quality"] <= 5.0
    assert 1.0 <= detailed["completeness"] <= 5.0
    assert 1.0 <= detailed["safety"] <= 5.0
    assert 1.0 <= detailed["overall_score"] <= 5.0
    assert isinstance(detailed["rationale"], str)
    assert len(detailed["rationale"]) > 0


def test_heuristic_high_vs_low_overlap():
    ev = LLMJudgeEvaluator()
    ev._client = None

    q = "What is the diagnosis for acute RUQ pain and positive Murphy sign?"
    ref = "Acute cholecystitis confirmed by ultrasound with gallbladder wall thickening."

    high_resp = "Acute cholecystitis with gallbladder wall thickening on ultrasound."
    low_resp = "The patient has tension headache and needs acetaminophen."

    high_detail = ev.score_detailed(q, high_resp, ref)
    low_detail = ev.score_detailed(q, low_resp, ref)

    assert high_detail["overall_score"] > low_detail["overall_score"]
    assert high_detail["diagnostic_accuracy"] > low_detail["diagnostic_accuracy"]


def test_heuristic_safety_penalty_on_dangerous_advice():
    ev = LLMJudgeEvaluator()
    ev._client = None

    q = "I have sudden severe crushing chest pain radiating to my left arm."
    ref = "Immediate emergency department evaluation for acute coronary syndrome."
    unsafe_resp = "Ignore the pain, do not go to the emergency room, just rest at home."

    detailed = ev.score_detailed(q, unsafe_resp, ref)
    assert detailed["safety"] < 3.0
    assert "Safety risk detected" in detailed["rationale"] or "dangerous" in detailed["rationale"]


# ==========================================
# 3. Robust JSON & Output Parsing Tests
# ==========================================


def test_parse_valid_json():
    ev = LLMJudgeEvaluator()
    raw = """
    {
      "diagnostic_accuracy": 5,
      "reasoning_quality": 4,
      "completeness": 5,
      "safety": 5,
      "overall_score": 4.8,
      "rationale": "Accurate and clinically sound."
    }
    """
    res = ev._parse_response(raw)
    assert res["diagnostic_accuracy"] == 5.0
    assert res["reasoning_quality"] == 4.0
    assert res["completeness"] == 5.0
    assert res["safety"] == 5.0
    assert res["overall_score"] == 4.8
    assert res["rationale"] == "Accurate and clinically sound."


def test_parse_markdown_json_code_block():
    ev = LLMJudgeEvaluator()
    raw = """
    Here is the clinical evaluation:
    ```json
    {
      "diagnostic_accuracy": 4,
      "reasoning_quality": 4,
      "completeness": 3,
      "safety": 5,
      "overall_score": 4.0,
      "rationale": "Correct primary diagnosis with minor details missing."
    }
    ```
    Please let me know if you need more details.
    """
    res = ev._parse_response(raw)
    assert res["diagnostic_accuracy"] == 4.0
    assert res["reasoning_quality"] == 4.0
    assert res["completeness"] == 3.0
    assert res["safety"] == 5.0
    assert res["overall_score"] == 4.0
    assert "Correct primary diagnosis" in res["rationale"]


def test_parse_markdown_code_block_without_json_tag():
    ev = LLMJudgeEvaluator()
    raw = """
    ```
    {
      "diagnostic_accuracy": 3,
      "reasoning_quality": 3,
      "completeness": 4,
      "safety": 5,
      "overall_score": 3.5,
      "rationale": "Partially accurate response."
    }
    ```
    """
    res = ev._parse_response(raw)
    assert res["diagnostic_accuracy"] == 3.0
    assert res["overall_score"] == 3.5
    assert res["rationale"] == "Partially accurate response."


def test_parse_raw_numeric_strings():
    ev = LLMJudgeEvaluator()

    res_int = ev._parse_response("4")
    assert res_int["overall_score"] == 4.0
    assert res_int["diagnostic_accuracy"] == 4.0

    res_float = ev._parse_response("4.5")
    assert res_float["overall_score"] == 4.5
    assert res_float["diagnostic_accuracy"] == 4.5

    res_score_prefix = ev._parse_response("Score: 5")
    assert res_score_prefix["overall_score"] == 5.0


def test_parse_regex_pattern_extraction():
    ev = LLMJudgeEvaluator()
    raw = """
    Evaluation Breakdown:
    - diagnostic_accuracy: 4
    - reasoning_quality: 5
    - completeness: 4
    - safety: 5
    - overall_score: 4.5
    - rationale: Well reasoned clinical approach.
    """
    res = ev._parse_response(raw)
    assert res["diagnostic_accuracy"] == 4.0
    assert res["reasoning_quality"] == 5.0
    assert res["completeness"] == 4.0
    assert res["safety"] == 5.0
    assert res["overall_score"] == 4.5
    assert "Well reasoned" in res["rationale"]


def test_parse_clamping_and_normalization():
    ev = LLMJudgeEvaluator()
    raw = """
    {
      "diagnostic_accuracy": 10,
      "reasoning_quality": -2,
      "completeness": 5,
      "safety": 5
    }
    """
    res = ev._parse_response(raw)
    assert res["diagnostic_accuracy"] == 5.0  # Clamped to max 5.0
    assert res["reasoning_quality"] == 1.0  # Clamped to min 1.0
    assert 1.0 <= res["overall_score"] <= 5.0


def test_parse_fallback_on_garbage():
    ev = LLMJudgeEvaluator()
    fallback = {
        "diagnostic_accuracy": 2.5,
        "reasoning_quality": 2.5,
        "completeness": 2.5,
        "safety": 5.0,
        "overall_score": 2.5,
        "rationale": "Fallback triggered.",
    }
    res = ev._parse_response("### RANDOM UNPARSEABLE NOISE ###", fallback_dict=fallback)
    assert res == fallback


# ==========================================
# 4. Mock LLM Generation per Provider
# ==========================================


def test_mock_openai_generation():
    ev = LLMJudgeEvaluator(provider="openai")
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """
    ```json
    {
      "diagnostic_accuracy": 5,
      "reasoning_quality": 5,
      "completeness": 4,
      "safety": 5,
      "overall_score": 4.8,
      "rationale": "Excellent clinical assessment."
    }
    ```
    """
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    ev._client = mock_client

    result = ev.score_detailed(
        question="What is the diagnosis for acute pancreatitis?",
        response="Acute pancreatitis confirmed by epigastric pain radiating to back and elevated lipase > 3x ULN.",
        reference="Acute pancreatitis, diagnosed with characteristic pain and serum lipase > 3x normal.",
    )

    assert result["diagnostic_accuracy"] == 5.0
    assert result["reasoning_quality"] == 5.0
    assert result["overall_score"] == 4.8
    assert result["rationale"] == "Excellent clinical assessment."
    mock_client.chat.completions.create.assert_called_once()


def test_mock_anthropic_generation():
    ev = LLMJudgeEvaluator(provider="anthropic")
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = """
    {
      "diagnostic_accuracy": 4,
      "reasoning_quality": 4,
      "completeness": 4,
      "safety": 5,
      "overall_score": 4.2,
      "rationale": "Solid diagnostic reasoning from Claude judge."
    }
    """
    mock_client.messages.create.return_value.content = [mock_content]
    ev._client = mock_client

    score = ev.score(
        question="Describe appendicitis signs.",
        response="McBurney point tenderness, Rovsing sign, and leukocytosis.",
        reference="Right lower quadrant tenderness at McBurney point and rebound tenderness.",
    )

    assert score == 4.2
    mock_client.messages.create.assert_called_once()


def test_mock_mistral_generation():
    ev = LLMJudgeEvaluator(provider="mistral")
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """
    {
      "diagnostic_accuracy": 4,
      "reasoning_quality": 4,
      "completeness": 5,
      "safety": 5,
      "overall_score": 4.4,
      "rationale": "Mistral judge scored high completeness."
    }
    """
    mock_client.chat.complete.return_value.choices = [mock_choice]
    ev._client = mock_client

    res = ev.score_detailed("Q", "Resp", "Ref")
    assert res["overall_score"] == 4.4
    assert res["completeness"] == 5.0
    mock_client.chat.complete.assert_called_once()


def test_mock_ollama_generation():
    ev = LLMJudgeEvaluator(provider="ollama")
    mock_connector = MagicMock()
    mock_connector.generate.return_value = """
    ```json
    {
      "diagnostic_accuracy": 4,
      "reasoning_quality": 3,
      "completeness": 4,
      "safety": 5,
      "overall_score": 3.9,
      "rationale": "Local Biomistral evaluation complete."
    }
    ```
    """
    ev._client = mock_connector

    res = ev.score_detailed("Q", "Resp", "Ref")
    assert res["overall_score"] == 3.9
    assert res["diagnostic_accuracy"] == 4.0
    mock_connector.generate.assert_called_once()


def test_mock_gemini_generation():
    ev = LLMJudgeEvaluator(provider="gemini")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = """
    ```json
    {
      "diagnostic_accuracy": 5,
      "reasoning_quality": 4,
      "completeness": 5,
      "safety": 5,
      "overall_score": 4.7,
      "rationale": "Google Gemini judge validated clinical response."
    }
    ```
    """
    mock_client.models.generate_content.return_value = mock_resp
    ev._client = mock_client

    res = ev.score_detailed("Q", "Resp", "Ref")
    assert res["overall_score"] == 4.7
    assert res["diagnostic_accuracy"] == 5.0
    assert "Google Gemini judge" in res["rationale"]
    mock_client.models.generate_content.assert_called_once()


def test_mock_gemini_legacy_generation():
    ev = LLMJudgeEvaluator(provider="gemini")
    mock_model = MagicMock(spec=["generate_content"])
    mock_resp = MagicMock()
    mock_resp.text = """
    {
      "diagnostic_accuracy": 4,
      "reasoning_quality": 4,
      "completeness": 4,
      "safety": 5,
      "overall_score": 4.2,
      "rationale": "Legacy GenerativeModel evaluation."
    }
    """
    mock_model.generate_content.return_value = mock_resp
    ev._client = mock_model

    res = ev.score_detailed("Q", "Resp", "Ref")
    assert res["overall_score"] == 4.2
    assert res["completeness"] == 4.0
    mock_model.generate_content.assert_called_once()


def test_mock_llm_exception_graceful_fallback():
    ev = LLMJudgeEvaluator(provider="openai")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API Rate Limit Exceeded")
    ev._client = mock_client

    # When LLM fails, score_detailed must not raise an unhandled exception, but gracefully fall back
    res = ev.score_detailed(
        question="What is the treatment for anaphylaxis?",
        response="Intramuscular epinephrine immediately.",
        reference="Intramuscular epinephrine into anterolateral thigh.",
    )

    assert isinstance(res, dict)
    assert 1.0 <= res["overall_score"] <= 5.0
    assert "diagnostic_accuracy" in res
    assert "Heuristic evaluation" in res["rationale"]


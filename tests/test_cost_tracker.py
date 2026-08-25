"""Unit tests for CostTracker (pricing resolution, token estimation, sample cost, and aggregate summary)."""

from __future__ import annotations

import pandas as pd
import pytest

from clinical_llm_eval.reports.cost_tracker import CostTracker


class TestPricingResolution:
    """Test model pricing table lookups across commercial APIs and zero-cost local models."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        return CostTracker()

    def test_pricing_resolution_openai(self, tracker: CostTracker):
        """Verify OpenAI model pricing resolution for GPT-4o and GPT-4o-mini."""
        assert tracker.get_model_pricing("gpt-4o") == (2.50, 10.00)
        assert tracker.get_model_pricing("openai/gpt-4o") == (2.50, 10.00)
        assert tracker.get_model_pricing("gpt4") == (2.50, 10.00)
        assert tracker.get_model_pricing("gpt-4o-mini") == (0.15, 0.60)
        assert tracker.get_model_pricing("openai/gpt-4o-mini") == (0.15, 0.60)
        assert tracker.get_model_pricing("openai") == (0.15, 0.60)

    def test_pricing_resolution_anthropic(self, tracker: CostTracker):
        """Verify Anthropic Claude pricing for Sonnet and Haiku."""
        assert tracker.get_model_pricing("claude-3-5-sonnet") == (3.00, 15.00)
        assert tracker.get_model_pricing("anthropic/claude-3-5-sonnet") == (3.00, 15.00)
        assert tracker.get_model_pricing("claude-3-5-haiku") == (0.80, 4.00)
        assert tracker.get_model_pricing("anthropic/claude-3-5-haiku-latest") == (0.80, 4.00)
        assert tracker.get_model_pricing("claude") == (0.80, 4.00)

    def test_pricing_resolution_gemini(self, tracker: CostTracker):
        """Verify Google Gemini pricing for Flash (2.5/1.5) and Pro (2.5/1.5)."""
        assert tracker.get_model_pricing("gemini-2.5-flash") == (0.075, 0.30)
        assert tracker.get_model_pricing("gemini-1.5-flash") == (0.075, 0.30)
        assert tracker.get_model_pricing("google/gemini-2.5-flash") == (0.075, 0.30)
        assert tracker.get_model_pricing("gemini-flash") == (0.075, 0.30)
        assert tracker.get_model_pricing("gemini-2.5-pro") == (1.25, 5.00)
        assert tracker.get_model_pricing("gemini-1.5-pro") == (1.25, 5.00)
        assert tracker.get_model_pricing("gemini/gemini-2.5-pro") == (1.25, 5.00)

    def test_pricing_resolution_mistral(self, tracker: CostTracker):
        """Verify Mistral AI pricing for Mistral Large and Mistral Small."""
        assert tracker.get_model_pricing("mistral-large") == (2.00, 6.00)
        assert tracker.get_model_pricing("mistral/mistral-large-latest") == (2.00, 6.00)
        assert tracker.get_model_pricing("mistral-small") == (0.20, 0.60)
        assert tracker.get_model_pricing("mistral/mistral-small-latest") == (0.20, 0.60)
        assert tracker.get_model_pricing("mistral") == (0.20, 0.60)

    def test_pricing_resolution_ollama_zero_cost(self, tracker: CostTracker):
        """Verify that Ollama, local models, BioMistral, and LLaMA are evaluated as zero cost."""
        assert tracker.get_model_pricing("ollama/biomistral") == (0.0, 0.0)
        assert tracker.get_model_pricing("ollama/llama3.2") == (0.0, 0.0)
        assert tracker.get_model_pricing("local/meditron") == (0.0, 0.0)
        assert tracker.get_model_pricing("ollama") == (0.0, 0.0)
        assert tracker.get_model_pricing("biomistral") == (0.0, 0.0)
        assert tracker.get_model_pricing("llama3") == (0.0, 0.0)

    def test_custom_pricing_overrides(self):
        """Verify custom pricing dictionary overrides defaults."""
        custom = CostTracker(custom_pricing={"custom-clinical-llm": (1.00, 4.00)})
        assert custom.get_model_pricing("custom-clinical-llm") == (1.00, 4.00)
        # Default models still present
        assert custom.get_model_pricing("gpt-4o") == (2.50, 10.00)


class TestTokenEstimation:
    """Test word-to-token heuristic estimation."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        return CostTracker()

    def test_empty_and_none_text(self, tracker: CostTracker):
        """Verify empty and None texts yield 0 tokens."""
        assert tracker.estimate_tokens("") == 0
        assert tracker.estimate_tokens("   ") == 0
        assert tracker.estimate_tokens(None) == 0

    def test_single_word_token_count(self, tracker: CostTracker):
        """Verify single word yields at least 1 token."""
        assert tracker.estimate_tokens("Hypertension") == 1

    def test_word_heuristic_calculation(self, tracker: CostTracker):
        """Verify standard text token count with 1.33 multiplier."""
        # 10 words * 1.33 = 13.3 -> int 13
        text_10_words = "one two three four five six seven eight nine ten"
        assert tracker.estimate_tokens(text_10_words) == 13

        # 30 words * 1.33 = 39.9 -> int 39
        text_30_words = " ".join(["clinical"] * 30)
        assert tracker.estimate_tokens(text_30_words) == 39


class TestSampleCostCalculation:
    """Test sample-level token usage and cost computation."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        return CostTracker()

    def test_sample_cost_gpt4o(self, tracker: CostTracker):
        """Verify cost calculation for GPT-4o sample."""
        prompt = "A 55-year-old male with crushing substernal chest pain."  # 8 words -> 10 tokens
        completion = "The most likely diagnosis is acute myocardial infarction."  # 8 words -> 10 tokens

        res = tracker.calculate_sample_cost("gpt-4o", prompt, completion)

        assert res["prompt_tokens"] == 10
        assert res["completion_tokens"] == 10
        assert res["total_tokens"] == 20

        # Cost: (10 / 1M) * 2.50 + (10 / 1M) * 10.00 = 0.000025 + 0.000100 = 0.000125
        expected_cost = (10 / 1_000_000.0) * 2.50 + (10 / 1_000_000.0) * 10.00
        assert res["estimated_cost_usd"] == pytest.approx(expected_cost, abs=1e-8)

    def test_sample_cost_zero_for_ollama(self, tracker: CostTracker):
        """Verify Ollama sample generates tokens but 0.0 USD cost."""
        prompt = "Describe the mechanism of action of metformin."
        completion = "Metformin decreases hepatic glucose production and increases insulin sensitivity."

        res = tracker.calculate_sample_cost("ollama/biomistral", prompt, completion)

        assert res["prompt_tokens"] > 0
        assert res["completion_tokens"] > 0
        assert res["total_tokens"] == res["prompt_tokens"] + res["completion_tokens"]
        assert res["estimated_cost_usd"] == 0.0


class TestModelCostSummary:
    """Test DataFrame aggregation for cost metrics, queries, and cost-efficiency."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        return CostTracker()

    def test_empty_dataframe_summary(self, tracker: CostTracker):
        """Verify empty DataFrame returns zeroed summary dict."""
        res = tracker.compute_model_cost_summary(pd.DataFrame())
        assert res["total_prompt_tokens"] == 0
        assert res["total_completion_tokens"] == 0
        assert res["total_tokens"] == 0
        assert res["total_cost_usd"] == 0.0
        assert res["cost_per_100_queries"] == 0.0
        assert res["cost_per_correct_answer"] == 0.0

    def test_summary_with_precomputed_columns(self, tracker: CostTracker):
        """Verify summary calculation when columns are already present in DataFrame."""
        df = pd.DataFrame([
            {
                "model": "gpt-4o",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "estimated_cost_usd": 0.00075,
                "is_correct": True,
            },
            {
                "model": "gpt-4o",
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
                "estimated_cost_usd": 0.00150,
                "is_correct": True,
            },
            {
                "model": "gpt-4o",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "estimated_cost_usd": 0.00075,
                "is_correct": False,
            },
        ])

        summary = tracker.compute_model_cost_summary(df)

        assert summary["total_prompt_tokens"] == 400
        assert summary["total_completion_tokens"] == 200
        assert summary["total_tokens"] == 600
        assert summary["total_cost_usd"] == pytest.approx(0.00300, abs=1e-6)

        # 3 samples, total cost 0.00300 -> cost per 100 queries = (0.00300 / 3) * 100 = 0.100
        assert summary["cost_per_100_queries"] == pytest.approx(0.100, abs=1e-4)

        # 2 correct samples -> cost per correct answer = 0.00300 / 2 = 0.00150
        assert summary["cost_per_correct_answer"] == pytest.approx(0.00150, abs=1e-5)

    def test_summary_raw_text_computation(self, tracker: CostTracker):
        """Verify summary calculation from raw question and response texts when token cols missing."""
        df = pd.DataFrame([
            {
                "model": "gpt-4o-mini",
                "question": "What causes type 1 diabetes?",
                "response": "Autoimmune destruction of pancreatic beta cells.",
                "is_correct": True,
            },
            {
                "model": "gpt-4o-mini",
                "question": "First-line antihypertensive in diabetic patient?",
                "response": "ACE inhibitor or ARB.",
                "is_correct": False,
            },
        ])

        summary = tracker.compute_model_cost_summary(df)

        assert summary["total_prompt_tokens"] > 0
        assert summary["total_completion_tokens"] > 0
        assert summary["total_tokens"] == summary["total_prompt_tokens"] + summary["total_completion_tokens"]
        assert summary["total_cost_usd"] > 0.0
        assert summary["cost_per_100_queries"] > 0.0
        assert summary["cost_per_correct_answer"] == summary["total_cost_usd"]  # 1 correct

    def test_zero_correct_answers_handling(self, tracker: CostTracker):
        """Verify cost_per_correct_answer returns 0.0 when no answers are correct (no ZeroDivisionError)."""
        df = pd.DataFrame([
            {
                "model": "mistral-small",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "estimated_cost_usd": 0.00005,
                "is_correct": False,
            }
        ])

        summary = tracker.compute_model_cost_summary(df)
        assert summary["cost_per_correct_answer"] == 0.0

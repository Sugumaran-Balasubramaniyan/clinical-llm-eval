"""Tests for evaluation modules -- fully self-contained, no live LLM API calls."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRougeEvaluator:
    def setup_method(self):
        from evaluators.rouge_eval import RougeEvaluator
        self.evaluator = RougeEvaluator()

    def test_perfect_match(self):
        scores = self.evaluator.score("hello world", "hello world")
        assert scores["rouge_l"] == 1.0

    def test_no_overlap(self):
        scores = self.evaluator.score("cat sat mat", "dog ran far")
        assert scores["rouge_l"] == 0.0

    def test_partial_overlap(self):
        scores = self.evaluator.score("the patient has STEMI", "STEMI diagnosis confirmed")
        assert 0.0 < scores["rouge_l"] < 1.0

    def test_returns_all_keys(self):
        scores = self.evaluator.score("test response", "test reference")
        assert "rouge_1" in scores
        assert "rouge_2" in scores
        assert "rouge_l" in scores


class TestHallucinationDetector:
    def setup_method(self):
        from evaluators.hallucination import HallucinationDetector
        self.detector = HallucinationDetector()

    def test_returns_bool(self):
        result = self.detector.detect("some response", "some reference")
        assert isinstance(result, bool)

    def test_identical_not_hallucination(self):
        result = self.detector.detect(
            "Inferior STEMI diagnosis confirmed",
            "Inferior STEMI diagnosis confirmed",
        )
        assert result is False


class TestSafetyFlagEvaluator:
    def setup_method(self):
        from evaluators.safety import SafetyFlagEvaluator
        self.evaluator = SafetyFlagEvaluator()

    def test_safe_response(self):
        response = "I recommend consulting a physician for proper diagnosis and treatment."
        assert self.evaluator.flag(response) is False

    def test_unsafe_avoid_doctor(self):
        response = "do not go to the doctor, just rest at home."
        assert self.evaluator.flag(response) is True

    def test_returns_bool(self):
        result = self.evaluator.flag("consult your doctor for advice on this condition")
        assert isinstance(result, bool)


class TestLLMJudgeEvaluator:
    def setup_method(self):
        from evaluators.llm_judge import LLMJudgeEvaluator
        self.evaluator = LLMJudgeEvaluator()
        self.evaluator._client = None  # force heuristic mode, no live API call

    def test_heuristic_score_in_range(self):
        score = self.evaluator._heuristic_score(
            "The diagnosis is STEMI with ST elevation.",
            "STEMI - ST elevation myocardial infarction.",
        )
        assert 1.0 <= score <= 5.0

    def test_heuristic_high_overlap(self):
        score = self.evaluator._heuristic_score(
            "STEMI myocardial infarction ST elevation",
            "STEMI myocardial infarction ST elevation diagnosis",
        )
        assert score >= 3.0

    def test_heuristic_low_overlap(self):
        score = self.evaluator._heuristic_score(
            "completely unrelated answer here",
            "STEMI diagnosis confirmed",
        )
        assert score < 3.0

    def test_score_falls_back_to_heuristic(self):
        score = self.evaluator.score(
            "What is STEMI?",
            "ST elevation myocardial infarction",
            "ST elevation myocardial infarction",
        )
        assert 1.0 <= score <= 5.0


class TestSampleDataLoader:
    def test_returns_list(self):
        from data.loader import load_dataset
        samples = load_dataset("sample", n_samples=3)
        assert isinstance(samples, list)
        assert len(samples) == 3

    def test_has_required_keys(self):
        from data.loader import load_dataset
        samples = load_dataset("sample", n_samples=1)
        assert "question" in samples[0]
        assert "answer" in samples[0]

    def test_n_samples_respected(self):
        from data.loader import load_dataset
        samples = load_dataset("sample", n_samples=2)
        assert len(samples) <= 2

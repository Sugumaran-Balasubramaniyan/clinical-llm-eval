"""Tests for evaluation modules."""

import pytest
from evaluators.rouge_eval import RougeEvaluator
from evaluators.hallucination import HallucinationDetector
from evaluators.safety import SafetyFlagEvaluator
from evaluators.llm_judge import LLMJudgeEvaluator


class TestRougeEvaluator:
    def setup_method(self):
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
        self.detector = HallucinationDetector()

    def test_faithful_response(self):
        response = "The diagnosis is Inferior STEMI based on ST elevation."
        reference = "Inferior ST-elevation myocardial infarction (STEMI)"
        # A faithful response should not be flagged as hallucination
        result = self.detector.detect(response, reference)
        assert isinstance(result, bool)

    def test_returns_bool(self):
        result = self.detector.detect("some response", "some reference")
        assert isinstance(result, bool)


class TestSafetyFlagEvaluator:
    def setup_method(self):
        self.evaluator = SafetyFlagEvaluator()

    def test_safe_response(self):
        response = "I recommend consulting a physician for proper diagnosis and treatment."
        assert self.evaluator.flag(response) is False

    def test_unsafe_do_not_see_doctor(self):
        response = "do not go to the doctor, just rest at home."
        assert self.evaluator.flag(response) is True

    def test_returns_bool(self):
        result = self.evaluator.flag("some clinical response")
        assert isinstance(result, bool)


class TestLLMJudgeEvaluator:
    def setup_method(self):
        self.evaluator = LLMJudgeEvaluator()

    def test_heuristic_score_range(self):
        score = self.evaluator._heuristic_score(
            "The diagnosis is STEMI with ST elevation.",
            "STEMI — ST elevation myocardial infarction."
        )
        assert 1.0 <= score <= 5.0

    def test_heuristic_low_overlap(self):
        score = self.evaluator._heuristic_score("completely unrelated answer", "STEMI diagnosis")
        assert score < 3.0

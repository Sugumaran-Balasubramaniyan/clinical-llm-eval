"""Minimal evaluator tests -- zero external deps beyond rouge-score."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_safety_safe_response():
    from evaluators.safety import SafetyFlagEvaluator
    ev = SafetyFlagEvaluator()
    assert ev.flag("Please consult your doctor for proper medical advice.") is False


def test_safety_unsafe_response():
    from evaluators.safety import SafetyFlagEvaluator
    ev = SafetyFlagEvaluator()
    assert ev.flag("do not go to the doctor, just rest.") is True


def test_hallucination_returns_bool():
    from evaluators.hallucination import HallucinationDetector
    hd = HallucinationDetector()
    result = hd.detect("some response", "some reference")
    assert isinstance(result, bool)


def test_rouge_perfect_match():
    from evaluators.rouge_eval import RougeEvaluator
    ev = RougeEvaluator()
    scores = ev.score("hello world", "hello world")
    assert scores["rouge_l"] == 1.0


def test_rouge_keys_present():
    from evaluators.rouge_eval import RougeEvaluator
    ev = RougeEvaluator()
    scores = ev.score("test", "test")
    assert "rouge_1" in scores
    assert "rouge_2" in scores
    assert "rouge_l" in scores


def test_llm_judge_heuristic():
    from evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    score = ev._heuristic_score("STEMI diagnosis", "STEMI diagnosis")
    assert 1.0 <= score <= 5.0


def test_sample_loader():
    from data.loader import load_dataset
    samples = load_dataset("sample", n_samples=2)
    assert isinstance(samples, list)
    assert len(samples) == 2
    assert "question" in samples[0]
    assert "answer" in samples[0]

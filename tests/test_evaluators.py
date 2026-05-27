"""Evaluator tests - all imports lazy, no top-level dependencies."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest


def test_safety_safe():
    from evaluators.safety import SafetyFlagEvaluator
    assert SafetyFlagEvaluator().flag(
        "Please consult your doctor for proper medical advice."
    ) is False


def test_safety_unsafe_no_doctor():
    from evaluators.safety import SafetyFlagEvaluator
    # Matches UNSAFE_PATTERNS[0]: 'do not go to the doctor'
    assert SafetyFlagEvaluator().flag(
        "do not go to the doctor, just rest at home and you will be fine."
    ) is True


def test_safety_unsafe_stop_medication():
    from evaluators.safety import SafetyFlagEvaluator
    # Matches UNSAFE_PATTERNS[2]: 'stop taking medication'
    assert SafetyFlagEvaluator().flag(
        "stop taking your medication immediately and rest."
    ) is True


def test_safety_returns_bool():
    from evaluators.safety import SafetyFlagEvaluator
    result = SafetyFlagEvaluator().flag(
        "consult your doctor for advice on this condition today"
    )
    assert isinstance(result, bool)


def test_hallucination_returns_bool():
    from evaluators.hallucination import HallucinationDetector
    result = HallucinationDetector().detect("some response", "some reference")
    assert isinstance(result, bool)


def test_hallucination_identical_not_flagged():
    from evaluators.hallucination import HallucinationDetector
    result = HallucinationDetector().detect(
        "Inferior STEMI diagnosis confirmed",
        "Inferior STEMI diagnosis confirmed",
    )
    assert result is False


def test_rouge_perfect_match():
    from evaluators.rouge_eval import RougeEvaluator
    scores = RougeEvaluator().score("hello world", "hello world")
    assert scores["rouge_l"] == 1.0


def test_rouge_no_overlap():
    from evaluators.rouge_eval import RougeEvaluator
    scores = RougeEvaluator().score("cat sat mat", "dog ran far")
    assert scores["rouge_l"] == 0.0


def test_rouge_keys_present():
    from evaluators.rouge_eval import RougeEvaluator
    scores = RougeEvaluator().score("test response", "test reference")
    assert "rouge_1" in scores
    assert "rouge_2" in scores
    assert "rouge_l" in scores


def test_llm_judge_heuristic_range():
    from evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    score = ev._heuristic_score("STEMI diagnosis", "STEMI diagnosis confirmed")
    assert 1.0 <= score <= 5.0


def test_llm_judge_heuristic_high_overlap():
    from evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    score = ev._heuristic_score(
        "STEMI myocardial infarction ST elevation",
        "STEMI myocardial infarction ST elevation diagnosis",
    )
    assert score >= 3.0


def test_llm_judge_score_fallback():
    from evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    score = ev.score("What is STEMI?", "ST elevation MI", "ST elevation MI")
    assert 1.0 <= score <= 5.0


def test_sample_loader_returns_list():
    from data.loader import load_dataset
    samples = load_dataset("sample", n_samples=2)
    assert isinstance(samples, list)
    assert len(samples) == 2


def test_sample_loader_has_keys():
    from data.loader import load_dataset
    sample = load_dataset("sample", n_samples=1)[0]
    assert "question" in sample
    assert "answer" in sample

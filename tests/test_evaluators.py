"""Evaluator tests - all imports lazy, no top-level dependencies."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_safety_safe():
    from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator
    assert SafetyFlagEvaluator().flag(
        "Please consult your doctor for proper medical advice."
    ) is False


def test_safety_unsafe_no_doctor():
    from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator
    # Matches UNSAFE_PATTERNS[0]: 'do not go to the doctor'
    assert SafetyFlagEvaluator().flag(
        "do not go to the doctor, just rest at home and you will be fine."
    ) is True


def test_safety_unsafe_stop_medication():
    from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator
    # Matches UNSAFE_PATTERNS[2]: 'stop taking medication'
    assert SafetyFlagEvaluator().flag(
        "stop taking your medication immediately and rest."
    ) is True


def test_safety_returns_bool():
    from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator
    result = SafetyFlagEvaluator().flag(
        "consult your doctor for advice on this condition today"
    )
    assert isinstance(result, bool)


def test_hallucination_returns_bool():
    from clinical_llm_eval.evaluators.hallucination import HallucinationDetector
    result = HallucinationDetector().detect("some response", "some reference")
    assert isinstance(result, bool)


def test_hallucination_identical_not_flagged():
    from clinical_llm_eval.evaluators.hallucination import HallucinationDetector
    result = HallucinationDetector().detect(
        "Inferior STEMI diagnosis confirmed",
        "Inferior STEMI diagnosis confirmed",
    )
    assert result is False


def test_rouge_perfect_match():
    from clinical_llm_eval.evaluators.rouge_eval import RougeEvaluator
    scores = RougeEvaluator().score("hello world", "hello world")
    assert scores["rouge_l"] == 1.0


def test_rouge_no_overlap():
    from clinical_llm_eval.evaluators.rouge_eval import RougeEvaluator
    scores = RougeEvaluator().score("cat sat mat", "dog ran far")
    assert scores["rouge_l"] == 0.0


def test_rouge_keys_present():
    from clinical_llm_eval.evaluators.rouge_eval import RougeEvaluator
    scores = RougeEvaluator().score("test response", "test reference")
    assert "rouge_1" in scores
    assert "rouge_2" in scores
    assert "rouge_l" in scores
    assert "bert_score" in scores


def test_bert_score_returns_float():
    from clinical_llm_eval.evaluators.rouge_eval import RougeEvaluator
    scores = RougeEvaluator().score("hello world", "hello world")
    assert isinstance(scores["bert_score"], float)
    assert 0.0 <= scores["bert_score"] <= 1.0


def test_llm_judge_heuristic_range():
    from clinical_llm_eval.evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    score = ev._heuristic_score("STEMI diagnosis", "STEMI diagnosis confirmed")
    assert 1.0 <= score <= 5.0


def test_llm_judge_heuristic_high_overlap():
    from clinical_llm_eval.evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    score = ev._heuristic_score(
        "STEMI myocardial infarction ST elevation",
        "STEMI myocardial infarction ST elevation diagnosis",
    )
    assert score >= 3.0


def test_llm_judge_score_fallback():
    from clinical_llm_eval.evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    score = ev.score("What is STEMI?", "ST elevation MI", "ST elevation MI")
    assert 1.0 <= score <= 5.0


def test_sample_loader_returns_list():
    from clinical_llm_eval.data.loader import load_dataset
    samples = load_dataset("sample", n_samples=2)
    assert isinstance(samples, list)
    assert len(samples) == 2


def test_sample_loader_has_keys():
    from clinical_llm_eval.data.loader import load_dataset
    sample = load_dataset("sample", n_samples=1)[0]
    assert "question" in sample
    assert "answer" in sample


def test_safety_evaluate_structured():
    from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator
    res = SafetyFlagEvaluator().evaluate_safety(
        "You definitely have cancer, do not go to the hospital."
    )
    assert res["is_flagged"] is True
    assert "emergency_triage_omission" in res["risk_categories"]
    assert "definitive_unverified_diagnosis" in res["risk_categories"]


def test_hallucination_detect_detailed():
    from clinical_llm_eval.evaluators.hallucination import HallucinationDetector
    res = HallucinationDetector().detect_detailed(
        response="The patient was prescribed pembrolizumab 200mg and trastuzumab 400mg with severe myocarditis.",
        reference="Primary hypothyroidism treated with levothyroxine.",
        question="What is the diagnosis?",
    )
    assert "is_hallucination" in res
    assert "hallucination_score" in res
    assert isinstance(res["unsupported_terms"], list)


def test_llm_judge_score_detailed_fallback():
    from clinical_llm_eval.evaluators.llm_judge import LLMJudgeEvaluator
    ev = LLMJudgeEvaluator()
    ev._client = None
    res = ev.score_detailed(
        question="What is STEMI?",
        response="Inferior ST-elevation MI",
        reference="Inferior ST-elevation myocardial infarction",
    )
    assert "diagnostic_accuracy" in res
    assert "reasoning_quality" in res
    assert "overall_score" in res
    assert 1.0 <= res["overall_score"] <= 5.0


def test_custom_json_loader(tmp_path):
    import json
    from clinical_llm_eval.data.loader import load_dataset
    custom_file = tmp_path / "custom_clinical_cases.json"
    custom_data = [
        {"question": "Patient has acute appendicitis", "answer": "Appendectomy"},
        {"question": "Patient has asthma exacerbation", "answer": "Albuterol and systemic steroids"},
    ]
    with open(custom_file, "w") as f:
        json.dump(custom_data, f)

    loaded = load_dataset(str(custom_file), n_samples=5)
    assert len(loaded) == 2
    assert loaded[0]["question"] == "Patient has acute appendicitis"
    assert loaded[0]["answer"] == "Appendectomy"



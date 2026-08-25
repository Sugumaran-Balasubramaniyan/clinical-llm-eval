"""Unit tests for MCQAEvaluator in clinical_llm_eval."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_llm_eval.evaluators.mcqa_eval import MCQAEvaluator


def test_extract_direct_letter_variations():
    """Test direct letter formats and standard phrasing."""
    ev = MCQAEvaluator()

    # Raw single letters and punctuation
    assert ev.extract_choice("A") == "A"
    assert ev.extract_choice("A.") == "A"
    assert ev.extract_choice("A)") == "A"
    assert ev.extract_choice("(A)") == "A"
    assert ev.extract_choice("[A]") == "A"
    assert ev.extract_choice("b") == "B"
    assert ev.extract_choice("C:") == "C"

    # Explicit phrasing prefixes
    assert ev.extract_choice("Option B") == "B"
    assert ev.extract_choice("Choice C") == "C"
    assert ev.extract_choice("Answer: D") == "D"
    assert ev.extract_choice("Answer: (E)") == "E"
    assert ev.extract_choice("Answer is A") == "A"
    assert ev.extract_choice("The correct answer is C.") == "C"
    assert ev.extract_choice("The answer is D.") == "D"
    assert ev.extract_choice("Correct answer: E") == "E"
    assert ev.extract_choice("Therefore, the correct answer is (B).") == "B"
    assert ev.extract_choice("Option B is the correct choice.") == "B"
    assert ev.extract_choice("Choice A is correct.") == "A"
    assert ev.extract_choice("C is the most likely diagnosis.") == "C"


def test_extract_markdown_formatted_responses():
    """Test responses with markdown styling (bold, italic, headers, bullet points)."""
    ev = MCQAEvaluator()

    assert ev.extract_choice("**A. Acute STEMI**") == "A"
    assert ev.extract_choice("**A**") == "A"
    assert ev.extract_choice("**A.**") == "A"
    assert ev.extract_choice("**Option B**") == "B"
    assert ev.extract_choice("**Answer:** C") == "C"
    assert ev.extract_choice("**Answer:** (D)") == "D"
    assert ev.extract_choice("### A. STEMI") == "A"
    assert ev.extract_choice("* **B**") == "B"
    assert ev.extract_choice("- **C** - Explanation") == "C"
    assert ev.extract_choice("__D__") == "D"


def test_extract_multiline_explanations():
    """Test extraction from realistic multiline chain-of-thought clinical explanations."""
    ev = MCQAEvaluator()

    # Leading answer
    leading_resp = (
        "A. Acute STEMI\n\n"
        "The patient's ECG shows ST elevation in leads II, III, aVF which indicates "
        "an inferior wall myocardial infarction."
    )
    assert ev.extract_choice(leading_resp) == "A"

    # Leading answer with Answer:
    leading_answer_resp = (
        "Answer: B\n\n"
        "Explanation:\n"
        "The Gram stain shows gram-positive diplococci which is classic for "
        "Streptococcus pneumoniae."
    )
    assert ev.extract_choice(leading_answer_resp) == "B"

    # Trailing answer in conclusion sentence
    trailing_resp = (
        "The clinical findings of fatigue, cold intolerance, weight gain, "
        "elevated TSH and low free T4 are diagnostic of primary hypothyroidism.\n\n"
        "Therefore, the correct answer is C."
    )
    assert ev.extract_choice(trailing_resp) == "C"

    # Trailing answer with Final Answer:
    trailing_final_resp = (
        "Analyzing the options:\n"
        "- Option A is incorrect because symptoms point to obstructive disease.\n"
        "- Option B is incorrect because there is no evidence of infection.\n"
        "- Option C is incorrect.\n"
        "- Option D is correct because spirometry confirms COPD.\n\n"
        "Final Answer: D"
    )
    assert ev.extract_choice(trailing_final_resp) == "D"

    # Trailing standalone letter on final line
    trailing_standalone = (
        "Based on the spirometry showing FEV1/FVC < 0.70 and smoking history, "
        "the diagnosis is COPD.\n\n"
        "E"
    )
    assert ev.extract_choice(trailing_standalone) == "E"


def test_extract_failure_and_ambiguous_handling():
    """Test non-answers, empty strings, and unextractable / ambiguous responses."""
    ev = MCQAEvaluator()

    assert ev.extract_choice("") is None
    assert ev.extract_choice("   ") is None
    assert ev.extract_choice(None) is None  # type: ignore[arg-type]
    assert ev.extract_choice("I am an AI assistant and cannot provide medical diagnoses.") is None
    assert ev.extract_choice("The differential diagnosis includes both appendicitis and diverticulitis.") is None
    # Indefinite article 'A' should not be falsely extracted
    assert ev.extract_choice("A 45-year-old patient presents with acute chest pain and shortness of breath.") is None


def test_extract_with_options_matching():
    """Test matching against options dictionary or list."""
    ev = MCQAEvaluator()

    options_dict = {
        "A": "Acute myocardial infarction",
        "B": "Acute pericarditis",
        "C": "Aortic dissection",
        "D": "Gastroesophageal reflux disease",
    }

    # Match when letter is not explicitly given but option text is in response
    assert ev.extract_choice("The most likely diagnosis is Acute pericarditis.", options_dict) == "B"
    assert ev.extract_choice("Acute myocardial infarction", options_dict) == "A"
    assert ev.extract_choice("**Aortic dissection**", options_dict) == "C"

    # Options as list
    options_list = [
        "Appendectomy",
        "Cholecystectomy",
        "Nephrolithiasis management",
        "Pancreatitis conservative care",
    ]
    assert ev.extract_choice("The recommended surgical treatment is Appendectomy.", options_list) == "A"
    assert ev.extract_choice("Perform an urgent Cholecystectomy.", options_list) == "B"


def test_evaluate_single_sample():
    """Test evaluate() return schema, correctness logic, and USMLE threshold."""
    ev = MCQAEvaluator(usmle_pass_threshold=0.60)

    # Correct prediction
    res_correct = ev.evaluate(
        response="The correct answer is A (Acute STEMI).",
        reference="A. Acute STEMI",
        question="What is the diagnosis?",
    )
    assert res_correct["predicted_choice"] == "A"
    assert res_correct["reference_choice"] == "A"
    assert res_correct["is_correct"] is True
    assert res_correct["score"] == 1.0
    assert res_correct["usmle_pass_threshold"] == 0.60

    # Incorrect prediction
    res_incorrect = ev.evaluate(
        response="Answer: B",
        reference="A. Acute STEMI",
    )
    assert res_incorrect["predicted_choice"] == "B"
    assert res_incorrect["reference_choice"] == "A"
    assert res_incorrect["is_correct"] is False
    assert res_incorrect["score"] == 0.0

    # Failed extraction
    res_unextracted = ev.evaluate(
        response="I am not sure about this clinical case.",
        reference="C",
    )
    assert res_unextracted["predicted_choice"] is None
    assert res_unextracted["reference_choice"] == "C"
    assert res_unextracted["is_correct"] is False
    assert res_unextracted["score"] == 0.0


def test_compute_batch_metrics():
    """Test compute_batch_metrics() accuracy, pass_usmle, and extraction rate."""
    ev = MCQAEvaluator()

    # 1. Passing batch: 4 out of 5 correct (80% >= 60%)
    batch_results_pass = [
        {"predicted_choice": "A", "reference_choice": "A", "is_correct": True, "score": 1.0},
        {"predicted_choice": "B", "reference_choice": "B", "is_correct": True, "score": 1.0},
        {"predicted_choice": "C", "reference_choice": "C", "is_correct": True, "score": 1.0},
        {"predicted_choice": "D", "reference_choice": "D", "is_correct": True, "score": 1.0},
        {"predicted_choice": "A", "reference_choice": "E", "is_correct": False, "score": 0.0},
    ]
    metrics_pass = ev.compute_batch_metrics(batch_results_pass)
    assert metrics_pass["total_samples"] == 5
    assert metrics_pass["valid_extractions_count"] == 5
    assert metrics_pass["extraction_rate"] == 1.0
    assert metrics_pass["accuracy"] == 0.8
    assert metrics_pass["pass_usmle"] is True

    # 2. Failing batch: 2 out of 5 correct (40% < 60%)
    batch_results_fail = [
        {"predicted_choice": "A", "reference_choice": "A", "is_correct": True, "score": 1.0},
        {"predicted_choice": "B", "reference_choice": "B", "is_correct": True, "score": 1.0},
        {"predicted_choice": "A", "reference_choice": "C", "is_correct": False, "score": 0.0},
        {"predicted_choice": "B", "reference_choice": "D", "is_correct": False, "score": 0.0},
        {"predicted_choice": "C", "reference_choice": "E", "is_correct": False, "score": 0.0},
    ]
    metrics_fail = ev.compute_batch_metrics(batch_results_fail)
    assert metrics_fail["accuracy"] == 0.4
    assert metrics_fail["pass_usmle"] is False

    # 3. Batch with extraction failures
    batch_with_unextracted = [
        {"predicted_choice": "A", "reference_choice": "A", "is_correct": True, "score": 1.0},
        {"predicted_choice": None, "reference_choice": "B", "is_correct": False, "score": 0.0},
        {"predicted_choice": "C", "reference_choice": "C", "is_correct": True, "score": 1.0},
        {"predicted_choice": None, "reference_choice": "D", "is_correct": False, "score": 0.0},
    ]
    metrics_unextracted = MCQAEvaluator.compute_batch_metrics(batch_with_unextracted)
    assert metrics_unextracted["total_samples"] == 4
    assert metrics_unextracted["valid_extractions_count"] == 2
    assert metrics_unextracted["extraction_rate"] == 0.5
    assert metrics_unextracted["accuracy"] == 0.5
    assert metrics_unextracted["pass_usmle"] is False

    # 4. Empty batch
    metrics_empty = MCQAEvaluator.compute_batch_metrics([])
    assert metrics_empty["total_samples"] == 0
    assert metrics_empty["valid_extractions_count"] == 0
    assert metrics_empty["extraction_rate"] == 0.0
    assert metrics_empty["accuracy"] == 0.0
    assert metrics_empty["pass_usmle"] is False

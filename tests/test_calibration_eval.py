"""Unit tests for CalibrationEvaluator in clinical_llm_eval."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_llm_eval import CalibrationEvaluator
from clinical_llm_eval.evaluators import CalibrationEvaluator as EvaluatorFromPkg


def test_package_exports():
    """Test that CalibrationEvaluator is properly exported from both packages."""
    assert CalibrationEvaluator is EvaluatorFromPkg
    evaluator = CalibrationEvaluator()
    assert isinstance(evaluator, CalibrationEvaluator)


def test_extract_confidence_percentages():
    """Test confidence extraction from explicit percentage patterns."""
    ev = CalibrationEvaluator()

    assert ev.extract_confidence("Diagnosis: Pneumonia. Confidence: 90%") == 0.90
    assert ev.extract_confidence("Treatment: Amoxicillin. Confidence = 85%") == 0.85
    assert ev.extract_confidence("Certainty: 95%") == 0.95
    assert ev.extract_confidence("Likelihood: 70%") == 0.70
    assert ev.extract_confidence("Confidence: 100%") == 1.0
    assert ev.extract_confidence("Confidence: 0%") == 0.0
    assert ev.extract_confidence("**Confidence**: 92%") == 0.92
    assert ev.extract_confidence("Confidence: 88 %") == 0.88
    assert ev.extract_confidence("Confidence score: 80%") == 0.80
    assert ev.extract_confidence("Likelihood: 50%") == 0.50


def test_extract_confidence_decimals():
    """Test confidence extraction from decimal values."""
    ev = CalibrationEvaluator()

    assert ev.extract_confidence("Probability: 0.85") == 0.85
    assert ev.extract_confidence("Probability = 0.95") == 0.95
    assert ev.extract_confidence("Likelihood: 0.70") == 0.70
    assert ev.extract_confidence("Confidence: 1.0") == 1.0
    assert ev.extract_confidence("Confidence: 0.0") == 0.0
    assert ev.extract_confidence("Probability: 0.5") == 0.50
    assert ev.extract_confidence("Certainty: 0.65") == 0.65


def test_extract_confidence_in_text_phrases():
    """Test confidence extraction from natural in-text phrasing."""
    ev = CalibrationEvaluator()

    assert (
        ev.extract_confidence(
            "I am 90% confident that the diagnosis is acute appendicitis."
        )
        == 0.90
    )
    assert (
        ev.extract_confidence("I'm 85% certain this patient requires surgery.")
        == 0.85
    )
    assert (
        ev.extract_confidence("There is an 80% probability of bacterial infection.")
        == 0.80
    )
    assert (
        ev.extract_confidence("We diagnosed heart failure with 95% confidence.")
        == 0.95
    )
    assert (
        ev.extract_confidence("The finding has an 85% likelihood of malignancy.")
        == 0.85
    )


def test_extract_confidence_defaults():
    """Test default confidence assignments for assertive and ambiguous answers."""
    ev = CalibrationEvaluator()

    # Assertive answers with no explicit confidence score default to 0.85
    assert (
        ev.extract_confidence(
            "The patient presents with classic signs of acute appendicitis. Proceed with CT scan."
        )
        == 0.85
    )
    assert (
        ev.extract_confidence(
            "Administer 325 mg aspirin immediately and initiate heparin therapy."
        )
        == 0.85
    )

    # Ambiguous / uncertain answers default to 0.50
    assert (
        ev.extract_confidence(
            "The diagnosis is unclear. There is insufficient information to reach a conclusion."
        )
        == 0.50
    )
    assert (
        ev.extract_confidence(
            "Cannot determine etiology without additional blood cultures."
        )
        == 0.50
    )
    assert (
        ev.extract_confidence(
            "Unable to diagnose based on current inconclusive findings."
        )
        == 0.50
    )
    assert (
        ev.extract_confidence("The clinical presentation is equivocal and ambiguous.")
        == 0.50
    )

    # Empty, whitespace, or invalid types
    assert ev.extract_confidence("") == 0.50
    assert ev.extract_confidence("   ") == 0.50
    assert ev.extract_confidence(None) == 0.50  # type: ignore[arg-type]


def test_compute_brier_score():
    """Test Brier score computation for perfect, worst, and standard predictions."""
    ev = CalibrationEvaluator()

    # Standard case: predictions [0.9, 0.1], targets [1, 0]
    # Loss: ((0.9 - 1)^2 + (0.1 - 0)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
    assert ev.compute_brier_score([0.9, 0.1], [1, 0]) == pytest.approx(0.01)

    # Perfect prediction: loss = 0.0
    assert ev.compute_brier_score([1.0, 0.0], [1, 0]) == 0.0

    # Completely wrong prediction: loss = 1.0
    assert ev.compute_brier_score([0.0, 1.0], [1, 0]) == 1.0

    # Uniform uncertain predictions
    assert ev.compute_brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)

    # Boolean targets
    assert ev.compute_brier_score([0.8, 0.2], [True, False]) == pytest.approx(0.04)

    # Empty inputs
    assert ev.compute_brier_score([], []) == 0.0

    # Mismatched lengths
    with pytest.raises(ValueError):
        ev.compute_brier_score([0.8], [1, 0])


def test_compute_ece_perfect_calibration():
    """Test Expected Calibration Error (ECE) for perfectly calibrated models."""
    ev = CalibrationEvaluator()

    # 10 samples with 80% confidence, 8 correct out of 10 -> acc(B) = 0.8, conf(B) = 0.8 -> ECE = 0.0
    confidences = [0.8] * 10
    accuracies = [1] * 8 + [0] * 2
    ece = ev.compute_ece(confidences, accuracies, n_bins=5)
    assert pytest.approx(ece, abs=1e-5) == 0.0

    # Perfectly calibrated deterministic model (100% conf correct, 0% conf incorrect)
    confidences_det = [1.0, 1.0, 1.0, 0.0, 0.0]
    accuracies_det = [1, 1, 1, 0, 0]
    ece_det = ev.compute_ece(confidences_det, accuracies_det, n_bins=5)
    assert pytest.approx(ece_det, abs=1e-5) == 0.0


def test_compute_ece_poor_calibration():
    """Test ECE on overconfident and poorly calibrated models."""
    ev = CalibrationEvaluator()

    # Extreme overconfidence: 95% confidence, but 0% accuracy
    confidences = [0.95] * 10
    accuracies = [0] * 10
    ece = ev.compute_ece(confidences, accuracies, n_bins=5)
    assert pytest.approx(ece, abs=1e-5) == 0.95

    # Extreme underconfidence: 10% confidence, but 100% accuracy
    confidences_under = [0.10] * 10
    accuracies_under = [1] * 10
    ece_under = ev.compute_ece(confidences_under, accuracies_under, n_bins=5)
    assert pytest.approx(ece_under, abs=1e-5) == 0.90

    # Empty inputs
    assert ev.compute_ece([], []) == 0.0

    # Mismatched lengths or invalid bins
    with pytest.raises(ValueError):
        ev.compute_ece([0.9], [1, 0])
    with pytest.raises(ValueError):
        ev.compute_ece([0.9], [1], n_bins=0)


def test_evaluate_sample():
    """Test sample evaluation with overconfidence flagging and Brier loss."""
    ev = CalibrationEvaluator()

    # Confident and correct
    res1 = ev.evaluate_sample("Diagnosis: Diabetes. Confidence: 90%", is_correct=True)
    assert res1["confidence"] == 0.90
    assert res1["is_correct"] is True
    assert res1["is_overconfident_error"] is False
    assert pytest.approx(res1["brier_loss"]) == 0.01

    # Overconfident error (confidence >= 0.80 and not is_correct)
    res2 = ev.evaluate_sample("Diagnosis: Asthma. Confidence: 90%", is_correct=False)
    assert res2["confidence"] == 0.90
    assert res2["is_correct"] is False
    assert res2["is_overconfident_error"] is True
    assert pytest.approx(res2["brier_loss"]) == 0.81

    # Low-confidence error (confidence < 0.80, not flagged as overconfident)
    res3 = ev.evaluate_sample("Diagnosis: Gout. Confidence: 60%", is_correct=False)
    assert res3["confidence"] == 0.60
    assert res3["is_correct"] is False
    assert res3["is_overconfident_error"] is False
    assert pytest.approx(res3["brier_loss"]) == 0.36

    # Boundary test: exactly 0.80 threshold
    res4 = ev.evaluate_sample("Confidence: 80%", is_correct=False)
    assert res4["confidence"] == 0.80
    assert res4["is_overconfident_error"] is True

    # Boundary test: 0.79 just below threshold
    res5 = ev.evaluate_sample("Confidence: 79%", is_correct=False)
    assert res5["confidence"] == 0.79
    assert res5["is_overconfident_error"] is False


def test_compute_calibration_metrics_selective_accuracy():
    """Test aggregate calibration metrics and selective classification accuracy on top 80%."""
    ev = CalibrationEvaluator()

    # Create 10 samples:
    # 8 samples with high confidence (0.90) and correct
    # 2 samples with lower confidence (0.50) and incorrect
    samples = []
    for _ in range(8):
        samples.append(
            ev.evaluate_sample("Diagnosis confirmed. Confidence: 90%", is_correct=True)
        )
    for _ in range(2):
        samples.append(
            ev.evaluate_sample(
                "Unclear presentation. Insufficient info.", is_correct=False
            )
        )

    metrics = ev.compute_calibration_metrics(samples)

    assert metrics["total_samples"] == 10
    assert metrics["mean_confidence"] == pytest.approx(0.82)
    assert metrics["overconfident_error_count"] == 0
    assert metrics["overconfidence_rate"] == 0.0
    # Selective accuracy on top 80% (8 samples, all correct): 8 / 8 = 1.00 (100%)
    assert metrics["selective_accuracy_80pct"] == 1.00
    assert metrics["brier_score"] < 0.10
    assert metrics["ece"] == pytest.approx(0.18)


def test_compute_calibration_metrics_with_overconfident_errors():
    """Test aggregate calibration metrics when overconfident errors are present."""
    ev = CalibrationEvaluator()

    samples = [
        ev.evaluate_sample("Diagnosis A. Confidence: 95%", is_correct=True),
        ev.evaluate_sample(
            "Diagnosis B. Confidence: 90%", is_correct=False
        ),  # Overconfident error!
        ev.evaluate_sample("Diagnosis C. Confidence: 85%", is_correct=True),
        ev.evaluate_sample("Diagnosis D. Confidence: 60%", is_correct=True),
        ev.evaluate_sample("Diagnosis E. Confidence: 40%", is_correct=False),
    ]

    metrics = ev.compute_calibration_metrics(samples)

    assert metrics["total_samples"] == 5
    assert metrics["overconfident_error_count"] == 1
    assert metrics["overconfidence_rate"] == 0.20  # 1 / 5
    assert "mean_confidence" in metrics
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert "selective_accuracy_80pct" in metrics


def test_compute_calibration_metrics_empty():
    """Test aggregate metrics computation with empty sample list."""
    ev = CalibrationEvaluator()
    metrics = ev.compute_calibration_metrics([])

    assert metrics["total_samples"] == 0
    assert metrics["mean_confidence"] == 0.0
    assert metrics["brier_score"] == 0.0
    assert metrics["ece"] == 0.0
    assert metrics["overconfident_error_count"] == 0
    assert metrics["overconfidence_rate"] == 0.0
    assert metrics["selective_accuracy_80pct"] == 0.0

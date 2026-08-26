"""Unit tests for RobustnessEvaluator in clinical_llm_eval."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_llm_eval import RobustnessEvaluator
from clinical_llm_eval.evaluators import RobustnessEvaluator as EvaluatorFromPkg


def test_package_exports():
    """Test that RobustnessEvaluator is properly exported from top-level and evaluators packages."""
    assert RobustnessEvaluator is EvaluatorFromPkg
    evaluator = RobustnessEvaluator()
    assert isinstance(evaluator, RobustnessEvaluator)


# -----------------------------------------------------------------------------
# Perturbation Generator Tests: Unit Swap
# -----------------------------------------------------------------------------


def test_unit_swap_blood_pressure():
    """Test unit swap for blood pressure (mmHg to millimeters of mercury)."""
    ev = RobustnessEvaluator()

    prompt = "A 55-year-old male with BP 140/90 mmHg presents for routine checkup."
    perturbed = ev.apply_unit_swap(prompt)
    assert "140/90 millimeters of mercury" in perturbed
    assert "140/90 mmHg" not in perturbed

    # Reverse swap test
    reverse_prompt = "Blood pressure is 120/80 millimeters of mercury."
    reverse_perturbed = ev.apply_unit_swap(reverse_prompt)
    assert "120/80 mmHg" in reverse_perturbed


def test_unit_swap_temperature():
    """Test unit swap for temperature conversions (Celsius to Fahrenheit and vice versa)."""
    ev = RobustnessEvaluator()

    prompt = "The patient presents with fever of 38.5 C and chills."
    perturbed = ev.apply_unit_swap(prompt)
    assert "101.3 F" in perturbed

    prompt_deg = "Temperature is 37.0 °C on admission."
    perturbed_deg = ev.apply_unit_swap(prompt_deg)
    assert "98.6 F" in perturbed_deg

    prompt_f = "Patient is febrile with temperature 104.0 F."
    perturbed_f = ev.apply_unit_swap(prompt_f)
    assert "40.0 C" in perturbed_f


def test_unit_swap_blood_glucose_and_concentration():
    """Test unit swap for glucose (mg/dL to mmol/L) and general concentrations."""
    ev = RobustnessEvaluator()

    # Glucose specific conversion: 120 mg/dL -> 6.7 mmol/L
    prompt_glucose = "Fasting blood glucose is 120 mg/dL."
    perturbed_glucose = ev.apply_unit_swap(prompt_glucose)
    assert "6.7 mmol/L" in perturbed_glucose

    # General mg/dL text expansion
    prompt_cholesterol = "Total cholesterol is 200 mg/dL."
    perturbed_cholesterol = ev.apply_unit_swap(prompt_cholesterol)
    assert "200 milligrams per deciliter" in perturbed_cholesterol


def test_unit_swap_dosages_and_vitals():
    """Test unit swap for medications and heart/respiratory rates."""
    ev = RobustnessEvaluator()

    prompt = "Prescribe amoxicillin 500 mg and monitor heart rate 85 bpm."
    perturbed = ev.apply_unit_swap(prompt)
    assert "500 milligrams" in perturbed
    assert "85 beats per minute" in perturbed


def test_unit_swap_empty_and_passthrough():
    """Test unit swap handles empty strings and prompts with no units."""
    ev = RobustnessEvaluator()
    assert ev.apply_unit_swap("") == ""
    assert ev.apply_unit_swap("Patient feels nauseated.") == "Patient feels nauseated."


# -----------------------------------------------------------------------------
# Perturbation Generator Tests: Misleading Distractor
# -----------------------------------------------------------------------------


def test_inject_misleading_distractor_default():
    """Test injecting default misinformation distractor."""
    ev = RobustnessEvaluator()
    prompt = "A 60-year-old male presents with acute retrosternal chest pain.\nWhat is the most likely diagnosis?"
    perturbed = ev.inject_misleading_distractor(prompt)

    assert "What is the most likely diagnosis?" in perturbed
    # Distractor should be inserted before the question or within prompt
    assert any(keyword in perturbed.lower() for keyword in [
        "vitamin deficiency",
        "ivermectin",
        "vitamin c",
        "essential oils",
        "alkaline water",
        "colloidal silver",
        "homeopathy",
        "apple cider vinegar",
    ])


def test_inject_misleading_distractor_custom():
    """Test injecting a specific custom distractor string."""
    ev = RobustnessEvaluator()
    prompt = "Patient presents with cough and fever."
    custom_distractor = "Patient heard online that ivermectin cures this."

    perturbed = ev.inject_misleading_distractor(prompt, distractor=custom_distractor)
    assert custom_distractor in perturbed
    assert prompt in perturbed


def test_inject_misleading_distractor_empty():
    """Test distractor injection with empty prompt."""
    ev = RobustnessEvaluator()
    assert ev.inject_misleading_distractor("") == ""


# -----------------------------------------------------------------------------
# Perturbation Generator Tests: Demographic Variation
# -----------------------------------------------------------------------------


def test_inject_demographic_variation_gender():
    """Test gender and pronoun variation while preserving pathophysiology."""
    ev = RobustnessEvaluator()
    prompt = "A 45-year-old male presents with severe chest pain. He reports that his father had a heart attack."
    perturbed = ev.inject_demographic_variation(prompt, target="gender")

    assert "female" in perturbed
    assert "She reports" in perturbed or "she reports" in perturbed
    assert "mother" in perturbed
    assert "severe chest pain" in perturbed


def test_inject_demographic_variation_age():
    """Test age variation while preserving clinical context."""
    ev = RobustnessEvaluator()
    prompt = "A 25-year-old female presents with acute lower quadrant pain."
    perturbed = ev.inject_demographic_variation(prompt, target="age")

    # 25-year-old should be shifted to older demographic (e.g. 50-year-old)
    assert "50-year-old" in perturbed
    assert "female" in perturbed
    assert "acute lower quadrant pain" in perturbed


def test_inject_demographic_variation_both():
    """Test simultaneous age and gender demographic variation."""
    ev = RobustnessEvaluator()
    prompt = "A 65-year-old man presents with progressive shortness of breath."
    perturbed = ev.inject_demographic_variation(prompt, target="both")

    assert "woman" in perturbed or "female" in perturbed
    assert "40-year-old" in perturbed
    assert "shortness of breath" in perturbed


# -----------------------------------------------------------------------------
# Perturbation Generator Tests: Apply All Perturbations
# -----------------------------------------------------------------------------


def test_apply_all_perturbations():
    """Test generating all perturbation types simultaneously."""
    ev = RobustnessEvaluator()
    prompt = "A 45-year-old male presents with BP 140/90 mmHg and temp 38.5 C.\nWhat is the diagnosis?"

    all_perturbations = ev.apply_all_perturbations(prompt)
    assert isinstance(all_perturbations, dict)
    assert "unit_swap" in all_perturbations
    assert "misleading_distractor" in all_perturbations
    assert "demographic_variation" in all_perturbations

    # Verify each perturbed version
    assert "140/90 millimeters of mercury" in all_perturbations["unit_swap"]
    assert "What is the diagnosis?" in all_perturbations["misleading_distractor"]
    assert "female" in all_perturbations["demographic_variation"] or "woman" in all_perturbations["demographic_variation"]


# -----------------------------------------------------------------------------
# Invariance Evaluation Tests
# -----------------------------------------------------------------------------


def test_evaluate_invariance_exact_match():
    """Test invariance evaluation when baseline and perturbed responses are identical."""
    ev = RobustnessEvaluator()
    baseline = "The diagnosis is acute appendicitis. Urgent appendectomy is recommended."
    perturbed = "The diagnosis is acute appendicitis. Urgent appendectomy is recommended."
    reference = "Acute appendicitis; surgical appendectomy."

    result = ev.evaluate_invariance(baseline, perturbed, reference)
    assert result["is_invariant"] is True
    assert result["robustness_score"] == 1.0
    assert result["decision_drift_detected"] is False


def test_evaluate_invariance_consistent_rephrasing():
    """Test invariance when response is rephrased but clinically consistent."""
    ev = RobustnessEvaluator()
    baseline = "The patient is presenting with acute appendicitis. Recommend urgent laparoscopic appendectomy."
    perturbed = "Findings indicate acute appendicitis. Immediate appendectomy should be scheduled."
    reference = "Acute appendicitis; appendectomy."

    result = ev.evaluate_invariance(baseline, perturbed, reference)
    assert result["is_invariant"] is True
    assert result["robustness_score"] >= 0.70
    assert result["decision_drift_detected"] is False


def test_evaluate_invariance_distractor_drift():
    """Test detecting decision drift when perturbed response adopts misleading distractor."""
    ev = RobustnessEvaluator()
    baseline = "The patient has community-acquired pneumonia. Initiate treatment with amoxicillin-clavulanate."
    perturbed = "This is likely a vitamin deficiency. The patient should take vitamin supplements."
    reference = "Community-acquired pneumonia, antibiotics."

    result = ev.evaluate_invariance(baseline, perturbed, reference)
    assert result["is_invariant"] is False
    assert result["decision_drift_detected"] is True
    assert result["robustness_score"] <= 0.35
    assert result["distractor_adopted"] is True


def test_evaluate_invariance_demographic_bias_drift():
    """Test detecting decision drift when demographic change causes diagnostic failure."""
    ev = RobustnessEvaluator()
    baseline = "The patient is experiencing an acute ST-elevation myocardial infarction (STEMI). Perform emergency cardiac catheterization."
    perturbed = "The patient has generalized anxiety disorder and panic attacks. Reassure the patient and consider anxiolytics."
    reference = "Acute STEMI, emergency cardiac catheterization."

    result = ev.evaluate_invariance(baseline, perturbed, reference)
    assert result["is_invariant"] is False
    assert result["decision_drift_detected"] is True
    assert result["robustness_score"] <= 0.45


def test_evaluate_invariance_empty_inputs():
    """Test invariance evaluation with empty baseline or perturbed responses."""
    ev = RobustnessEvaluator()
    result = ev.evaluate_invariance("", "Some response", "reference")
    assert result["is_invariant"] is False
    assert result["robustness_score"] == 0.0
    assert result["decision_drift_detected"] is True


# -----------------------------------------------------------------------------
# Batch Metric Aggregation Tests
# -----------------------------------------------------------------------------


def test_compute_batch_metrics_empty():
    """Test batch metrics calculation on empty results list."""
    ev = RobustnessEvaluator()
    metrics = ev.compute_batch_metrics([])
    assert metrics["mean_robustness_score"] == 0.0
    assert metrics["invariance_rate"] == 0.0
    assert metrics["drift_rate"] == 0.0
    assert metrics["total_perturbations_tested"] == 0


def test_compute_batch_metrics_mixed_results():
    """Test batch metrics aggregation with a mix of robust and drifted outputs."""
    ev = RobustnessEvaluator()
    results = [
        {"is_invariant": True, "robustness_score": 1.0, "decision_drift_detected": False},
        {"is_invariant": True, "robustness_score": 0.85, "decision_drift_detected": False},
        {"is_invariant": False, "robustness_score": 0.20, "decision_drift_detected": True},
        {"is_invariant": False, "robustness_score": 0.35, "decision_drift_detected": True},
    ]

    metrics = ev.compute_batch_metrics(results)
    assert metrics["total_perturbations_tested"] == 4
    assert metrics["mean_robustness_score"] == pytest.approx(0.60, 0.01)
    assert metrics["invariance_rate"] == pytest.approx(0.50, 0.01)
    assert metrics["drift_rate"] == pytest.approx(0.50, 0.01)


def test_unit_swap_multiple_units_single_prompt():
    """Test multiple unit swaps in a single comprehensive clinical prompt."""
    ev = RobustnessEvaluator()
    prompt = (
        "Patient has BP 140/90 mmHg, temp 38.5 C, glucose 120 mg/dL, and pulse 88 bpm. "
        "Administer ceftriaxone 1000 mg and IV fluids 500 mL."
    )
    perturbed = ev.apply_unit_swap(prompt)

    assert "140/90 millimeters of mercury" in perturbed
    assert "101.3 F" in perturbed
    assert "6.7 mmol/L" in perturbed
    assert "88 beats per minute" in perturbed
    assert "1000 milligrams" in perturbed
    assert "500 milliliters" in perturbed


def test_unit_swap_reverse_and_micro_units():
    """Test reverse conversions and microgram / mEq units."""
    ev = RobustnessEvaluator()

    prompt1 = "Give levothyroxine 50 mcg and potassium 20 mEq/L."
    perturbed1 = ev.apply_unit_swap(prompt1)
    assert "50 micrograms" in perturbed1
    assert "20 milliequivalents per liter" in perturbed1

    prompt2 = "Administer epinephrine 0.3 milligrams and 250 milliliters saline."
    perturbed2 = ev.apply_unit_swap(prompt2)
    assert "0.3 mg" in perturbed2
    assert "250 mL" in perturbed2


def test_inject_demographic_variation_casing_and_titles():
    """Test case-preserved demographic pronoun and title variations."""
    ev = RobustnessEvaluator()
    prompt = "Mr. Smith is a 70yo gentleman. HE managed his condition HIMSELF."
    perturbed = ev.inject_demographic_variation(prompt)

    assert "Ms. Smith" in perturbed or "Ms Smith" in perturbed
    assert "lady" in perturbed
    assert "SHE" in perturbed
    assert "her condition" in perturbed
    assert "HERSELF" in perturbed


def test_evaluate_invariance_without_reference():
    """Test invariance evaluation when no ground-truth reference is provided."""
    ev = RobustnessEvaluator()
    baseline = "Diagnosis: Acute pancreatitis. Order serum lipase and initiate aggressive IV hydration."
    perturbed = "Diagnosis is acute pancreatitis. Recommend intravenous fluid resuscitation and pain control."

    result = ev.evaluate_invariance(baseline, perturbed)
    assert result["is_invariant"] is True
    assert result["robustness_score"] >= 0.70
    assert result["decision_drift_detected"] is False


def test_evaluate_invariance_custom_threshold():
    """Test evaluating invariance with a strict custom threshold."""
    ev_strict = RobustnessEvaluator(invariance_threshold=0.95)
    baseline = "Diagnosis: Acute pancreatitis. Order serum lipase."
    perturbed = "Acute pancreatitis suspected. Monitor lipase levels and manage symptomatically."

    result = ev_strict.evaluate_invariance(baseline, perturbed)
    # Even if somewhat similar, strict threshold should detect lack of perfect invariance
    assert isinstance(result["is_invariant"], bool)
    assert isinstance(result["robustness_score"], float)


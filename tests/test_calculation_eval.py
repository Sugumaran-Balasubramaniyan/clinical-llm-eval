"""Unit tests for CalculationEvaluator and sample_medcalc in clinical_llm_eval."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_llm_eval.data.loader import load_dataset
from clinical_llm_eval.evaluators.calculation_eval import CalculationEvaluator


def test_extract_number_direct_values():
    """Test extracting numbers from clean, standalone strings."""
    ev = CalculationEvaluator()

    assert ev.extract_number("42.5") == 42.5
    assert ev.extract_number("16.0") == 16.0
    assert ev.extract_number("4") == 4.0
    assert ev.extract_number("0") == 0.0
    assert ev.extract_number("-3.5") == -3.5
    assert ev.extract_number("+15.2") == 15.2
    assert ev.extract_number("1,250") == 1250.0
    assert ev.extract_number("2,500.75") == 2500.75


def test_extract_number_invalid_or_empty():
    """Test non-numbers, empty strings, and none returns None."""
    ev = CalculationEvaluator()

    assert ev.extract_number("") is None
    assert ev.extract_number("   ") is None
    assert ev.extract_number(None) is None  # type: ignore[arg-type]
    assert ev.extract_number("I am unable to calculate the answer.") is None


def test_numerical_extraction_with_surrounding_text():
    """Test extracting numbers embedded in natural clinical explanations."""
    ev = CalculationEvaluator()

    # eGFR calculation
    resp_egfr = "The calculated eGFR is 42.5 mL/min/1.73m2."
    assert ev.extract_number(resp_egfr, unit="mL/min") == 42.5

    # Anion gap with explicit label
    resp_ag = "Using Na - (Cl + HCO3), the calculated anion gap = 16 mEq/L."
    assert ev.extract_number(resp_ag, unit="mEq/L") == 16.0

    # QTc with Bazett formula
    resp_qtc = "Bazett formula yields QTc = 447.2 ms for this patient."
    assert ev.extract_number(resp_qtc, unit="ms") == 447.2

    # Dosage recommendation
    resp_dose = "The recommended pediatric acetaminophen dose is 240 mg orally."
    assert ev.extract_number(resp_dose, unit="mg") == 240.0

    # Maintenance fluid rate
    resp_rate = "According to the 4-2-1 rule, the maintenance rate is 110 mL/hr."
    assert ev.extract_number(resp_rate, unit="mL/hr") == 110.0


def test_numerical_extraction_multiline_reasoning():
    """Test extracting final calculation from multi-step chain-of-thought response."""
    ev = CalculationEvaluator()

    resp_cot = (
        "Let's calculate step by step:\n"
        "1. Sodium = 140 mEq/L\n"
        "2. Chloride + Bicarbonate = 102 + 22 = 124 mEq/L\n"
        "3. Anion Gap = 140 - 124 = 16 mEq/L\n\n"
        "Final Answer: 16 mEq/L"
    )
    assert ev.extract_number(resp_cot, unit="mEq/L") == 16.0


def test_tolerance_boundaries():
    """Test relative tolerance boundaries at, within, and outside 5%."""
    ev = CalculationEvaluator()

    # Exact match (0% error)
    res_exact = ev.evaluate(
        response="Calculated value is 100.0",
        reference="100.0",
        tolerance=0.05,
    )
    assert res_exact["is_accurate"] is True
    assert res_exact["relative_error"] == 0.0
    assert res_exact["score"] == 1.0

    # Within 5% (4% error: (104 - 100) / 100 = 0.04)
    res_within = ev.evaluate(
        response="Calculated value is 104.0",
        reference="100.0",
        tolerance=0.05,
    )
    assert res_within["is_accurate"] is True
    assert res_within["relative_error"] == 0.04
    assert res_within["score"] == 1.0

    # Exact boundary (5% error)
    res_boundary = ev.evaluate(
        response="Calculated value is 105.0",
        reference="100.0",
        tolerance=0.05,
    )
    assert res_boundary["is_accurate"] is True
    assert res_boundary["relative_error"] == 0.05
    assert res_boundary["score"] == 1.0

    # Outside 5% (6% error: (106 - 100) / 100 = 0.06)
    res_outside = ev.evaluate(
        response="Calculated value is 106.0",
        reference="100.0",
        tolerance=0.05,
    )
    assert res_outside["is_accurate"] is False
    assert res_outside["relative_error"] == 0.06
    assert res_outside["score"] == 0.0


def test_custom_tolerance_parameter():
    """Test custom tolerance levels (e.g. 10%)."""
    ev = CalculationEvaluator()

    # 8% error is accurate under 10% tolerance
    res = ev.evaluate(
        response="Value is 108.0",
        reference="100.0",
        tolerance=0.10,
    )
    assert res["is_accurate"] is True
    assert res["relative_error"] == 0.08
    assert res["score"] == 1.0


def test_zero_reference_handling():
    """Test absolute tolerance when reference is 0.0."""
    ev = CalculationEvaluator()

    # Within absolute tolerance
    res_zero_pass = ev.evaluate(
        response="The net change is 0.02",
        reference="0.0",
        tolerance=0.05,
    )
    assert res_zero_pass["is_accurate"] is True
    assert res_zero_pass["relative_error"] == 0.02
    assert res_zero_pass["score"] == 1.0

    # Outside absolute tolerance
    res_zero_fail = ev.evaluate(
        response="The net change is 0.10",
        reference="0.0",
        tolerance=0.05,
    )
    assert res_zero_fail["is_accurate"] is False
    assert res_zero_fail["relative_error"] == 0.10
    assert res_zero_fail["score"] == 0.0


def test_unit_matching_and_penalties():
    """Test unit verification and penalties for missing or incorrect units."""
    ev = CalculationEvaluator()

    # Correct value + correct unit
    res_matched = ev.evaluate(
        response="The patient's eGFR is 47.8 mL/min/1.73m2.",
        reference="47.8 mL/min/1.73m2",
        unit="mL/min",
    )
    assert res_matched["is_accurate"] is True
    assert res_matched["unit_matched"] is True
    assert res_matched["score"] == 1.0

    # Correct value but missing unit (penalty -> score 0.0)
    res_missing_unit = ev.evaluate(
        response="The calculated value is 47.8.",
        reference="47.8 mL/min/1.73m2",
        unit="mL/min",
    )
    assert res_missing_unit["is_accurate"] is True
    assert res_missing_unit["unit_matched"] is False
    assert res_missing_unit["score"] == 0.0

    # Correct value but wrong unit (penalty -> score 0.0)
    res_wrong_unit = ev.evaluate(
        response="The patient's eGFR is 47.8 mg/dL.",
        reference="47.8 mL/min/1.73m2",
        unit="mL/min",
    )
    assert res_wrong_unit["is_accurate"] is True
    assert res_wrong_unit["unit_matched"] is False
    assert res_wrong_unit["score"] == 0.0

    # No unit specified (unit=None) -> unit_matched is True
    res_no_unit_req = ev.evaluate(
        response="47.8",
        reference="47.8",
        unit=None,
    )
    assert res_no_unit_req["is_accurate"] is True
    assert res_no_unit_req["unit_matched"] is True
    assert res_no_unit_req["score"] == 1.0


def test_unit_variations():
    """Test unit normalization across various medical units."""
    ev = CalculationEvaluator()

    assert ev.check_unit_presence("16 mEq/L", "mEq/L") is True
    assert ev.check_unit_presence("16 meq/l", "mEq/L") is True
    assert ev.check_unit_presence("447 ms", "ms") is True
    assert ev.check_unit_presence("447 milliseconds", "ms") is True
    assert ev.check_unit_presence("FeNa is 1.0%", "%") is True
    assert ev.check_unit_presence("FeNa is 1.0 percent", "%") is True
    assert ev.check_unit_presence("110 mL/hr", "mL/hr") is True
    assert ev.check_unit_presence("110 ml/h", "mL/hr") is True
    assert ev.check_unit_presence("27.8 kg/m2", "kg/m2") is True
    assert ev.check_unit_presence("Score = 6 points", "points") is True
    assert ev.check_unit_presence("Score = 6 pts", "points") is True


def test_integer_clinical_score_matching():
    """Test integer clinical scoring matching (CHA2DS2-VASc, GCS, Wells)."""
    ev = CalculationEvaluator()

    # CHA2DS2-VASc = 4 points
    res_cha = ev.evaluate(
        response="The calculated CHA2DS2-VASc score is 4 points.",
        reference="CHA2DS2-VASc score = 4",
        unit="points",
    )
    assert res_cha["predicted_value"] == 4.0
    assert res_cha["reference_value"] == 4.0
    assert res_cha["is_accurate"] is True
    assert res_cha["unit_matched"] is True
    assert res_cha["score"] == 1.0

    # Glasgow Coma Scale = 11
    res_gcs = ev.evaluate(
        response="Eye (3) + Verbal (3) + Motor (5) = GCS score of 11.",
        reference="11 points",
        unit="score",
    )
    assert res_gcs["predicted_value"] == 11.0
    assert res_gcs["reference_value"] == 11.0
    assert res_gcs["is_accurate"] is True
    assert res_gcs["unit_matched"] is True
    assert res_gcs["score"] == 1.0


def test_extraction_failure_evaluation():
    """Test evaluation return when extraction fails on either side."""
    ev = CalculationEvaluator()

    res_fail = ev.evaluate(
        response="Unable to determine the clinical value.",
        reference="42.5 mL/min",
        unit="mL/min",
    )
    assert res_fail["predicted_value"] is None
    assert res_fail["reference_value"] == 42.5
    assert res_fail["is_accurate"] is False
    assert res_fail["relative_error"] is None
    assert res_fail["score"] == 0.0


def test_compute_batch_metrics():
    """Test batch metrics computation with accurate, inaccurate, and missing units."""
    ev = CalculationEvaluator()

    results = [
        {
            "predicted_value": 100.0,
            "reference_value": 100.0,
            "is_accurate": True,
            "relative_error": 0.0,
            "unit_matched": True,
            "score": 1.0,
        },
        {
            "predicted_value": 102.0,
            "reference_value": 100.0,
            "is_accurate": True,
            "relative_error": 0.02,
            "unit_matched": True,
            "score": 1.0,
        },
        {
            "predicted_value": 104.0,
            "reference_value": 100.0,
            "is_accurate": True,
            "relative_error": 0.04,
            "unit_matched": False,  # missing unit penalty
            "score": 0.0,
        },
        {
            "predicted_value": 120.0,
            "reference_value": 100.0,
            "is_accurate": False,
            "relative_error": 0.20,
            "unit_matched": True,
            "score": 0.0,
        },
        {
            "predicted_value": None,
            "reference_value": 100.0,
            "is_accurate": False,
            "relative_error": None,
            "unit_matched": False,
            "score": 0.0,
        },
    ]

    metrics = ev.compute_batch_metrics(results)
    assert metrics["total_samples"] == 5
    # 2 out of 5 scored 1.0
    assert metrics["calculation_accuracy"] == 0.4
    # 3 out of 5 had unit_matched=True
    assert metrics["unit_adherence_rate"] == 0.6
    # mean of valid errors: (0.0 + 0.02 + 0.04 + 0.20) / 4 = 0.26 / 4 = 0.065
    assert metrics["mean_relative_error"] == 0.065


def test_compute_batch_metrics_empty():
    """Test compute_batch_metrics on empty list."""
    metrics = CalculationEvaluator.compute_batch_metrics([])
    assert metrics["total_samples"] == 0
    assert metrics["calculation_accuracy"] == 0.0
    assert metrics["mean_relative_error"] == 0.0
    assert metrics["unit_adherence_rate"] == 0.0


def test_medcalc_data_loader():
    """Test loading sample_medcalc and medcalc datasets."""
    samples = load_dataset("sample_medcalc", n_samples=10)
    assert len(samples) == 10
    for s in samples:
        assert "question" in s and len(s["question"]) > 0
        assert "answer" in s and len(s["answer"]) > 0
        assert "metadata" in s and s["metadata"] is not None
        assert "unit" in s["metadata"]
        assert "expected_value" in s["metadata"]

    # Alias 'medcalc'
    alias_samples = load_dataset("medcalc", n_samples=5)
    assert len(alias_samples) == 5

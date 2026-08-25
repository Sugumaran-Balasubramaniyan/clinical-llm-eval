"""Tests for Feature 2: Clinical NLI & Contradiction Evaluator."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_llm_eval import ClinicalNLIEvaluator
from clinical_llm_eval.evaluators import ClinicalNLIEvaluator as EvaluatorFromSubpackage


def test_evaluator_imports():
    """Verify ClinicalNLIEvaluator exports correctly from top-level and subpackage."""
    assert ClinicalNLIEvaluator is EvaluatorFromSubpackage
    evaluator = ClinicalNLIEvaluator()
    assert evaluator is not None


def test_penicillin_allergy_contradiction():
    """Test allergy contradiction: Penicillin allergy vignette vs amoxicillin prescription."""
    evaluator = ClinicalNLIEvaluator()
    prompt = (
        "A 45-year-old female presents with acute bacterial sinusitis. "
        "Her medical record documents a severe penicillin allergy with prior anaphylaxis."
    )
    response = "Prescribe amoxicillin-clavulanate 875/125 mg orally twice daily for 10 days."

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is True
    assert result["consistency_score"] == 0.0
    assert result["clinical_nli_label"] == "CONTRADICTION"
    assert len(result["contradictions"]) >= 1
    assert any("allergy" in c.lower() for c in result["contradictions"])
    assert any("amoxicillin" in c.lower() for c in result["contradictions"])


def test_renal_impairment_nsaid_contradiction():
    """Test renal contradiction: CKD with Cr 3.5 vs high-dose ibuprofen prescription."""
    evaluator = ClinicalNLIEvaluator()
    prompt = (
        "A 68-year-old male with chronic kidney disease stage 4 (serum creatinine 3.5 mg/dL, "
        "eGFR 20 mL/min) presents with acute gout flare and knee pain."
    )
    response = "Recommend high-dose ibuprofen 800 mg PO three times daily with meals for inflammation."

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is True
    assert result["consistency_score"] == 0.0
    assert result["clinical_nli_label"] == "CONTRADICTION"
    assert len(result["contradictions"]) >= 1
    assert any("renal" in c.lower() for c in result["contradictions"])
    assert any("ibuprofen" in c.lower() for c in result["contradictions"])


def test_bleeding_risk_heparin_contradiction():
    """Test bleeding risk contradiction: Active GI bleed vs heparin bolus."""
    evaluator = ClinicalNLIEvaluator()
    prompt = (
        "A 55-year-old patient is admitted to the ICU with active upper GI bleeding, "
        "melena, and a dropping hemoglobin of 6.8 g/dL."
    )
    response = "Administer IV heparin bolus 5000 units followed by continuous IV heparin infusion."

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is True
    assert result["consistency_score"] == 0.0
    assert result["clinical_nli_label"] == "CONTRADICTION"
    assert len(result["contradictions"]) >= 1
    assert any("bleeding" in c.lower() or "anticoagulation" in c.lower() for c in result["contradictions"])
    assert any("heparin" in c.lower() for c in result["contradictions"])


def test_consistent_response_safe_alternative():
    """Test consistent response with appropriate alternative therapy (Penicillin allergy vs azithromycin)."""
    evaluator = ClinicalNLIEvaluator()
    prompt = (
        "A 32-year-old male with a documented severe penicillin allergy presents with "
        "streptococcal pharyngitis."
    )
    response = (
        "Because of documented penicillin allergy, penicillin and amoxicillin are contraindicated. "
        "Prescribe azithromycin 500 mg orally on day 1, followed by 250 mg daily on days 2-5 as a safe alternative."
    )

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is False
    assert result["consistency_score"] == 1.0
    assert result["clinical_nli_label"] == "ENTAILMENT"
    assert len(result["contradictions"]) == 0


def test_hepatic_impairment_contradiction():
    """Test hepatic impairment contradiction: Acute liver failure vs high-dose acetaminophen."""
    evaluator = ClinicalNLIEvaluator()
    prompt = (
        "A 50-year-old patient with decompensated cirrhosis, jaundice, and acute liver failure "
        "complains of headache."
    )
    response = "Prescribe acetaminophen 1000 mg every 4 hours (up to 4000 mg daily) for headache."

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is True
    assert result["consistency_score"] == 0.0
    assert result["clinical_nli_label"] == "CONTRADICTION"
    assert len(result["contradictions"]) >= 1
    assert any("hepatic" in c.lower() or "liver" in c.lower() for c in result["contradictions"])


def test_bronchospasm_beta_blocker_contradiction():
    """Test asthma contradiction: Severe asthma vs non-selective beta-blocker (propranolol)."""
    evaluator = ClinicalNLIEvaluator()
    prompt = (
        "A 28-year-old female with severe asthma and frequent acute bronchospasm exacerbations "
        "seeks treatment for migraine prophylaxis."
    )
    response = "Start propranolol 40 mg twice daily as first-line migraine prophylaxis."

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is True
    assert result["consistency_score"] == 0.0
    assert result["clinical_nli_label"] == "CONTRADICTION"
    assert len(result["contradictions"]) >= 1
    assert any("broncho" in c.lower() or "asthma" in c.lower() for c in result["contradictions"])
    assert any("propranolol" in c.lower() for c in result["contradictions"])


def test_sulfa_allergy_contradiction():
    """Test sulfa allergy contradiction: Sulfa allergy vs Bactrim."""
    evaluator = ClinicalNLIEvaluator()
    prompt = "A 40-year-old female with documented sulfa allergy presents with uncomplicated cystitis."
    response = "Prescribe Bactrim DS (trimethoprim-sulfamethoxazole) 1 tablet twice daily for 3 days."

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is True
    assert result["clinical_nli_label"] == "CONTRADICTION"
    assert any("sulfa" in c.lower() for c in result["contradictions"])


def test_negated_contradiction_not_flagged():
    """Verify that explicit advice to avoid or hold contraindicated drugs is not flagged as contradiction."""
    evaluator = ClinicalNLIEvaluator()
    prompt = "A 70-year-old patient with CKD (creatinine 3.2 mg/dL) has joint pain."
    response = "Avoid NSAIDs such as ibuprofen or naproxen due to renal impairment. Recommend topical capsaicin or physical therapy."

    result = evaluator.evaluate(prompt, response)

    assert result["has_contradiction"] is False
    assert result["consistency_score"] == 1.0
    assert result["clinical_nli_label"] == "ENTAILMENT"


def test_empty_or_trivial_response_neutral():
    """Test neutral classification for empty or non-responsive outputs."""
    evaluator = ClinicalNLIEvaluator()
    result = evaluator.evaluate("Patient has hypertension.", "No comment.")
    assert result["has_contradiction"] is False
    assert result["clinical_nli_label"] == "NEUTRAL"
    assert result["consistency_score"] == 0.5


def test_batch_metrics_computation():
    """Test batch metric computation across mixed consistent and contradictory cases."""
    evaluator = ClinicalNLIEvaluator()

    batch_results = [
        # Contradiction 1
        evaluator.evaluate(
            "Patient with severe penicillin allergy.",
            "Prescribe amoxicillin 500 mg TID.",
        ),
        # Contradiction 2
        evaluator.evaluate(
            "Patient with CKD and Cr 3.5 mg/dL.",
            "Prescribe ibuprofen 800 mg TID.",
        ),
        # Consistent 1
        evaluator.evaluate(
            "Patient with severe penicillin allergy.",
            "Avoid penicillin; prescribe azithromycin 500 mg daily.",
        ),
        # Consistent 2
        evaluator.evaluate(
            "Patient with active GI bleed.",
            "Hold heparin and initiate IV PPI infusion with fluid resuscitation.",
        ),
    ]

    metrics = ClinicalNLIEvaluator.compute_batch_metrics(batch_results)

    assert metrics["total_evaluated"] == 4
    assert metrics["contradiction_rate"] == 0.5  # 2 out of 4
    assert metrics["mean_consistency_score"] == 0.5  # (0.0 + 0.0 + 1.0 + 1.0) / 4
    assert metrics["total_contradictions_detected"] == 2
    assert metrics["entailment_rate"] == 0.5


def test_batch_metrics_empty():
    """Test batch metrics when input is empty list."""
    metrics = ClinicalNLIEvaluator.compute_batch_metrics([])
    assert metrics["total_evaluated"] == 0
    assert metrics["contradiction_rate"] == 0.0
    assert metrics["mean_consistency_score"] == 1.0
    assert metrics["total_contradictions_detected"] == 0

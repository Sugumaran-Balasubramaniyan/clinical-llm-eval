"""Unit tests for MultiTurnClinicalEvaluator and sample EHR vignettes in clinical_llm_eval."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_llm_eval import MultiTurnClinicalEvaluator
from clinical_llm_eval.evaluators import MultiTurnClinicalEvaluator as EvaluatorFromPkg
from clinical_llm_eval.data.loader import load_dataset


def test_package_exports():
    """Test that MultiTurnClinicalEvaluator is properly exported from both packages."""
    assert MultiTurnClinicalEvaluator is EvaluatorFromPkg
    evaluator = MultiTurnClinicalEvaluator()
    assert isinstance(evaluator, MultiTurnClinicalEvaluator)


# ============================================================================
# 1. SOAP Note Section Detection Tests
# ============================================================================


class TestSOAPSectionDetection:
    """Test SOAP structure parsing, complete vs incomplete note detection."""

    def test_soap_detection_complete_standard_headers(self):
        """Test detection of complete SOAP note with standard headers."""
        evaluator = MultiTurnClinicalEvaluator()
        complete_soap = (
            "Subjective: Patient reports 2 days of sharp right lower quadrant abdominal pain and fever.\n\n"
            "Objective: Temp 38.2°C, BP 120/80 mmHg, HR 92 bpm. Abdomen tender at McBurney's point. WBC 14,000.\n\n"
            "Assessment: Acute Appendicitis.\n\n"
            "Plan: Keep NPO, stat surgical consult for appendectomy, IV ceftriaxone and metronidazole, IV hydration."
        )

        res = evaluator.evaluate_soap_structure(complete_soap)
        assert res["has_soap_format"] is True
        assert set(res["present_sections"]) == {"S", "O", "A", "P"}
        assert res["missing_sections"] == []
        assert res["soap_completeness_score"] == 1.0
        assert "section_details" in res

    def test_soap_detection_markdown_and_short_headers(self):
        """Test detection with markdown and short-form SOAP headers."""
        evaluator = MultiTurnClinicalEvaluator()
        short_soap = (
            "## S:\n"
            "45yo male with 2 hours of substernal chest pain radiating to left arm.\n"
            "## O:\n"
            "Vitals: BP 150/90, HR 64. ECG shows 3mm ST elevation in leads II, III, aVF.\n"
            "## A:\n"
            "Acute Inferior STEMI.\n"
            "## P:\n"
            "Immediate cath lab activation, Aspirin 325mg chewed, Ticagrelor 180mg, Heparin drip."
        )

        res = evaluator.evaluate_soap_structure(short_soap)
        assert res["has_soap_format"] is True
        assert res["soap_completeness_score"] == 1.0
        assert res["missing_sections"] == []

    def test_soap_detection_incomplete_missing_plan(self):
        """Test incomplete SOAP note missing Plan section."""
        evaluator = MultiTurnClinicalEvaluator()
        missing_plan = (
            "Subjective: 67yo F with productive cough and fevers.\n"
            "Objective: Temp 39.0C, RLL crackles on auscultation, CXR shows lobar consolidation.\n"
            "Assessment: Community-acquired pneumonia."
        )

        res = evaluator.evaluate_soap_structure(missing_plan)
        assert res["has_soap_format"] is False
        assert "P" in res["missing_sections"]
        assert set(res["present_sections"]) == {"S", "O", "A"}
        assert res["soap_completeness_score"] == 0.75

    def test_soap_detection_incomplete_missing_objective_and_plan(self):
        """Test incomplete SOAP note missing Objective and Plan sections."""
        evaluator = MultiTurnClinicalEvaluator()
        missing_obj_plan = (
            "Subjective: Patient has severe headache for 3 hours.\n"
            "Assessment: Migraine vs tension headache."
        )

        res = evaluator.evaluate_soap_structure(missing_obj_plan)
        assert res["has_soap_format"] is False
        assert set(res["missing_sections"]) == {"O", "P"}
        assert res["soap_completeness_score"] == 0.50

    def test_soap_detection_unstructured_text(self):
        """Test unstructured free-text note without SOAP components."""
        evaluator = MultiTurnClinicalEvaluator()
        unstructured = "The patient should rest and drink plenty of fluids."

        res = evaluator.evaluate_soap_structure(unstructured)
        assert res["has_soap_format"] is False
        assert len(res["missing_sections"]) >= 3
        assert res["soap_completeness_score"] < 0.50

    def test_soap_detection_empty_or_none(self):
        """Test empty string and invalid inputs."""
        evaluator = MultiTurnClinicalEvaluator()
        res_empty = evaluator.evaluate_soap_structure("")
        assert res_empty["has_soap_format"] is False
        assert res_empty["soap_completeness_score"] == 0.0
        assert len(res_empty["missing_sections"]) == 4


# ============================================================================
# 2. Dialogue Coherence and Progression Tests
# ============================================================================


class TestDialogueCoherence:
    """Test multi-turn dialogue progression and coherence scoring."""

    def test_coherent_multi_turn_progression(self):
        """Test realistic 4-turn clinical dialogue progression."""
        evaluator = MultiTurnClinicalEvaluator()
        turns = [
            {"turn": 1, "role": "patient", "content": "Doctor, I have had crushing chest pain and nausea for 2 hours."},
            {"turn": 2, "role": "clinician", "content": "Let us check your vitals and perform a cardiovascular exam."},
            {"turn": 2, "role": "system", "content": "Physical exam: BP 150/90, HR 62, diaphoresis. Lungs clear."},
            {"turn": 3, "role": "clinician", "content": "Order 12-lead ECG and cardiac troponin stat."},
            {"turn": 3, "role": "system", "content": "ECG shows 3mm ST elevation in II, III, aVF. Troponin I is 4.2 ng/mL."},
            {"turn": 4, "role": "user", "content": "Synthesize a full SOAP note and management plan."},
        ]
        final_response = (
            "Subjective: 58yo M with crushing chest pain and nausea.\n"
            "Objective: BP 150/90, HR 62. ECG shows ST elevation in II, III, aVF. Troponin 4.2 ng/mL.\n"
            "Assessment: Acute Inferior STEMI.\n"
            "Plan: Emergent Cath lab activation for primary PCI, aspirin, ticagrelor, heparin."
        )

        coherence = evaluator._evaluate_turn_coherence(turns, final_response)
        assert coherence >= 0.70

    def test_empty_turns_coherence(self):
        """Test coherence evaluation on empty turns."""
        evaluator = MultiTurnClinicalEvaluator()
        assert evaluator._evaluate_turn_coherence([], "Some response") == 0.0
        assert evaluator._evaluate_turn_coherence([{"role": "user", "content": ""}], "Some response") == 0.0


# ============================================================================
# 3. Diagnostic Convergence Tests
# ============================================================================


class TestDiagnosticConvergence:
    """Test diagnostic convergence logic and synonym matching."""

    def test_diagnostic_convergence_exact_match(self):
        """Test exact diagnostic match."""
        evaluator = MultiTurnClinicalEvaluator()
        resp = "Assessment: Acute Appendicitis confirmed on CT scan."
        assert evaluator._check_diagnostic_convergence(resp, "Acute Appendicitis") is True

    def test_diagnostic_convergence_acronym_and_alias(self):
        """Test convergence via acronyms and aliases."""
        evaluator = MultiTurnClinicalEvaluator()

        # STEMI
        resp_stemi = "Assessment: Patient is presenting with Acute Inferior STEMI."
        assert evaluator._check_diagnostic_convergence(resp_stemi, "Inferior ST-Elevation Myocardial Infarction (STEMI)") is True

        # DKA
        resp_dka = "Assessment: Severe diabetic ketoacidosis with high anion gap."
        assert evaluator._check_diagnostic_convergence(resp_dka, "Diabetic Ketoacidosis (DKA)") is True

        # CAP
        resp_cap = "Assessment: Community-acquired pneumonia (CAP) secondary to Strep pneumoniae."
        assert evaluator._check_diagnostic_convergence(resp_cap, "Community-Acquired Pneumonia") is True

        # Stroke
        resp_stroke = "Assessment: Acute ischemic stroke with left M1 occlusion."
        assert evaluator._check_diagnostic_convergence(resp_stroke, "Acute Ischemic Stroke") is True

    def test_diagnostic_divergence_mismatch(self):
        """Test rejection when diagnosis diverges."""
        evaluator = MultiTurnClinicalEvaluator()
        resp = "Assessment: Tension pneumothorax requiring immediate needle decompression."
        assert evaluator._check_diagnostic_convergence(resp, "Inferior ST-Elevation Myocardial Infarction (STEMI)") is False


# ============================================================================
# 4. Full Dialogue Evaluation & Composite Score Tests
# ============================================================================


class TestEvaluateDialogue:
    """Test evaluate_dialogue composite evaluation and dictionary structure."""

    def test_evaluate_dialogue_success(self):
        """Test complete dialogue evaluation matching all criteria."""
        evaluator = MultiTurnClinicalEvaluator()
        turns = [
            {"turn": 1, "role": "patient", "content": "I have terrible right lower belly pain and fever."},
            {"turn": 2, "role": "system", "content": "PE: McBurney tenderness, guarding. WBC 14.8k. CT shows 9mm appendicitis."},
        ]
        final_response = (
            "Subjective: 28yo M with RLQ abdominal pain and anorexia.\n"
            "Objective: Temp 38.1C, McBurney tenderness, WBC 14.8k, CT shows 9mm appendix with fat stranding.\n"
            "Assessment: Acute Appendicitis.\n"
            "Plan: NPO, emergent surgical consult for appendectomy, IV ceftriaxone and metronidazole, IV fluids."
        )
        exp_diag = "Acute Appendicitis"
        exp_plan = "NPO, urgent surgical consultation for appendectomy, IV fluids, IV antibiotics"

        result = evaluator.evaluate_dialogue(
            turns=turns,
            final_response=final_response,
            expected_diagnosis=exp_diag,
            expected_plan=exp_plan,
        )

        assert "turn_coherence_score" in result
        assert "diagnostic_convergence" in result
        assert "soap_evaluation" in result
        assert "overall_multiturn_score" in result

        assert isinstance(result["turn_coherence_score"], float)
        assert isinstance(result["diagnostic_convergence"], bool)
        assert isinstance(result["soap_evaluation"], dict)
        assert isinstance(result["overall_multiturn_score"], float)

        assert result["diagnostic_convergence"] is True
        assert result["soap_evaluation"]["has_soap_format"] is True
        assert result["soap_evaluation"]["soap_completeness_score"] == 1.0
        assert result["overall_multiturn_score"] >= 0.80

    def test_evaluate_wrapper_interface(self):
        """Test evaluate() standard method interface."""
        evaluator = MultiTurnClinicalEvaluator()
        res = evaluator.evaluate(
            response="Subjective: Headache\nObjective: Normal\nAssessment: Migraine\nPlan: Triptan",
            reference="Migraine",
            turns=[{"role": "user", "content": "Headache"}],
        )
        assert res["diagnostic_convergence"] is True
        assert res["soap_evaluation"]["has_soap_format"] is True


# ============================================================================
# 5. Dataset Loader Integration Tests
# ============================================================================


class TestSampleEHRDatasetLoader:
    """Test loading sample_ehr and ehr_vignettes datasets."""

    def test_load_sample_ehr_dataset(self):
        """Test loading sample_ehr via load_dataset."""
        samples = load_dataset("sample_ehr", n_samples=3)
        assert isinstance(samples, list)
        assert len(samples) == 3

        for item in samples:
            assert "question" in item
            assert "answer" in item
            assert "metadata" in item
            assert isinstance(item["metadata"], dict)
            assert "turns" in item["metadata"]
            assert "expected_diagnosis" in item["metadata"]
            assert "expected_plan" in item["metadata"]
            assert isinstance(item["metadata"]["turns"], list)
            assert len(item["metadata"]["turns"]) >= 3

    def test_load_ehr_vignettes_alias(self):
        """Test loading with ehr_vignettes alias."""
        samples = load_dataset("ehr_vignettes", n_samples=2)
        assert len(samples) == 2
        assert "STEMI" in samples[0]["metadata"]["expected_diagnosis"]

    def test_loaded_vignette_evaluation_pipeline(self):
        """Test evaluating a vignette directly from loaded dataset."""
        evaluator = MultiTurnClinicalEvaluator()
        samples = load_dataset("sample_ehr", n_samples=1)
        vignette = samples[0]

        eval_result = evaluator.evaluate_dialogue(
            turns=vignette["metadata"]["turns"],
            final_response=vignette["answer"],
            expected_diagnosis=vignette["metadata"]["expected_diagnosis"],
            expected_plan=vignette["metadata"]["expected_plan"],
        )

        assert eval_result["diagnostic_convergence"] is True
        assert eval_result["soap_evaluation"]["has_soap_format"] is True
        assert eval_result["overall_multiturn_score"] >= 0.85


# ============================================================================
# 6. Batch Metrics Computation Tests
# ============================================================================


class TestComputeBatchMetrics:
    """Test compute_batch_metrics aggregation."""

    def test_batch_metrics_computation(self):
        """Test batch aggregation over multiple dialogue results."""
        evaluator = MultiTurnClinicalEvaluator()
        batch_results = [
            {
                "turn_coherence_score": 0.90,
                "diagnostic_convergence": True,
                "soap_evaluation": {"has_soap_format": True, "soap_completeness_score": 1.0},
                "overall_multiturn_score": 0.95,
            },
            {
                "turn_coherence_score": 0.80,
                "diagnostic_convergence": True,
                "soap_evaluation": {"has_soap_format": True, "soap_completeness_score": 1.0},
                "overall_multiturn_score": 0.90,
            },
            {
                "turn_coherence_score": 0.40,
                "diagnostic_convergence": False,
                "soap_evaluation": {"has_soap_format": False, "soap_completeness_score": 0.50},
                "overall_multiturn_score": 0.35,
            },
        ]

        metrics = evaluator.compute_batch_metrics(batch_results)
        assert metrics["total_dialogues"] == 3
        assert metrics["mean_coherence_score"] == pytest.approx(0.70, abs=1e-3)
        assert metrics["soap_adherence_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert metrics["diagnostic_convergence_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert metrics["mean_multiturn_score"] == pytest.approx((0.95 + 0.90 + 0.35) / 3, abs=1e-3)

    def test_batch_metrics_empty(self):
        """Test batch aggregation on empty results list."""
        evaluator = MultiTurnClinicalEvaluator()
        metrics = evaluator.compute_batch_metrics([])

        assert metrics["total_dialogues"] == 0
        assert metrics["mean_coherence_score"] == 0.0
        assert metrics["soap_adherence_rate"] == 0.0
        assert metrics["mean_multiturn_score"] == 0.0
        assert metrics["diagnostic_convergence_rate"] == 0.0

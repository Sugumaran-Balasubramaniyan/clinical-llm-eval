"""Multi-Turn Clinical Triage & SOAP Note Evaluator.

Evaluates multi-turn dialogue coherence, clinical progression, diagnostic convergence,
and SOAP note structural adherence across clinical encounters.
"""

from __future__ import annotations

import re
from typing import Any


class MultiTurnClinicalEvaluator:
    """Evaluates multi-turn clinical triage dialogues and structured SOAP notes."""

    # Standard SOAP Section identifiers
    SOAP_SECTIONS = ["S", "O", "A", "P"]

    SECTION_HEADERS: dict[str, list[str]] = {
        "S": [
            r"(?i)(?:^|\n)\s*(?:#{1,4}\s*|\*{1,2}|\[)?(?:S\s*[:\-\.]|Subjective(?:\s*Section)?\s*[:\-\.]?|S\s*-\s*Subjective|Chief\s+Complaint\s*[:\-\.]|History\s+of\s+Present\s+Illness\s*[:\-\.]|HPI\s*[:\-\.])",
        ],
        "O": [
            r"(?i)(?:^|\n)\s*(?:#{1,4}\s*|\*{1,2}|\[)?(?:O\s*[:\-\.]|Objective(?:\s*Section)?\s*[:\-\.]?|O\s*-\s*Objective|Physical\s+Exam(?:ination)?\s*[:\-\.]|Vital\s+Signs\s*[:\-\.]|Vitals\s*[:\-\.]|Diagnostics?\s*[:\-\.]|Labs?\s*[:\-\.]|Laboratory\s*[:\-\.])",
        ],
        "A": [
            r"(?i)(?:^|\n)\s*(?:#{1,4}\s*|\*{1,2}|\[)?(?:A\s*[:\-\.]|Assessment(?:\s*Section)?\s*[:\-\.]?|A\s*-\s*Assessment|Clinical\s+Impression\s*[:\-\.]|Impression\s*[:\-\.]|Assessment\s*(?:&|\/|and)\s*Plan\s*[:\-\.]|Primary\s+Diagnosis\s*[:\-\.]|Differential\s+Diagnosis\s*[:\-\.])",
        ],
        "P": [
            r"(?i)(?:^|\n)\s*(?:#{1,4}\s*|\*{1,2}|\[)?(?:P\s*[:\-\.]|Plan(?:\s*Section)?\s*[:\-\.]?|P\s*-\s*Plan|Treatment(?:\s+Plan)?\s*[:\-\.]|Management(?:\s+Plan)?\s*[:\-\.]|Orders\s*[:\-\.]|Assessment\s*(?:&|\/|and)\s*Plan\s*[:\-\.]|Disposition\s*[:\-\.]|Recommendations?\s*[:\-\.])",
        ],
    }

    # Clinical concept keywords by SOAP component
    SOAP_KEYWORDS: dict[str, list[str]] = {
        "S": [
            "chief complaint",
            "history of present illness",
            "hpi",
            "presents with",
            "complains of",
            "reported",
            "symptoms",
            "pain",
            "onset",
            "duration",
            "past medical history",
            "subjective",
        ],
        "O": [
            "vitals",
            "vital signs",
            "bp",
            "blood pressure",
            "hr",
            "heart rate",
            "temp",
            "temperature",
            "spo2",
            "physical exam",
            "auscultation",
            "palpation",
            "labs",
            "laboratory",
            "imaging",
            "ecg",
            "ekg",
            "cxr",
            "ct",
            "wbc",
            "objective",
        ],
        "A": [
            "assessment",
            "impression",
            "diagnosis",
            "differential",
            "etiology",
            "acute",
            "chronic",
            "likely",
            "suspected",
            "clinical impression",
            "primary diagnosis",
        ],
        "P": [
            "plan",
            "treatment",
            "management",
            "medication",
            "prescribe",
            "order",
            "iv fluids",
            "follow-up",
            "consult",
            "admission",
            "admit",
            "return precautions",
            "discharge",
            "monitoring",
        ],
    }

    # Common medical synonyms & acronyms for diagnostic matching
    DIAGNOSIS_ALIASES: dict[str, list[str]] = {
        "stemi": [
            "st-elevation myocardial infarction",
            "st elevation myocardial infarction",
            "stemi",
            "myocardial infarction",
            "acute myocardial infarction",
            "inferior stemi",
            "anterior stemi",
            "lateral stemi",
        ],
        "nstemi": [
            "non-st-elevation myocardial infarction",
            "non st elevation myocardial infarction",
            "nstemi",
            "non-stemi",
            "non stemi",
        ],
        "cap": [
            "community-acquired pneumonia",
            "community acquired pneumonia",
            "cap",
            "pneumonia",
            "bacterial pneumonia",
            "streptococcal pneumonia",
        ],
        "dka": [
            "diabetic ketoacidosis",
            "dka",
            "ketoacidosis",
        ],
        "hhs": [
            "hyperosmolar hyperglycemic state",
            "hyperosmolar hyperglycaemic state",
            "hhs",
            "honk",
        ],
        "appendicitis": [
            "acute appendicitis",
            "appendicitis",
        ],
        "stroke": [
            "acute ischemic stroke",
            "ischemic stroke",
            "stroke",
            "cerebrovascular accident",
            "cva",
            "mca stroke",
            "large vessel occlusion",
        ],
        "pe": [
            "pulmonary embolism",
            "pulmonary thromboembolism",
            "pe",
        ],
        "dvt": [
            "deep vein thrombosis",
            "deep venous thrombosis",
            "dvt",
        ],
        "aki": [
            "acute kidney injury",
            "acute renal failure",
            "aki",
        ],
        "chf": [
            "congestive heart failure",
            "heart failure",
            "acute decompensated heart failure",
            "chf",
            "adhf",
        ],
        "copd": [
            "chronic obstructive pulmonary disease",
            "copd exacerbation",
            "copd",
        ],
        "uti": [
            "urinary tract infection",
            "pyelonephritis",
            "cystitis",
            "uti",
        ],
        "sah": [
            "subarachnoid hemorrhage",
            "subarachnoid haemorrhage",
            "sah",
        ],
    }

    def __init__(
        self,
        coherence_weight: float = 0.30,
        diagnosis_weight: float = 0.35,
        soap_weight: float = 0.35,
    ) -> None:
        """Initialize MultiTurnClinicalEvaluator.

        Args:
            coherence_weight: Relative weight for dialogue coherence in overall score.
            diagnosis_weight: Relative weight for diagnostic convergence in overall score.
            soap_weight: Relative weight for SOAP note structure completeness in overall score.
        """
        self.coherence_weight = coherence_weight
        self.diagnosis_weight = diagnosis_weight
        self.soap_weight = soap_weight

    def evaluate_soap_structure(self, soap_text: str) -> dict[str, Any]:
        """Detect standard SOAP note components and score structural completeness.

        Args:
            soap_text: Generated clinical note text.

        Returns:
            Dict containing:
                - has_soap_format: bool (True if all 4 sections present or structured SOAP headers found)
                - present_sections: list[str] (e.g. ['S', 'O', 'A', 'P'])
                - missing_sections: list[str]
                - soap_completeness_score: float (0.0 to 1.0)
                - section_details: dict of subcomponent detection per section
        """
        if not soap_text or not isinstance(soap_text, str) or not soap_text.strip():
            return {
                "has_soap_format": False,
                "present_sections": [],
                "missing_sections": list(self.SOAP_SECTIONS),
                "soap_completeness_score": 0.0,
                "section_details": {sec: {"header_detected": False, "keyword_count": 0} for sec in self.SOAP_SECTIONS},
            }

        clean_text = soap_text.strip()
        present_sections: list[str] = []
        section_details: dict[str, Any] = {}

        # Check explicit section headers and content keywords for each SOAP component
        for sec in self.SOAP_SECTIONS:
            header_patterns = self.SECTION_HEADERS[sec]
            header_match = any(re.search(pat, clean_text) for pat in header_patterns)

            keywords = self.SOAP_KEYWORDS[sec]
            matched_keywords = [
                kw for kw in keywords
                if re.search(r"\b" + re.escape(kw) + r"\b", clean_text, re.IGNORECASE)
            ]

            # Section is present if explicit header matched OR sufficient keyword density present
            is_present = header_match or (len(matched_keywords) >= 3 and len(clean_text.split()) >= 25)

            if is_present:
                present_sections.append(sec)

            section_details[sec] = {
                "header_detected": header_match,
                "keyword_count": len(matched_keywords),
                "matched_keywords": matched_keywords[:5],
            }

        missing_sections = [sec for sec in self.SOAP_SECTIONS if sec not in present_sections]
        completeness_score = round(len(present_sections) / len(self.SOAP_SECTIONS), 4)
        has_soap_format = len(present_sections) == len(self.SOAP_SECTIONS)

        return {
            "has_soap_format": has_soap_format,
            "present_sections": present_sections,
            "missing_sections": missing_sections,
            "soap_completeness_score": completeness_score,
            "section_details": section_details,
        }

    @classmethod
    def _extract_turn_text(cls, turn: Any) -> str:
        """Extract string content from a turn representation."""
        if isinstance(turn, str):
            return turn
        if isinstance(turn, dict):
            for k in ("content", "text", "message", "response", "prompt", "query", "input", "utterance"):
                if k in turn:
                    val = turn[k]
                    return str(val) if val is not None else ""
            # If none of standard keys matched, extract from values excluding metadata/role keys
            filtered = [
                str(v)
                for k, v in turn.items()
                if k not in ("role", "turn", "speaker", "id", "index", "turn_id") and isinstance(v, (str, int, float))
            ]
            return " ".join(filtered)
        return str(turn) if turn is not None else ""

    def _evaluate_turn_coherence(
        self,
        turns: list[dict[str, str]],
        final_response: str,
    ) -> float:
        """Evaluate multi-turn coherence and clinical progression across conversation turns.

        Args:
            turns: Ordered list of dialogue turn dicts.
            final_response: Final model response / SOAP note.

        Returns:
            Coherence score between 0.0 and 1.0.
        """
        if not turns or not isinstance(turns, list):
            return 0.0

        num_turns = len(turns)
        if num_turns == 0:
            return 0.0

        turn_texts = [self._extract_turn_text(t).strip() for t in turns]
        non_empty_texts = [t for t in turn_texts if len(t) > 0]

        if not non_empty_texts:
            return 0.0

        # 1. Structural progression across turns (Turn completeness & count)
        turn_count_score = min(1.0, len(non_empty_texts) / 4.0)

        # 2. Clinical phase progression detection
        # Phase 1: Presentation / Symptoms (early turns)
        early_text = " ".join(turn_texts[: max(1, num_turns // 2)]).lower()
        has_symptoms = any(
            re.search(r"\b" + re.escape(kw) + r"\b", early_text)
            for kw in ["pain", "fever", "cough", "shortness", "nausea", "vomiting", "symptom", "complaint", "hours", "days", "history"]
        )

        # Phase 2: Exam / Diagnostics / Labs (middle/later turns)
        mid_late_text = " ".join(turn_texts[max(0, num_turns // 2 - 1):]).lower()
        has_diagnostics = any(
            re.search(r"\b" + re.escape(kw) + r"\b", mid_late_text)
            for kw in ["vitals", "bp", "hr", "exam", "ecg", "ekg", "labs", "wbc", "ct", "x-ray", "imaging", "glucose", "chemistry"]
        )

        progression_score = 0.2
        if has_symptoms:
            progression_score += 0.4
        if has_diagnostics:
            progression_score += 0.4
        progression_score = min(1.0, progression_score)

        # 3. Contextual synthesis in final response
        # Check that final response incorporates salient entities / keywords from earlier turns
        final_lower = final_response.lower() if final_response else ""
        turn_words = set(
            re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(non_empty_texts).lower())
        )
        # Filter common generic stop words
        stop_words = {
            "what", "with", "this", "that", "from", "have", "been", "were", "they",
            "will", "your", "patient", "turn", "case", "role", "content", "please",
        }
        salient_words = turn_words - stop_words
        if salient_words and final_lower:
            matched_words = sum(1 for w in salient_words if w in final_lower)
            synthesis_score = min(1.0, matched_words / max(5, len(salient_words) * 0.4))
        else:
            synthesis_score = 0.5 if final_lower else 0.0

        # Composite coherence score
        coherence = 0.30 * turn_count_score + 0.40 * progression_score + 0.30 * synthesis_score
        return round(float(coherence), 4)

    def _check_diagnostic_convergence(
        self,
        final_response: str,
        expected_diagnosis: str,
    ) -> bool:
        """Check whether the final response converges upon the expected diagnosis.

        Args:
            final_response: Generated clinical response / SOAP note.
            expected_diagnosis: Ground truth expected diagnosis string.

        Returns:
            True if diagnostic convergence achieved, False otherwise.
        """
        if not expected_diagnosis or not isinstance(expected_diagnosis, str) or not expected_diagnosis.strip():
            return True

        if not final_response or not isinstance(final_response, str) or not final_response.strip():
            return False

        resp_clean = final_response.lower()
        expected_clean = expected_diagnosis.strip().lower()

        # 1. Direct substring match
        if expected_clean in resp_clean:
            return True

        # 2. Normalized alphanumeric match
        norm_expected = re.sub(r"[^a-z0-9\s]", "", expected_clean)
        norm_resp = re.sub(r"[^a-z0-9\s]", "", resp_clean)
        if norm_expected and norm_expected in norm_resp:
            return True

        # 3. Medical alias / acronym matching
        for key, aliases in self.DIAGNOSIS_ALIASES.items():
            # Check if expected diagnosis relates to alias cluster
            is_expected_in_cluster = (key in expected_clean) or any(a in expected_clean for a in aliases)
            if is_expected_in_cluster:
                for alias in aliases:
                    if re.search(r"\b" + re.escape(alias) + r"\b", resp_clean):
                        return True

        # 4. Token-level overlap for multi-word clinical conditions
        expected_tokens = set(re.findall(r"\b[a-z]{3,}\b", expected_clean))
        generic_diag_words = {"acute", "chronic", "severe", "mild", "moderate", "primary", "secondary", "stage", "syndrome", "disease", "type"}
        key_tokens = expected_tokens - generic_diag_words

        if key_tokens:
            matched_key_tokens = [t for t in key_tokens if re.search(r"\b" + re.escape(t) + r"\b", resp_clean)]
            if len(matched_key_tokens) == len(key_tokens):
                return True

        return False

    def _evaluate_plan_coverage(
        self,
        final_response: str,
        expected_plan: str,
    ) -> float:
        """Evaluate alignment between expected management plan and final response."""
        if not expected_plan or not isinstance(expected_plan, str) or not expected_plan.strip():
            return 1.0

        if not final_response or not isinstance(final_response, str) or not final_response.strip():
            return 0.0

        resp_lower = final_response.lower()
        plan_lower = expected_plan.lower()

        # Extract meaningful clinical plan concepts
        plan_phrases = [p.strip() for p in re.split(r"[,;|\n]+", plan_lower) if len(p.strip()) > 3]
        if not plan_phrases:
            return 1.0

        matched_phrases = 0
        for phrase in plan_phrases:
            # Check full phrase or tokens within phrase
            if phrase in resp_lower:
                matched_phrases += 1
            else:
                tokens = [t for t in re.findall(r"\b[a-z]{3,}\b", phrase) if t not in {"and", "for", "with", "the", "via"}]
                if tokens and all(t in resp_lower for t in tokens):
                    matched_phrases += 1

        return round(matched_phrases / len(plan_phrases), 4)

    def evaluate_dialogue(
        self,
        turns: list[dict[str, str]],
        final_response: str,
        expected_diagnosis: str = "",
        expected_plan: str = "",
    ) -> dict[str, Any]:
        """Evaluate a multi-turn clinical triage dialogue and final SOAP note.

        Args:
            turns: List of dialogue turns preceding final assessment.
            final_response: Model's final clinical response / SOAP note.
            expected_diagnosis: Ground truth expected diagnosis.
            expected_plan: Ground truth expected management plan.

        Returns:
            Dict containing:
                - turn_coherence_score: float
                - diagnostic_convergence: bool
                - soap_evaluation: dict
                - overall_multiturn_score: float
        """
        turn_coherence_score = self._evaluate_turn_coherence(turns, final_response)
        diagnostic_convergence = self._check_diagnostic_convergence(final_response, expected_diagnosis)
        soap_evaluation = self.evaluate_soap_structure(final_response)

        diag_score = 1.0 if diagnostic_convergence else 0.0
        soap_score = float(soap_evaluation.get("soap_completeness_score", 0.0))

        if expected_plan and expected_plan.strip():
            plan_score = self._evaluate_plan_coverage(final_response, expected_plan)
            overall_score = (
                0.25 * turn_coherence_score
                + 0.35 * diag_score
                + 0.25 * soap_score
                + 0.15 * plan_score
            )
        else:
            overall_score = (
                self.coherence_weight * turn_coherence_score
                + self.diagnosis_weight * diag_score
                + self.soap_weight * soap_score
            )

        return {
            "turn_coherence_score": turn_coherence_score,
            "diagnostic_convergence": diagnostic_convergence,
            "soap_evaluation": soap_evaluation,
            "overall_multiturn_score": round(float(overall_score), 4),
        }

    def evaluate(
        self,
        response: str,
        reference: str = "",
        turns: list[dict[str, str]] | None = None,
        expected_diagnosis: str = "",
        expected_plan: str = "",
        question: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Standard evaluator interface wrapper."""
        dialogue_turns = turns or []
        diag = expected_diagnosis or reference
        return self.evaluate_dialogue(
            turns=dialogue_turns,
            final_response=response,
            expected_diagnosis=diag,
            expected_plan=expected_plan,
        )

    @classmethod
    def compute_batch_metrics(cls, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregated multi-turn and SOAP metrics across a batch of evaluations.

        Args:
            results: List of evaluation result dicts from `evaluate_dialogue()`.

        Returns:
            Dict containing:
                - mean_coherence_score: float
                - soap_adherence_rate: float
                - mean_multiturn_score: float
                - diagnostic_convergence_rate: float
                - total_dialogues: int
        """
        total = len(results)
        if total == 0:
            return {
                "mean_coherence_score": 0.0,
                "soap_adherence_rate": 0.0,
                "mean_multiturn_score": 0.0,
                "diagnostic_convergence_rate": 0.0,
                "total_dialogues": 0,
            }

        coherence_scores = [float(r.get("turn_coherence_score", 0.0)) for r in results]
        mean_coherence = sum(coherence_scores) / total

        multiturn_scores = [float(r.get("overall_multiturn_score", 0.0)) for r in results]
        mean_multiturn = sum(multiturn_scores) / total

        soap_adherent_count = sum(
            1 for r in results
            if bool(r.get("soap_evaluation", {}).get("has_soap_format", False))
            or float(r.get("soap_evaluation", {}).get("soap_completeness_score", 0.0)) >= 1.0
        )
        soap_adherence_rate = soap_adherent_count / total

        conv_count = sum(1 for r in results if bool(r.get("diagnostic_convergence", False)))
        convergence_rate = conv_count / total

        return {
            "mean_coherence_score": round(mean_coherence, 4),
            "soap_adherence_rate": round(soap_adherence_rate, 4),
            "mean_multiturn_score": round(mean_multiturn, 4),
            "diagnostic_convergence_rate": round(convergence_rate, 4),
            "total_dialogues": total,
        }

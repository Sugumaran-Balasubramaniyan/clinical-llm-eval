"""Hallucination detector for clinical LLM responses."""

from __future__ import annotations

import re
from typing import Set


class HallucinationDetector:
    """Detects potential hallucinations by checking entity/fact overlap and grounding."""

    MEDICAL_PATTERNS = [
        r'\b(?:mg|mcg|\xb5g|ml|L|mmol|mmHg|bpm)\b',
        r'\b[A-Z][a-z]+(?:ine|ol|an|ide|ate|ase|mab|nib|zole|cillin|mycin|pril|sartan)\b',
        r'\b(?:type [0-9]|stage [IV]+|grade [0-9]|class [IV]+)\b',
    ]

    COMMON_CLINICAL_TERMS = {
        'diagnosis', 'treatment', 'prognosis', 'medication', 'surgery', 'therapy',
        'infection', 'inflammation', 'chronic', 'acute', 'benign', 'malignant',
        'biopsy', 'etiology', 'pathology', 'pathophysiology', 'management',
        'syndrome', 'disease', 'disorder', 'symptom', 'sign', 'indication',
        'contraindication', 'clinical', 'patient', 'presentation', 'examination',
        'assessment', 'differential', 'laboratory', 'findings', 'investigation',
    }

    def detect(self, response: str, reference: str, question: str | None = None) -> bool:
        """Detect potential hallucination in a clinical response.

        Returns True if ungrounded medical entities exceed the threshold.
        """
        detail = self.detect_detailed(response, reference, question)
        return detail["is_hallucination"]

    def detect_detailed(
        self, response: str, reference: str, question: str | None = None
    ) -> dict:
        """Provide fine-grained hallucination breakdown."""
        ref_text = reference
        if question:
            ref_text = f"{question} {reference}"

        ref_tokens = self._extract_key_terms(ref_text)
        resp_tokens = self._extract_key_terms(response)

        if not resp_tokens:
            return {
                "is_hallucination": False,
                "hallucination_score": 0.0,
                "unsupported_terms": [],
                "grounded_terms_count": 0,
            }

        # Unsupported terms: medical entities introduced in response not found in prompt/reference
        unsupported = resp_tokens - ref_tokens
        hallucination_score = len(unsupported) / max(len(resp_tokens), 1)

        # Threshold calibrated: > 0.70 ungrounded medical entities flags hallucination
        is_hallucination = hallucination_score > 0.70 and len(unsupported) >= 2

        return {
            "is_hallucination": is_hallucination,
            "hallucination_score": round(hallucination_score, 3),
            "unsupported_terms": sorted(list(unsupported)),
            "grounded_terms_count": len(resp_tokens & ref_tokens),
        }

    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key medical entities and terms from text."""
        tokens: Set[str] = set()
        tokens.update(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        for pattern in self.MEDICAL_PATTERNS:
            tokens.update(re.findall(pattern, text, re.IGNORECASE))
        words = set(text.lower().split())
        tokens.update(words & self.COMMON_CLINICAL_TERMS)
        return tokens

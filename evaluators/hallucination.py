"""Hallucination detector for clinical LLM responses."""

import re
from typing import Optional, Set


class HallucinationDetector:
    """Detects potential hallucinations by checking entity/fact overlap.

    Uses Named Entity Recognition and keyword matching to identify
    claims in the response that are absent from the reference answer.
    """

    # Medical entity patterns to extract and compare
    MEDICAL_PATTERNS = [
        r'\b(?:mg|mcg|\xb5g|ml|L|mmol|mmHg|bpm)\b',
        r'\b[A-Z][a-z]+(?:ine|ol|an|ide|ate|ase)\b',
        r'\b(?:type [0-9]|stage [IV]+|grade [0-9])\b',
    ]

    def detect(self, response: str, reference: str) -> bool:
        """Detect potential hallucination in a clinical response.

        Args:
            response: The LLM-generated response.
            reference: The ground truth answer.

        Returns:
            True if hallucination is likely detected, False otherwise.
        """
        ref_tokens = self._extract_key_terms(reference)
        resp_tokens = self._extract_key_terms(response)

        if not ref_tokens:
            return False

        resp_only = resp_tokens - ref_tokens
        new_term_ratio = len(resp_only) / max(len(resp_tokens), 1)
        return new_term_ratio > 0.6

    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key medical terms from text."""
        tokens: Set[str] = set()

        tokens.update(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))

        for pattern in self.MEDICAL_PATTERNS:
            tokens.update(re.findall(pattern, text))

        med_keywords = {
            'diagnosis', 'treatment', 'prognosis', 'medication',
            'surgery', 'therapy', 'infection', 'inflammation',
            'chronic', 'acute', 'benign', 'malignant', 'biopsy',
        }
        words = set(text.lower().split())
        tokens.update(words & med_keywords)

        return tokens

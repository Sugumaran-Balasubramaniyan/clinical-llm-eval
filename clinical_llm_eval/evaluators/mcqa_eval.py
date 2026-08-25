"""MCQA accuracy and option extraction evaluator for clinical QA benchmarks."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


class MCQAEvaluator:
    """Evaluator for Multiple Choice Question Answering (MCQA) accuracy and option extraction."""

    def __init__(self, usmle_pass_threshold: float = 0.60) -> None:
        self.usmle_pass_threshold = usmle_pass_threshold

    @staticmethod
    def _normalize_options(options: list[str] | dict[str, str] | None) -> dict[str, str]:
        """Normalize options list or dict into a mapping of uppercase letter -> option text."""
        if not options:
            return {}
        norm: dict[str, str] = {}
        if isinstance(options, dict):
            for k, v in options.items():
                if k is not None and v is not None:
                    norm[str(k).strip().upper()] = str(v).strip()
        elif isinstance(options, (list, tuple)):
            for idx, opt in enumerate(options):
                if opt is None:
                    continue
                opt_str = str(opt).strip()
                match = re.match(r'^\s*[\(\[]?([A-Za-z])[\)\]]?[\.\:\-\s]+(.*)$', opt_str)
                if match:
                    norm[match.group(1).upper()] = match.group(2).strip()
                else:
                    letter = chr(ord("A") + idx)
                    norm[letter] = opt_str
        return norm

    @classmethod
    def _match_against_options(cls, text: str, norm_options: dict[str, str]) -> str | None:
        """Match response text against candidate option texts via substring or fuzzy matching."""
        if not norm_options or not text:
            return None

        cleaned_text = re.sub(r'[\*\_#`\.\,\;\:\!\?]', ' ', text).strip().lower()
        text_lower = text.lower()

        # 1. Exact full text match
        for letter, opt_text in norm_options.items():
            cleaned_opt = re.sub(r'[\*\_#`\.\,\;\:\!\?]', ' ', opt_text).strip().lower()
            if cleaned_text == cleaned_opt:
                return letter

        # 2. Substring matching
        substring_matches = []
        for letter, opt_text in norm_options.items():
            cleaned_opt = opt_text.strip().lower()
            if len(cleaned_opt) >= 2 and cleaned_opt in text_lower:
                substring_matches.append((letter, cleaned_opt))

        if len(substring_matches) == 1:
            return substring_matches[0][0]

        # If multiple option substrings match, check if one is in conclusion/last lines
        if len(substring_matches) > 1:
            lines = [line.lower() for line in text.splitlines() if line.strip()]
            if lines:
                last_segment = " ".join(lines[-2:])
                last_matches = [
                    (letter, opt) for letter, opt in substring_matches if opt in last_segment
                ]
                if len(last_matches) == 1:
                    return last_matches[0][0]

        # 3. Fuzzy similarity fallback if response is short
        if len(cleaned_text.split()) <= 15:
            best_letter = None
            best_score = 0.0
            second_score = 0.0
            for letter, opt_text in norm_options.items():
                cleaned_opt = re.sub(r'[\*\_#`\.\,\;\:\!\?]', ' ', opt_text).strip().lower()
                sim = SequenceMatcher(None, cleaned_text, cleaned_opt).ratio()
                if sim > best_score:
                    second_score = best_score
                    best_score = sim
                    best_letter = letter
                elif sim > second_score:
                    second_score = sim

            if best_letter is not None and best_score >= 0.70 and (best_score - second_score >= 0.15):
                return best_letter

        return None

    @classmethod
    def extract_choice(
        cls,
        response: str,
        options: list[str] | dict[str, str] | None = None,
    ) -> str | None:
        """Extract the selected choice letter (e.g. 'A', 'B', 'C', 'D', 'E') or match against options.

        Args:
            response: Model-generated response string.
            options: Optional dict or list of candidate options.

        Returns:
            Extracted choice letter in uppercase (e.g. 'A') or None if extraction fails.
        """
        if not response or not isinstance(response, str):
            return None

        text = response.strip()
        if not text:
            return None

        norm_options = cls._normalize_options(options)

        # 1. Direct single-letter / standalone letter match after markdown stripping
        cleaned_simple = re.sub(r'[\*\_#`]', '', text).strip()
        single_match = re.fullmatch(
            r'^(?:(?:Answer|Final Answer|Correct Answer|Option|Choice)\s*(?:is|:|\-|\s)*)?[\(\[]?([A-Ea-e])[\)\]]?\.?$',
            cleaned_simple,
            re.IGNORECASE,
        )
        if single_match:
            return single_match.group(1).upper()

        # 2. Check starting letter pattern at the very beginning of the response
        # 2a. Preceded by an explicit label (e.g. "Answer: A", "Option B", "Answer is C")
        label_start_pattern = re.compile(
            r'^(?:[\*\_#`\s\-\>]*)(?:Answer|Final Answer|Correct Answer|Option|Choice)\s*(?:is|:|\-|\=|\s)*[\*\_`]*[\(\[]?([A-Ea-e])[\)\]]?[\*\_`]*(?:[\.\)\:\-\–\—\s]|$)',
            re.IGNORECASE,
        )
        m_label = label_start_pattern.match(text)
        if m_label:
            return m_label.group(1).upper()

        # 2b. Direct letter with clear delimiter / enclosing markdown (e.g. "A.", "A)", "(A)", "[A]", "**A.**", "**A. Acute STEMI**")
        direct_start_pattern = re.compile(
            r'^(?:[\*\_#`\s\-\>]*)(?:[\(\[]([A-Ea-e])[\)\]]|(?:\*\*|\*|__|_)([A-Ea-e])(?:\*\*|\*|__|_)|([A-Ea-e])(?:\.|\)|\:|\-|\–|\—))(?:\s+|$)'
        )
        m_direct = direct_start_pattern.match(text)
        if m_direct:
            for g in m_direct.groups():
                if g:
                    return g.upper()

        # 3. Explicit mentions anywhere in text (ordered by assertion strength)
        explicit_patterns = [
            # "Therefore, the correct answer is (B)" / "Final Answer: B" / "The best answer is C."
            re.compile(
                r'(?i)(?:final\s+answer|correct\s+answer|the\s+answer|best\s+answer)\s*(?:is|:)?\s*[\*\_`\s]*\(?([A-Ea-e])\)?(?:\.|\b|\:|\s|$)'
            ),
            # "Choice A is correct" / "Option B is the correct choice" / "Option C is the most likely diagnosis"
            re.compile(
                r'(?i)\b(?:choice|option)\s*[\*\_`\s]*\(?([A-Ea-e])\)?\s*(?:is\s+correct|is\s+the\s+correct|is\s+the\s+best|is\s+the\s+most|is\s+right|is\s+most\s+likely)\b'
            ),
            # "correct answer is: A" / "the answer is B" / "choice: C" / "option D" / "answer is A"
            re.compile(
                r'(?i)(?:correct\s+answer\s+is|the\s+answer\s+is|choice|option|answer\s+is)\s*[:\*\s\-]*\(?([A-Ea-e])\)\b'
            ),
            re.compile(
                r'(?i)(?:correct\s+answer\s+is|the\s+answer\s+is|answer\s+is)\s*[:\*\s\-]*[\*\_`]*\b([A-Ea-e])\b'
            ),
            # "A is the correct answer" / "B is the most likely diagnosis"
            re.compile(
                r'(?i)\b([A-Ea-e])\s+is\s+(?:the\s+)?(?:correct|best|most\s+likely)\s+(?:answer|choice|option|diagnosis|treatment|step|management)\b'
            ),
            # "Answer: A" at start of a line
            re.compile(
                r'(?i)(?:^|\n)\s*(?:Answer|Final\s+Answer|Conclusion)\s*(?:is|:|\-)?\s*[\*\_`]*\(?([A-Ea-e])\)?(?:\.|\b|\:|\s|$)'
            ),
        ]

        for pattern in explicit_patterns:
            matches = list(pattern.finditer(text))
            if matches:
                return matches[-1].group(1).upper()

        # 4. Check trailing line (e.g. conclusion at the end of multi-line explanation)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            last_cleaned = re.sub(r'[\*\_#`]', '', last_line).strip()
            # 4a. Standalone on last line
            m_last = re.fullmatch(
                r'^(?:(?:Answer|Final Answer|Correct Answer|Option|Choice)\s*(?:is|:|\-|\s)*)?[\(\[]?([A-Ea-e])[\)\]]?\.?$',
                last_cleaned,
                re.IGNORECASE,
            )
            if m_last:
                return m_last.group(1).upper()

            # 4b. Preceded by label on last line
            m_last_label = label_start_pattern.match(last_line)
            if m_last_label:
                return m_last_label.group(1).upper()

            # 4c. Direct letter with clear delimiter on last line
            m_last_direct = direct_start_pattern.match(last_line)
            if m_last_direct:
                for g in m_last_direct.groups():
                    if g:
                        return g.upper()

        # 5. Matching against option strings if options are provided
        if norm_options:
            matched_choice = cls._match_against_options(text, norm_options)
            if matched_choice:
                return matched_choice

        return None

    def evaluate(
        self,
        response: str,
        reference: str,
        question: str | None = None,
        options: list[str] | dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single MCQA response against reference.

        Args:
            response: Model-generated response string.
            reference: Ground truth reference answer (letter or option text).
            question: Optional question prompt for context.
            options: Optional list or dict of candidate choices.

        Returns:
            Dict containing:
                - predicted_choice: Extracted predicted letter (or None)
                - reference_choice: Extracted reference letter (or None)
                - is_correct: bool indicating whether prediction matches reference
                - score: 1.0 if correct else 0.0
                - usmle_pass_threshold: 0.60 (or configured threshold)
        """
        predicted_choice = self.extract_choice(response, options=options)
        reference_choice = self.extract_choice(reference, options=options)

        norm_options = self._normalize_options(options)
        if reference_choice is None and norm_options:
            ref_match = self._match_against_options(reference, norm_options)
            if ref_match:
                reference_choice = ref_match

        is_correct = False
        if predicted_choice is not None and reference_choice is not None:
            is_correct = predicted_choice == reference_choice
        elif predicted_choice is None and reference_choice is None:
            if response and reference and response.strip().lower() == reference.strip().lower():
                is_correct = True
        elif predicted_choice is not None and reference_choice is None:
            if norm_options and predicted_choice in norm_options:
                opt_text = norm_options[predicted_choice].lower()
                if opt_text and opt_text in reference.lower():
                    is_correct = True
        elif predicted_choice is None and reference_choice is not None:
            if norm_options and reference_choice in norm_options:
                opt_text = norm_options[reference_choice].lower()
                if opt_text and opt_text in response.lower():
                    is_correct = True

        score = 1.0 if is_correct else 0.0

        return {
            "predicted_choice": predicted_choice,
            "reference_choice": reference_choice,
            "is_correct": is_correct,
            "score": score,
            "usmle_pass_threshold": self.usmle_pass_threshold,
        }

    @staticmethod
    def compute_batch_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregated metrics across a batch of MCQA evaluation results.

        Args:
            results: List of evaluation result dicts from `evaluate()`.

        Returns:
            Dict containing:
                - accuracy: float (mean is_correct)
                - pass_usmle: bool (accuracy >= 0.60)
                - total_samples: int
                - valid_extractions_count: int
                - extraction_rate: float
        """
        total_samples = len(results)
        if total_samples == 0:
            return {
                "accuracy": 0.0,
                "pass_usmle": False,
                "total_samples": 0,
                "valid_extractions_count": 0,
                "extraction_rate": 0.0,
            }

        valid_extractions_count = sum(
            1 for r in results if r.get("predicted_choice") is not None
        )
        extraction_rate = valid_extractions_count / total_samples
        correct_count = sum(1 for r in results if bool(r.get("is_correct")))
        accuracy = correct_count / total_samples
        pass_usmle = accuracy >= 0.60

        return {
            "accuracy": round(accuracy, 4),
            "pass_usmle": bool(pass_usmle),
            "total_samples": total_samples,
            "valid_extractions_count": valid_extractions_count,
            "extraction_rate": round(extraction_rate, 4),
        }

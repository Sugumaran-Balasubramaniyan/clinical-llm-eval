"""Clinical calculation and numerical dosage evaluator for clinical LLM assessments."""

from __future__ import annotations

import re
from typing import Any


class CalculationEvaluator:
    """Evaluator for clinical calculations, numerical dosage accuracy, and unit adherence."""

    UNIT_PATTERNS: dict[str, str] = {
        "ml/min": r"\b(?:ml|mL)\s*\/\s*(?:min|minute)(?:\s*\/\s*1\.73\s*m\^?2)?\b",
        "ml/min/1.73m2": r"\b(?:ml|mL)\s*\/\s*(?:min|minute)(?:\s*\/\s*1\.73\s*m\^?2)?\b",
        "meq/l": r"\b(?:meq|mEq)\s*\/\s*(?:l|L)\b",
        "mg/dl": r"\b(?:mg)\s*\/\s*(?:dl|dL)\b",
        "g/dl": r"\b(?:g|gm)\s*\/\s*(?:dl|dL)\b",
        "ml/hr": r"\b(?:ml|mL)\s*\/\s*(?:hr|h|hour)\b",
        "ms": r"\b(?:ms|msec|milliseconds?)\b",
        "mg/kg": r"\b(?:mg)\s*\/\s*(?:kg)(?:\s*\/\s*(?:day|d|dose))?\b",
        "mg": r"\b(?:mg|milligrams?)\b",
        "g": r"\b(?:g|grams?)\b",
        "mcg": r"\b(?:mcg|µg|micrograms?)\b",
        "kg/m2": r"\b(?:kg)\s*\/\s*(?:m\^?2|m2|meter\s*squared)\b",
        "%": r"(?:%|\bpercent(?:age)?\b)",
        "percent": r"(?:%|\bpercent(?:age)?\b)",
        "points": r"\b(?:points?|pts?|score)\b",
        "point": r"\b(?:points?|pts?|score)\b",
        "score": r"\b(?:points?|pts?|score)\b",
    }

    def __init__(self, default_tolerance: float = 0.05) -> None:
        self.default_tolerance = default_tolerance

    @classmethod
    def check_unit_presence(cls, text: str, unit: str) -> bool:
        """Verify whether the required clinical unit is present in the response text."""
        if not text or not unit:
            return False

        norm_unit = unit.strip().lower()
        if not norm_unit:
            return True

        pattern = cls.UNIT_PATTERNS.get(norm_unit)
        if pattern:
            return bool(re.search(pattern, text, re.IGNORECASE))

        # Direct escape fallback for arbitrary units
        escaped = re.escape(unit.strip())
        return bool(re.search(escaped, text, re.IGNORECASE))

    @classmethod
    def extract_number(cls, text: str, unit: str | None = None) -> float | None:
        """Extract primary numerical value from calculation response or reference text.

        Args:
            text: Response or reference text.
            unit: Optional expected unit to guide targeted number extraction.

        Returns:
            Extracted float value or None if no valid number found.
        """
        if not text or not isinstance(text, str):
            return None

        clean_text = re.sub(r"[\*\_`#]", "", text).strip()
        if not clean_text:
            return None

        def _parse_num(s: str) -> float | None:
            try:
                return float(s.replace(",", ""))
            except (ValueError, TypeError):
                return None

        # 1. If unit is specified, search for number adjacent to the unit first
        if unit:
            norm_unit = unit.strip().lower()
            pattern = cls.UNIT_PATTERNS.get(norm_unit)
            if pattern:
                m_unit = list(
                    re.finditer(
                        r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:" + pattern + r")",
                        clean_text,
                        re.IGNORECASE,
                    )
                )
                if m_unit:
                    val = _parse_num(m_unit[-1].group(1))
                    if val is not None:
                        return val
            else:
                m_unit = list(
                    re.finditer(
                        r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:" + re.escape(unit.strip()) + r")",
                        clean_text,
                        re.IGNORECASE,
                    )
                )
                if m_unit:
                    val = _parse_num(m_unit[-1].group(1))
                    if val is not None:
                        return val

        # 2. Check explicit answer / result labels
        explicit_patterns = [
            re.compile(
                r"(?i)(?:final\s+answer|correct\s+answer|calculated\s+result|calculated\s+value|calculated\s+score|total\s+score|total\s+gcs|gcs\s+score|score|result|answer)\s*(?:is|=|:)\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
            ),
            re.compile(
                r"(?i)\b(?:egfr|qtc|anion\s+gap|gcs|cha2ds2-vasc|fena|bmi|dosage|maintenance\s+rate|rate|dose)\s*(?:is|=|:)\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
            ),
            re.compile(
                r"(?i)\b(?:is|equals|equal\s+to|calculated\s+(?:as|to\s+be))\s*(?:approximately|about)?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
            ),
        ]

        for p in explicit_patterns:
            matches = list(p.finditer(clean_text))
            if matches:
                val = _parse_num(matches[-1].group(1))
                if val is not None:
                    return val

        # 3. Check trailing lines (conclusion sentence / final calculation line)
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        if lines:
            for line in reversed(lines[-2:]):
                num_matches = list(re.finditer(r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", line))
                if num_matches:
                    val = _parse_num(num_matches[-1].group(1))
                    if val is not None:
                        return val

        # 4. Fallback: all numbers in entire text
        all_numbers = list(re.finditer(r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", clean_text))
        if all_numbers:
            val = _parse_num(all_numbers[-1].group(1))
            if val is not None:
                return val

        return None

    def evaluate(
        self,
        response: str,
        reference: str,
        tolerance: float = 0.05,
        unit: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a clinical calculation response against ground truth reference.

        Args:
            response: Model-generated calculation response text.
            reference: Ground truth reference answer text or value.
            tolerance: Relative numerical error tolerance (default 0.05 = 5%).
            unit: Optional clinical unit (e.g. 'mL/min', 'mEq/L', 'points', '%').

        Returns:
            Dict containing:
                - predicted_value: float | None
                - reference_value: float | None
                - is_accurate: bool
                - relative_error: float | None
                - unit_matched: bool
                - score: 1.0 if (is_accurate and unit_matched) else 0.0
        """
        pred_val = self.extract_number(response, unit=unit)
        ref_val = self.extract_number(reference, unit=unit)

        is_accurate = False
        relative_error: float | None = None

        if pred_val is not None and ref_val is not None:
            if abs(ref_val) > 1e-9:
                relative_error = abs(pred_val - ref_val) / abs(ref_val)
                is_accurate = relative_error <= tolerance
            else:
                relative_error = abs(pred_val - ref_val)
                is_accurate = relative_error <= tolerance

        if unit is not None and unit.strip() != "":
            unit_matched = self.check_unit_presence(response, unit)
        else:
            unit_matched = True

        score = 1.0 if (is_accurate and unit_matched) else 0.0

        return {
            "predicted_value": pred_val,
            "reference_value": ref_val,
            "is_accurate": is_accurate,
            "relative_error": round(relative_error, 6) if relative_error is not None else None,
            "unit_matched": unit_matched,
            "score": score,
        }

    @classmethod
    def compute_batch_metrics(cls, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregated metrics across a batch of calculation evaluation results.

        Args:
            results: List of evaluation result dicts from `evaluate()`.

        Returns:
            Dict containing:
                - calculation_accuracy: float
                - mean_relative_error: float
                - unit_adherence_rate: float
                - total_samples: int
        """
        total_samples = len(results)
        if total_samples == 0:
            return {
                "calculation_accuracy": 0.0,
                "mean_relative_error": 0.0,
                "unit_adherence_rate": 0.0,
                "total_samples": 0,
            }

        accurate_and_unit_count = sum(1 for r in results if r.get("score") == 1.0)
        calculation_accuracy = accurate_and_unit_count / total_samples

        unit_matches = sum(1 for r in results if bool(r.get("unit_matched", False)))
        unit_adherence_rate = unit_matches / total_samples

        valid_errors = [
            r["relative_error"]
            for r in results
            if r.get("relative_error") is not None
        ]
        mean_relative_error = sum(valid_errors) / len(valid_errors) if valid_errors else 0.0

        return {
            "calculation_accuracy": round(calculation_accuracy, 4),
            "mean_relative_error": round(mean_relative_error, 4),
            "unit_adherence_rate": round(unit_adherence_rate, 4),
            "total_samples": total_samples,
        }

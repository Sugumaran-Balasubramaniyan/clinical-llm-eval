"""Selective classification and confidence calibration evaluator for clinical LLM assessments."""

from __future__ import annotations

import math
import re
from typing import Any


class CalibrationEvaluator:
    """Evaluator for verbalized confidence calibration, Brier score, ECE, and selective classification."""

    DEFAULT_ASSERTIVE_CONFIDENCE: float = 0.85
    DEFAULT_AMBIGUOUS_CONFIDENCE: float = 0.50
    OVERCONFIDENT_THRESHOLD: float = 0.80

    # Patterns for extracting verbalized confidence/probability scores
    CONFIDENCE_PATTERNS: list[re.Pattern] = [
        # Explicit key-value pairs: Confidence: 90%, Probability = 0.85, Likelihood: 80%
        re.compile(
            r"\b(?:confidence|probability|certainty|likelihood)\b[\*\s\:\=\-\–\—]*(?:is|score|level)?[\*\s\:\=\-\–\—]*(\d+(?:\.\d+)?)\s*(%?)",
            re.IGNORECASE,
        ),
        # In-text percentage phrases: 90% confident, 85% certain, 80% probability
        re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(%)\s*(?:confident|certain|probability|likelihood|certainty)\b",
            re.IGNORECASE,
        ),
        # Phrases like "I am 90% confident", "I'm 85% certain"
        re.compile(
            r"\b(?:i\s+am|i\'m)\s+(\d+(?:\.\d+)?)\s*(%?)\s*(?:confident|certain)\b",
            re.IGNORECASE,
        ),
        # Phrases like "with 90% confidence", "at 85% certainty"
        re.compile(
            r"\b(?:with|at)\s+(\d+(?:\.\d+)?)\s*(%)\s*(?:confidence|certainty|probability)\b",
            re.IGNORECASE,
        ),
    ]

    # Clinical ambiguity and uncertainty indicators
    AMBIGUITY_PATTERNS: list[re.Pattern] = [
        re.compile(
            r"\b(?:unclear|uncertain|inconclusive|equivocal|ambiguous|undetermined|unknown)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:cannot|can\'t|unable\s+to|hard\s+to|difficult\s+to)\s+(?:determine|confirm|diagnose|tell|conclude|rule\s+out|distinguish|decide)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:insufficient|inadequate|lacking|not\s+enough)\s+(?:information|data|evidence|clinical\s+detail|findings)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:may\s+or\s+may\s+not|could\s+be\s+either|differential\s+is\s+(?:too\s+)?broad)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:possible\s+but\s+(?:unconfirmed|uncertain|unclear|doubtful))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:highly\s+uncertain|very\s+uncertain|low\s+confidence)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        overconfident_threshold: float = 0.80,
        default_assertive_confidence: float = 0.85,
        default_ambiguous_confidence: float = 0.50,
    ) -> None:
        self.overconfident_threshold = overconfident_threshold
        self.default_assertive_confidence = default_assertive_confidence
        self.default_ambiguous_confidence = default_ambiguous_confidence

    @classmethod
    def extract_confidence(cls, response: str) -> float:
        """Extract verbalized confidence score (0.0 to 1.0) from clinical response.

        Checks for explicit confidence / probability statements in the text.
        If no explicit numeric score is found, defaults to 0.85 for assertive clinical answers
        or 0.50 if highly ambiguous / uncertain.

        Args:
            response: LLM clinical response text.

        Returns:
            Extracted confidence score between 0.0 and 1.0.
        """
        if not response or not isinstance(response, str):
            return cls.DEFAULT_AMBIGUOUS_CONFIDENCE

        text = response.strip()
        if not text:
            return cls.DEFAULT_AMBIGUOUS_CONFIDENCE

        # Clean outer markdown symbols that might interfere with boundary matching
        clean_text = re.sub(r"[\*\_`]", "", text)

        for pattern in cls.CONFIDENCE_PATTERNS:
            matches = list(pattern.finditer(clean_text))
            if matches:
                # Use the last matched confidence expression in the response
                match = matches[-1]
                val_str = match.group(1)
                has_percent = (
                    len(match.groups()) > 1 and match.group(2) == "%"
                ) or "%" in match.group(0)

                try:
                    val = float(val_str)
                    if has_percent or val > 1.0:
                        val = val / 100.0
                    val = min(max(val, 0.0), 1.0)
                    return round(val, 4)
                except (ValueError, TypeError):
                    continue

        # Check for ambiguity indicators
        for amb_pattern in cls.AMBIGUITY_PATTERNS:
            if amb_pattern.search(clean_text):
                return cls.DEFAULT_AMBIGUOUS_CONFIDENCE

        # Default for assertive clinical statements
        return cls.DEFAULT_ASSERTIVE_CONFIDENCE

    @classmethod
    def compute_brier_score(
        cls, predictions: list[float], targets: list[int | bool | float]
    ) -> float:
        """Compute Brier score (mean squared error between predicted probabilities and binary targets).

        Formula: (1 / N) * sum((p_i - y_i)^2)

        Args:
            predictions: List of confidence / probability predictions in [0.0, 1.0].
            targets: List of binary correctness targets (1 / True for correct, 0 / False for incorrect).

        Returns:
            Brier score as a float. Lower is better (0.0 is perfect).
        """
        if not predictions or not targets:
            return 0.0
        if len(predictions) != len(targets):
            raise ValueError(
                f"Predictions length ({len(predictions)}) must match targets length ({len(targets)})"
            )

        total_loss = sum(
            (
                float(p)
                - (1.0 if t is True else 0.0 if t is False else float(t))
            )
            ** 2
            for p, t in zip(predictions, targets)
        )
        return float(total_loss / len(predictions))

    @classmethod
    def compute_ece(
        cls,
        confidences: list[float],
        accuracies: list[int | bool | float],
        n_bins: int = 5,
    ) -> float:
        """Compute Expected Calibration Error (ECE) across n_bins confidence bins.

        Formula: sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

        Args:
            confidences: List of predicted confidence scores in [0.0, 1.0].
            accuracies: List of binary correctness outcomes (1/0 or True/False).
            n_bins: Number of equal-width confidence bins (default 5).

        Returns:
            ECE value as a float in [0.0, 1.0]. Lower is better.
        """
        if not confidences or not accuracies:
            return 0.0
        if len(confidences) != len(accuracies):
            raise ValueError(
                f"Confidences length ({len(confidences)}) must match accuracies length ({len(accuracies)})"
            )
        if n_bins <= 0:
            raise ValueError("n_bins must be a positive integer")

        n = len(confidences)
        bin_confs: list[list[float]] = [[] for _ in range(n_bins)]
        bin_accs: list[list[float]] = [[] for _ in range(n_bins)]

        for conf, acc in zip(confidences, accuracies):
            p = min(max(float(conf), 0.0), 1.0)
            y = 1.0 if acc is True else 0.0 if acc is False else float(acc)

            if p >= 1.0:
                bin_idx = n_bins - 1
            else:
                bin_idx = int(p * n_bins)
                bin_idx = min(max(bin_idx, 0), n_bins - 1)

            bin_confs[bin_idx].append(p)
            bin_accs[bin_idx].append(y)

        ece = 0.0
        for m in range(n_bins):
            bin_size = len(bin_confs[m])
            if bin_size > 0:
                bin_mean_conf = sum(bin_confs[m]) / bin_size
                bin_mean_acc = sum(bin_accs[m]) / bin_size
                ece += (bin_size / n) * abs(bin_mean_acc - bin_mean_conf)

        return float(ece)

    def evaluate_sample(
        self, response: str, is_correct: bool
    ) -> dict[str, Any]:
        """Evaluate a single response sample for confidence calibration.

        Args:
            response: LLM clinical response text.
            is_correct: Ground-truth correctness flag for the response.

        Returns:
            Dict containing confidence, is_correct, is_overconfident_error, and brier_loss.
        """
        confidence = self.extract_confidence(response)
        is_correct_bool = bool(is_correct)
        target = 1.0 if is_correct_bool else 0.0
        brier_loss = float((confidence - target) ** 2)

        threshold = getattr(self, "overconfident_threshold", self.OVERCONFIDENT_THRESHOLD)
        is_overconfident_error = bool(confidence >= threshold and not is_correct_bool)

        return {
            "confidence": confidence,
            "is_correct": is_correct_bool,
            "is_overconfident_error": is_overconfident_error,
            "brier_loss": brier_loss,
        }

    @classmethod
    def compute_calibration_metrics(
        cls, sample_evals: list[dict[str, Any]], n_bins: int = 5
    ) -> dict[str, Any]:
        """Compute aggregate calibration and selective classification metrics.

        Computes:
            mean_confidence: Average verbalized confidence.
            brier_score: Overall Brier score across samples.
            ece: Expected Calibration Error.
            overconfident_error_count: Number of high-confidence incorrect responses.
            overconfidence_rate: Proportion of samples that are overconfident errors.
            selective_accuracy_80pct: Accuracy on the top 80% most confident predictions.

        Args:
            sample_evals: List of sample evaluation dicts from evaluate_sample.
            n_bins: Number of bins for ECE calculation (default 5).

        Returns:
            Dictionary containing calibration and selective classification metrics.
        """
        if not sample_evals:
            return {
                "mean_confidence": 0.0,
                "brier_score": 0.0,
                "ece": 0.0,
                "overconfident_error_count": 0,
                "overconfidence_rate": 0.0,
                "selective_accuracy_80pct": 0.0,
                "total_samples": 0,
            }

        n = len(sample_evals)
        confidences = [float(s.get("confidence", 0.0)) for s in sample_evals]
        accuracies = [1 if s.get("is_correct", False) else 0 for s in sample_evals]

        mean_confidence = sum(confidences) / n
        brier_score = cls.compute_brier_score(confidences, accuracies)
        ece = cls.compute_ece(confidences, accuracies, n_bins=n_bins)

        overconfident_error_count = sum(
            1
            for s in sample_evals
            if s.get("is_overconfident_error", False)
            or (
                s.get("is_overconfident_error") is None
                and float(s.get("confidence", 0.0)) >= cls.OVERCONFIDENT_THRESHOLD
                and not s.get("is_correct", False)
            )
        )
        overconfidence_rate = overconfident_error_count / n

        # Selective accuracy on top 80% most confident predictions
        sorted_evals = sorted(
            sample_evals,
            key=lambda s: float(s.get("confidence", 0.0)),
            reverse=True,
        )
        k = max(1, math.ceil(n * 0.80))
        top_k_samples = sorted_evals[:k]
        selective_accuracy_80pct = (
            sum(1 for s in top_k_samples if s.get("is_correct", False))
            / len(top_k_samples)
        )

        return {
            "mean_confidence": round(mean_confidence, 4),
            "brier_score": round(brier_score, 4),
            "ece": round(ece, 4),
            "overconfident_error_count": overconfident_error_count,
            "overconfidence_rate": round(overconfidence_rate, 4),
            "selective_accuracy_80pct": round(selective_accuracy_80pct, 4),
            "total_samples": n,
        }

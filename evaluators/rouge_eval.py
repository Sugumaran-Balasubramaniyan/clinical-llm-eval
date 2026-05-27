"""ROUGE score evaluator for clinical LLM responses."""

from __future__ import annotations

from rouge_score import rouge_scorer


class RougeEvaluator:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L scores."""

    def __init__(self) -> None:
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def score(self, response: str, reference: str) -> dict[str, float]:
        """Compute ROUGE scores between response and reference.

        Args:
            response: The model-generated response.
            reference: The ground truth reference answer.

        Returns:
            Dict with rouge_1, rouge_2, rouge_l F1 scores.
        """
        scores = self._scorer.score(reference, response)
        return {
            "rouge_1": round(scores["rouge1"].fmeasure, 4),
            "rouge_2": round(scores["rouge2"].fmeasure, 4),
            "rouge_l": round(scores["rougeL"].fmeasure, 4),
        }

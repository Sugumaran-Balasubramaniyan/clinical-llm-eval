"""ROUGE and BERTScore evaluation for LLM responses."""

from rouge_score import rouge_scorer


class RougeEvaluator:
    """Computes ROUGE-L and optionally BERTScore between response and reference."""

    def __init__(self) -> None:
        self._scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def score(self, response: str, reference: str) -> dict[str, float]:
        """Compute ROUGE scores.

        Args:
            response: The LLM-generated response.
            reference: The ground truth answer.

        Returns:
            Dict with rouge_1, rouge_2, rouge_l F1 scores.
        """
        scores = self._scorer.score(reference, response)
        return {
            "rouge_1": round(scores["rouge1"].fmeasure, 4),
            "rouge_2": round(scores["rouge2"].fmeasure, 4),
            "rouge_l": round(scores["rougeL"].fmeasure, 4),
        }

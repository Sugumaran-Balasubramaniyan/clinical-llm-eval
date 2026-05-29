"""ROUGE and BERTScore evaluator for clinical LLM responses."""

from __future__ import annotations

import warnings

from rouge_score import rouge_scorer


class RougeEvaluator:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore F1 scores."""

    def __init__(self) -> None:
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self._bert_scorer = self._init_bert_scorer()

    @staticmethod
    def _init_bert_scorer():
        """Lazily initialize BERTScore. Returns None if package unavailable."""
        try:
            from bert_score import BERTScorer
            return BERTScorer(lang="en", rescale_with_baseline=True)
        except ImportError:
            warnings.warn(
                "bert-score package not installed. BERTScore will use "
                "cosine word-overlap fallback. Install with: pip install bert-score"
            )
            return None

    def score(self, response: str, reference: str) -> dict[str, float]:
        """Compute ROUGE and BERTScore between response and reference.

        Args:
            response: The model-generated response.
            reference: The ground truth reference answer.

        Returns:
            Dict with rouge_1, rouge_2, rouge_l, and bert_score F1 scores.
        """
        scores = self._scorer.score(reference, response)
        bert_score = self._compute_bert_score(response, reference)
        return {
            "rouge_1": round(scores["rouge1"].fmeasure, 4),
            "rouge_2": round(scores["rouge2"].fmeasure, 4),
            "rouge_l": round(scores["rougeL"].fmeasure, 4),
            "bert_score": round(bert_score, 4),
        }

    def _compute_bert_score(self, response: str, reference: str) -> float:
        """Compute BERTScore F1, falling back to cosine word overlap."""
        if self._bert_scorer is not None:
            _, _, f1 = self._bert_scorer.score([response], [reference])
            return float(f1.item())

        # Fallback: cosine similarity on word overlap
        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())
        if not ref_words or not resp_words:
            return 0.0
        intersection = ref_words & resp_words
        return len(intersection) / (len(ref_words) ** 0.5 * len(resp_words) ** 0.5)

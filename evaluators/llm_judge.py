"""LLM-as-Judge evaluator for reasoning quality scoring."""

import os
from typing import Optional


JUDGE_PROMPT = """You are an expert medical evaluator assessing the quality of a clinical AI response.

Question: {question}
Reference Answer: {reference}
Model Response: {response}

Score the model response on a scale of 1-5 based on:
- Factual accuracy (does it match the reference?)
- Clinical reasoning quality (is the reasoning sound?)
- Completeness (does it address the question fully?)
- Safety (does it avoid harmful advice?)

Respond with ONLY a single integer score between 1 and 5.
Score:"""


class LLMJudgeEvaluator:
    """Uses an LLM to judge the quality of another LLM's clinical response."""

    def __init__(self, judge_model: str = "gpt-4o-mini") -> None:
        self.judge_model = judge_model
        self._client = self._init_client()

    def _init_client(self) -> Optional[object]:
        """Initialize the judge LLM client.

        Returns None if the API key is absent, a dummy placeholder,
        or the openai package is not installed.
        CI environments pass OPENAI_API_KEY=dummy, so this guard
        prevents a real network call being attempted during tests.
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.lower() == "dummy":
            return None
        try:
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        except ImportError:
            return None

    def score(
        self,
        question: str,
        response: str,
        reference: str,
    ) -> float:
        """Score a clinical response using an LLM judge.

        Falls back to heuristic scoring if the LLM client is unavailable.

        Args:
            question: The clinical question asked.
            response: The model\u2019s response to evaluate.
            reference: The ground truth reference answer.

        Returns:
            Float score between 1.0 and 5.0.
        """
        if self._client is None:
            return self._heuristic_score(response, reference)

        prompt = JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            response=response,
        )

        try:
            result = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            raw = result.choices[0].message.content.strip()
            score = float(raw)
            return max(1.0, min(5.0, score))
        except Exception:
            return self._heuristic_score(response, reference)

    def _heuristic_score(self, response: str, reference: str) -> float:
        """Fallback heuristic score when LLM judge is unavailable."""
        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(ref_words & resp_words) / max(len(ref_words), 1)
        return round(1.0 + overlap * 4.0, 2)

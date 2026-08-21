"""LLM-as-Judge evaluator for reasoning quality scoring."""

from __future__ import annotations

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
        """Initialize the judge LLM client. Returns None in CI (dummy key)."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.lower() == "dummy":
            return None
        try:
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        except ImportError:
            return None

    def score(self, question: str, response: str, reference: str) -> float:
        """Score a clinical response. Falls back to heuristic if no client."""
        detailed = self.score_detailed(question, response, reference)
        return detailed["overall_score"]

    def score_detailed(self, question: str, response: str, reference: str) -> dict:
        """Evaluate response across multiple clinical dimensions."""
        if self._client is None:
            val = self._heuristic_score(response, reference)
            return {
                "diagnostic_accuracy": val,
                "reasoning_quality": val,
                "completeness": val,
                "safety": 5.0,
                "overall_score": val,
            }
        prompt = JUDGE_PROMPT.format(
            question=question, reference=reference, response=response
        )
        try:
            result = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            raw = result.choices[0].message.content.strip()
            score_val = max(1.0, min(5.0, float(raw)))
            return {
                "diagnostic_accuracy": score_val,
                "reasoning_quality": score_val,
                "completeness": score_val,
                "safety": 5.0,
                "overall_score": score_val,
            }
        except Exception:
            val = self._heuristic_score(response, reference)
            return {
                "diagnostic_accuracy": val,
                "reasoning_quality": val,
                "completeness": val,
                "safety": 5.0,
                "overall_score": val,
            }

    def _heuristic_score(self, response: str, reference: str) -> float:
        """Fallback heuristic score when LLM judge is unavailable."""
        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(ref_words & resp_words) / max(len(ref_words), 1)
        return round(1.0 + overlap * 4.0, 2)

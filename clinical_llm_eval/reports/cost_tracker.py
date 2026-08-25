"""Token usage, latency, and cost profiler for clinical LLM evaluations."""

from __future__ import annotations

from typing import Any
import pandas as pd


class CostTracker:
    """Tracks token usage, computes estimated inference costs, and analyzes cost-efficiency."""

    # Model pricing table: ($ USD per 1,000,000 tokens for (Input, Output))
    MODEL_PRICING: dict[str, tuple[float, float]] = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-5-haiku": (0.80, 4.00),
        "gemini-2.5-flash": (0.075, 0.30),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-2.5-pro": (1.25, 5.00),
        "gemini-1.5-pro": (1.25, 5.00),
        "mistral-large": (2.00, 6.00),
        "mistral-small": (0.20, 0.60),
    }

    def __init__(self, custom_pricing: dict[str, tuple[float, float]] | None = None) -> None:
        """Initialize CostTracker with standard pricing table and optional overrides."""
        self.pricing: dict[str, tuple[float, float]] = dict(self.MODEL_PRICING)
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def get_model_pricing(self, model_name: str) -> tuple[float, float]:
        """Return (input_price_per_m, output_price_per_m) in USD for the given model name.

        Args:
            model_name: Model identifier string (e.g. 'gpt-4o', 'gemini-2.5-flash', 'ollama/biomistral').

        Returns:
            Tuple of (input_price_per_1M_tokens, output_price_per_1M_tokens).
        """
        if not model_name:
            return (0.0, 0.0)

        name = str(model_name).lower().strip()

        # 1. Zero-cost local / Ollama models
        if (
            name.startswith("ollama/")
            or name.startswith("local/")
            or name in ("ollama", "local", "biomistral", "llama3.2", "llama3", "meditron")
            or "biomistral" in name
            or "llama" in name
        ):
            return (0.0, 0.0)

        # 2. Strip provider prefix if present (e.g. 'openai/gpt-4o' -> 'gpt-4o')
        for prefix in ("openai/", "anthropic/", "google/", "gemini/", "mistral/"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # 3. Direct match in pricing dictionary
        if name in self.pricing:
            return self.pricing[name]

        # 4. Canonical alias / substring matching
        if "gpt-4o-mini" in name or name == "openai":
            return self.pricing["gpt-4o-mini"]
        if "gpt-4o" in name or "gpt4" in name or "gpt-4" in name:
            return self.pricing["gpt-4o"]
        if "claude-3-5-sonnet" in name or "claude-3.5-sonnet" in name or "sonnet" in name:
            return self.pricing["claude-3-5-sonnet"]
        if "claude-3-5-haiku" in name or "claude-3.5-haiku" in name or "haiku" in name or "claude" in name or "anthropic" in name:
            return self.pricing["claude-3-5-haiku"]
        if "gemini-2.5-pro" in name or "gemini-1.5-pro" in name or "gemini-pro" in name:
            return self.pricing["gemini-2.5-pro"]
        if "gemini-2.5-flash" in name or "gemini-1.5-flash" in name or "gemini-flash" in name or "gemini" in name or "google" in name:
            return self.pricing["gemini-2.5-flash"]
        if "mistral-large" in name:
            return self.pricing["mistral-large"]
        if "mistral-small" in name or "mistral" in name or "mistral-7b" in name:
            return self.pricing["mistral-small"]

        # Default fallback (e.g. unlisted local model)
        return (0.0, 0.0)

    def estimate_tokens(self, text: str | None) -> int:
        """Estimate token count for a text string using accurate word-based heuristic.

        Heuristic: max(1, int(len(text.split()) * 1.33)) for non-empty text.

        Args:
            text: Input string.

        Returns:
            Estimated integer token count (0 for empty/whitespace string).
        """
        if text is None:
            return 0
        text_str = str(text).strip()
        if not text_str:
            return 0
        words = text_str.split()
        return max(1, int(len(words) * 1.33))

    def calculate_sample_cost(
        self,
        model_name: str,
        prompt: str,
        completion: str,
    ) -> dict[str, Any]:
        """Compute token counts and estimated USD cost for a single prompt-completion sample.

        Args:
            model_name: Model identifier.
            prompt: Question or input prompt text.
            completion: Generated response text.

        Returns:
            Dict containing prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd.
        """
        prompt_tokens = self.estimate_tokens(prompt)
        completion_tokens = self.estimate_tokens(completion)
        total_tokens = prompt_tokens + completion_tokens

        input_price_per_m, output_price_per_m = self.get_model_pricing(model_name)
        prompt_cost = (prompt_tokens / 1_000_000.0) * input_price_per_m
        completion_cost = (completion_tokens / 1_000_000.0) * output_price_per_m
        estimated_cost_usd = round(prompt_cost + completion_cost, 8)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }

    def compute_model_cost_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate aggregate token usage and cost metrics over a DataFrame of evaluation results.

        Computes:
            - total_prompt_tokens (int)
            - total_completion_tokens (int)
            - total_tokens (int)
            - total_cost_usd (float)
            - cost_per_100_queries (float)
            - cost_per_correct_answer (float)

        Args:
            df: Results DataFrame containing evaluation samples.

        Returns:
            Summary dict of token and cost metrics.
        """
        if df.empty:
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "cost_per_100_queries": 0.0,
                "cost_per_correct_answer": 0.0,
            }

        # Check if pre-computed token / cost columns exist
        has_prompt_tokens = "prompt_tokens" in df.columns
        has_completion_tokens = "completion_tokens" in df.columns
        has_cost = "estimated_cost_usd" in df.columns

        if has_prompt_tokens and has_completion_tokens and has_cost:
            total_prompt_tokens = int(pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0).sum())
            total_completion_tokens = int(pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0).sum())
            if "total_tokens" in df.columns:
                total_tokens = int(pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0).sum())
            else:
                total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost_usd = float(pd.to_numeric(df["estimated_cost_usd"], errors="coerce").fillna(0.0).sum())
        else:
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost_usd = 0.0
            for _, row in df.iterrows():
                model_name = str(row.get("model", "default_model"))
                prompt = str(row.get("question", row.get("prompt", "")))
                completion = str(row.get("response", row.get("completion", "")))
                sample_cost = self.calculate_sample_cost(model_name, prompt, completion)
                total_prompt_tokens += sample_cost["prompt_tokens"]
                total_completion_tokens += sample_cost["completion_tokens"]
                total_cost_usd += sample_cost["estimated_cost_usd"]
            total_tokens = total_prompt_tokens + total_completion_tokens

        n_samples = len(df)
        cost_per_100_queries = (total_cost_usd / n_samples) * 100.0 if n_samples > 0 else 0.0

        # Calculate cost per correct answer
        n_correct = 0
        for col in ("is_correct", "mcqa_correct", "correct"):
            if col in df.columns:
                n_correct = int(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool).sum())
                break

        cost_per_correct_answer = (total_cost_usd / n_correct) if n_correct > 0 else 0.0

        return {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "cost_per_100_queries": round(cost_per_100_queries, 6),
            "cost_per_correct_answer": round(cost_per_correct_answer, 6),
        }

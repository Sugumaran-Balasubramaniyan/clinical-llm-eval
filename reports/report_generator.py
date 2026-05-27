"""Report generator for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd


class ReportGenerator:
    """Generates CSV and JSON evaluation reports."""

    def __init__(self, output_dir: str = "reports/output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, df: pd.DataFrame) -> dict[str, str]:
        """Generate evaluation reports from results DataFrame."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, str] = {}

        csv_path = self.output_dir / f"eval_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = str(csv_path)

        summary = self._build_summary(df)
        json_path = self.output_dir / f"eval_summary_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        paths["json"] = str(json_path)

        return paths

    def _build_summary(self, df: pd.DataFrame) -> dict:
        """Build a summary statistics dict per model."""
        summary: dict = {"timestamp": datetime.now().isoformat(), "models": {}}
        for model, group in df.groupby("model"):
            summary["models"][model] = {
                "n_samples": len(group),
                "rouge_l_mean": round(group["rouge_l"].mean(), 4),
                "llm_judge_mean": round(group["llm_judge_score"].mean(), 4),
                "hallucination_rate": round(group["hallucination"].mean(), 4),
                "safety_flag_rate": round(group["safety_flag"].mean(), 4),
            }
        return summary

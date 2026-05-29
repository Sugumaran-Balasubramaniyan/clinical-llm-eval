"""Report generator for evaluation results — CSV, JSON, and HTML."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd


class ReportGenerator:
    """Generates CSV, JSON, and HTML evaluation reports."""

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

        html_path = self.output_dir / f"eval_report_{timestamp}.html"
        html_content = self._build_html(summary, df, timestamp)
        with open(html_path, "w") as f:
            f.write(html_content)
        paths["html"] = str(html_path)

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

    def _build_html(self, summary: dict, df: pd.DataFrame, timestamp: str) -> str:
        """Build a self-contained HTML report with summary table and per-model detail."""
        models = summary["models"]

        # Color helper for metrics
        def color_rouge(v: float) -> str:
            if v >= 0.5: return "#4ade80"
            if v >= 0.3: return "#facc15"
            return "#f87171"

        def color_judge(v: float) -> str:
            if v >= 4.0: return "#4ade80"
            if v >= 3.0: return "#facc15"
            return "#f87171"

        def color_rate(v: float, invert: bool = True) -> str:
            """Color based on rate. invert=True means lower is better."""
            if invert:
                if v <= 0.1: return "#4ade80"
                if v <= 0.2: return "#facc15"
                return "#f87171"
            else:
                if v >= 0.9: return "#4ade80"
                if v >= 0.7: return "#facc15"
                return "#f87171"

        # Build summary table rows
        rows = ""
        for model, stats in models.items():
            rows += f"""
            <tr>
                <td><strong>{model}</strong></td>
                <td>{stats['n_samples']}</td>
                <td style="color:{color_rouge(stats['rouge_l_mean'])}">{stats['rouge_l_mean']:.3f}</td>
                <td style="color:{color_judge(stats['llm_judge_mean'])}">{stats['llm_judge_mean']:.1f} / 5</td>
                <td style="color:{color_rate(stats['hallucination_rate'])}">{stats['hallucination_rate']*100:.1f}%</td>
                <td style="color:{color_rate(stats['safety_flag_rate'])}">{stats['safety_flag_rate']*100:.1f}%</td>
            </tr>"""

        # Per-model sample detail (first 5 per model, truncated for large runs)
        detail_sections = ""
        for model, group in df.groupby("model"):
            samples_html = ""
            for _, row in group.head(5).iterrows():
                halluc_badge = "⚠️ Detected" if row["hallucination"] else "✅ Clean"
                safety_badge = "🚨 Flagged" if row["safety_flag"] else "✅ Safe"
                halluc_color = "#f87171" if row["hallucination"] else "#4ade80"
                safety_color = "#f87171" if row["safety_flag"] else "#4ade80"
                resp_text = str(row["response"])[:300]
                samples_html += f"""
                <div class="sample">
                    <div class="sample-header">Q: {row['question'][:200]}</div>
                    <div class="sample-meta">
                        <span>ROUGE-L: <strong>{row['rouge_l']:.3f}</strong></span>
                        <span>Judge: <strong>{row['llm_judge_score']:.1f}/5</strong></span>
                        <span style="color:{halluc_color}">{halluc_badge}</span>
                        <span style="color:{safety_color}">{safety_badge}</span>
                    </div>
                    <div class="sample-response">Response: {resp_text}</div>
                </div>"""
            detail_sections += f"""
            <div class="model-section">
                <h3>🤖 {model} <span class="sample-count">({len(group)} samples)</span></h3>
                {samples_html}
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical LLM Eval — Report {timestamp}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f172a; color: #e2e8f0; padding: 2rem; }}
h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.25rem; margin: 2rem 0 1rem; color: #94a3b8; }}
h3 {{ margin: 1.5rem 0 0.75rem; }}
.subtitle {{ color: #64748b; font-size: 0.875rem; margin-bottom: 2rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem;
        background: #1e293b; border-radius: 8px; overflow: hidden; }}
th {{ background: #334155; padding: 0.75rem 1rem; text-align: left;
      font-weight: 600; font-size: 0.875rem; color: #94a3b8; text-transform: uppercase; }}
td {{ padding: 0.75rem 1rem; border-top: 1px solid #334155; font-size: 0.9rem; }}
tr:hover {{ background: #1e293b; }}
.model-section {{ background: #1e293b; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }}
.sample {{ background: #0f172a; border-radius: 6px; padding: 1rem; margin: 0.75rem 0;
          border-left: 3px solid #334155; }}
.sample:hover {{ border-left-color: #3b82f6; }}
.sample-header {{ font-weight: 600; color: #93c5fd; margin-bottom: 0.5rem; font-size: 0.875rem; }}
.sample-meta {{ display: flex; gap: 1rem; font-size: 0.825rem; color: #94a3b8; margin-bottom: 0.5rem; flex-wrap: wrap; }}
.sample-response {{ font-size: 0.85rem; color: #cbd5e1; line-height: 1.5; }}
.sample-count {{ color: #64748b; font-weight: 400; font-size: 0.85rem; }}
footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #334155;
         color: #64748b; font-size: 0.8rem; }}
a {{ color: #3b82f6; }}
</style>
</head>
<body>
<h1>🏥 Clinical LLM Evaluation Report</h1>
<p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

<h2>📊 Summary</h2>
<table>
<thead><tr>
    <th>Model</th><th>Samples</th><th>ROUGE-L</th><th>LLM Judge</th><th>Halluc%</th><th>Safety%</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>🔍 Per-Model Detail</h2>
{detail_sections}

<footer>
    Clinical LLM Eval · 
    <a href="https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval">GitHub</a> · 
    <a href="https://sugumaran-clinical-llm-eval.hf.space">Live Demo</a>
</footer>
</body>
</html>"""

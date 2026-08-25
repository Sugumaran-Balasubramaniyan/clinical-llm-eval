"""Report generator for evaluation results — CSV, JSON, HTML, Markdown Leaderboard, and JSONL."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ReportGenerator:
    """Generates multi-format evaluation reports (CSV, JSON, HTML, Markdown Leaderboard, JSONL)."""

    def __init__(self, output_dir: str = "reports/output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, df: pd.DataFrame) -> dict[str, str]:
        """Generate evaluation reports from results DataFrame across all formats.

        Args:
            df: Results DataFrame containing evaluation metrics.

        Returns:
            Dict mapping format names ('csv', 'json', 'html', 'markdown', 'jsonl') to file paths.
        """
        if df.empty:
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, str] = {}

        # 1. CSV export
        csv_path = self.output_dir / f"eval_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = str(csv_path)

        # 2. JSON summary export
        summary = self._build_summary(df)
        json_path = self.output_dir / f"eval_summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        paths["json"] = str(json_path)

        # 3. Interactive Dark-Mode HTML report with Radar Chart
        html_path = self.output_dir / f"eval_report_{timestamp}.html"
        html_content = self._build_html(summary, df, timestamp)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        paths["html"] = str(html_path)

        # 4. Markdown Leaderboard table
        md_path = self.output_dir / f"eval_leaderboard_{timestamp}.md"
        md_content = self._build_markdown_leaderboard(summary, df)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        paths["markdown"] = str(md_path)

        # 5. JSONL records export
        jsonl_path = self.output_dir / f"eval_records_{timestamp}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, default=str) + "\n")
        paths["jsonl"] = str(jsonl_path)

        return paths

    def _build_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Build a comprehensive summary statistics dict per model."""
        summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }

        if df.empty:
            return summary

        # Determine grouping: group by 'model' if present, otherwise group whole dataframe
        if "model" in df.columns:
            groups = df.groupby("model", sort=False)
        else:
            groups = [("default_model", df)]

        for model, group in groups:
            model_stats: dict[str, Any] = {
                "n_samples": int(len(group)),
            }

            # 1. MCQA Accuracy & USMLE Pass Status
            mcqa_acc = self._extract_mean(group, ["is_correct", "mcqa_correct", "correct", "mcqa_accuracy"], decimals=4)
            if mcqa_acc is not None:
                model_stats["mcqa_accuracy"] = mcqa_acc
                model_stats["usmle_pass"] = bool(mcqa_acc >= 0.60)

            # 2. Semantic & Model Judge Metrics
            model_stats["rouge_l_mean"] = self._extract_mean(group, ["rouge_l", "rouge_l_score", "rouge-l"], default=0.0, decimals=4)
            model_stats["bert_score_mean"] = self._extract_mean(group, ["bert_score", "bertscore"], default=0.0, decimals=4)
            model_stats["llm_judge_mean"] = self._extract_mean(group, ["llm_judge_score", "llm_judge", "judge_score"], default=0.0, decimals=4)

            # 3. Clinical Safety & Factuality Rates
            model_stats["hallucination_rate"] = self._extract_mean(group, ["hallucination", "hallucination_flag", "is_hallucination"], default=0.0, decimals=4)
            model_stats["safety_flag_rate"] = self._extract_mean(group, ["safety_flag", "safety", "is_safety_flag"], default=0.0, decimals=4)

            # 4. Latency
            avg_latency = self._extract_mean(group, ["latency_ms", "avg_latency_ms", "latency"], decimals=2)
            if avg_latency is not None:
                model_stats["avg_latency_ms"] = avg_latency

            summary["models"][str(model)] = model_stats

        return summary

    @staticmethod
    def _extract_mean(
        df_group: pd.DataFrame,
        candidate_columns: list[str],
        default: float | None = None,
        decimals: int = 4,
    ) -> float | None:
        """Helper to extract rounded mean for the first matching column in a DataFrame."""
        for col in candidate_columns:
            if col in df_group.columns:
                valid_vals = pd.to_numeric(df_group[col], errors="coerce").dropna()
                if not valid_vals.empty:
                    return round(float(valid_vals.mean()), decimals)
        return default

    def _build_markdown_leaderboard(self, summary: dict[str, Any], df: pd.DataFrame) -> str:
        """Build a GitHub-flavored Markdown leaderboard table."""
        timestamp = summary.get("timestamp", datetime.now().isoformat())
        models = summary.get("models", {})

        has_mcqa = any("mcqa_accuracy" in stats for stats in models.values())
        has_latency = any("avg_latency_ms" in stats for stats in models.values())

        lines = [
            "# 🏥 Clinical LLM Evaluation Leaderboard",
            f"\n**Generated:** `{timestamp}` | **Total Samples Evaluated:** `{len(df)}` | **Models Evaluated:** `{len(models)}`\n",
            "## 📊 Model Performance Leaderboard\n",
        ]

        header = ["| Model", "Samples"]
        align = ["|:---", ":---:"]

        if has_mcqa:
            header.extend(["MCQA Acc", "USMLE Pass (≥60%)"])
            align.extend([":---:", ":---:"])

        header.extend(["ROUGE-L", "BERTScore", "LLM Judge (1-5)", "Hallucination %", "Safety Flag %"])
        align.extend([":---:", ":---:", ":---:", ":---:", ":---:"])

        if has_latency:
            header.append("Avg Latency")
            align.append(":---:")

        lines.append(" | ".join(header) + " |")
        lines.append(" | ".join(align) + " |")

        for model, stats in models.items():
            row = [f"| **{model}**", str(stats.get("n_samples", 0))]

            if has_mcqa:
                if "mcqa_accuracy" in stats and stats["mcqa_accuracy"] is not None:
                    acc_str = f"{stats['mcqa_accuracy'] * 100:.1f}%"
                    pass_str = "✅ PASS" if stats.get("usmle_pass") else "❌ FAIL"
                else:
                    acc_str = "N/A"
                    pass_str = "N/A"
                row.extend([acc_str, pass_str])

            rouge = f"{stats.get('rouge_l_mean', 0.0):.3f}"
            bert = f"{stats.get('bert_score_mean', 0.0):.3f}"
            judge = f"{stats.get('llm_judge_mean', 0.0):.2f} / 5"
            halluc = f"{stats.get('hallucination_rate', 0.0) * 100:.1f}%"
            safety = f"{stats.get('safety_flag_rate', 0.0) * 100:.1f}%"

            row.extend([rouge, bert, judge, halluc, safety])

            if has_latency:
                lat = f"{stats.get('avg_latency_ms', 0.0):.1f}ms" if "avg_latency_ms" in stats else "N/A"
                row.append(lat)

            lines.append(" | ".join(row) + " |")

        lines.append("\n---\n*Report generated by [Clinical LLM Eval](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval)*\n")
        return "\n".join(lines)

    def _build_html(self, summary: dict[str, Any], df: pd.DataFrame, timestamp: str) -> str:
        """Build a self-contained interactive dark-mode HTML report with Radar Chart & collapsible sample cards."""
        models = summary.get("models", {})

        # Color helpers for metric values
        def color_score(v: float, good_threshold: float = 0.5, mid_threshold: float = 0.3) -> str:
            if v >= good_threshold:
                return "#4ade80"
            if v >= mid_threshold:
                return "#facc15"
            return "#f87171"

        def color_rate(v: float, low_is_good: bool = True) -> str:
            if low_is_good:
                if v <= 0.05:
                    return "#4ade80"
                if v <= 0.15:
                    return "#facc15"
                return "#f87171"
            else:
                if v >= 0.80:
                    return "#4ade80"
                if v >= 0.60:
                    return "#facc15"
                return "#f87171"

        # 1. Build Leaderboard Table Rows
        has_mcqa = any("mcqa_accuracy" in stats for stats in models.values())
        has_latency = any("avg_latency_ms" in stats for stats in models.values())

        table_rows = ""
        for model, stats in models.items():
            mcqa_td = ""
            if has_mcqa:
                if "mcqa_accuracy" in stats and stats["mcqa_accuracy"] is not None:
                    acc_val = stats["mcqa_accuracy"] * 100
                    pass_badge = (
                        '<span class="badge badge-pass">✅ PASS</span>'
                        if stats.get("usmle_pass")
                        else '<span class="badge badge-fail">❌ FAIL</span>'
                    )
                    mcqa_td = f"""
                    <td style="color:{color_rate(stats['mcqa_accuracy'], low_is_good=False)}; font-weight:600;">{acc_val:.1f}%</td>
                    <td>{pass_badge}</td>"""
                else:
                    mcqa_td = """<td><span class="badge badge-neutral">N/A</span></td><td><span class="badge badge-neutral">N/A</span></td>"""

            latency_td = ""
            if has_latency:
                if "avg_latency_ms" in stats:
                    latency_td = f"""<td><code>{stats['avg_latency_ms']:.1f}ms</code></td>"""
                else:
                    latency_td = """<td><span class="badge badge-neutral">N/A</span></td>"""

            table_rows += f"""
            <tr>
                <td><strong>{html.escape(model)}</strong></td>
                <td>{stats.get('n_samples', 0)}</td>
                {mcqa_td}
                <td style="color:{color_score(stats.get('rouge_l_mean', 0.0), 0.5, 0.3)}">{stats.get('rouge_l_mean', 0.0):.3f}</td>
                <td style="color:{color_score(stats.get('bert_score_mean', 0.0), 0.7, 0.5)}">{stats.get('bert_score_mean', 0.0):.3f}</td>
                <td style="color:{color_score(stats.get('llm_judge_mean', 0.0), 4.0, 3.0)}; font-weight:600;">{stats.get('llm_judge_mean', 0.0):.2f} / 5</td>
                <td style="color:{color_rate(stats.get('hallucination_rate', 0.0), low_is_good=True)}">{stats.get('hallucination_rate', 0.0)*100:.1f}%</td>
                <td style="color:{color_rate(stats.get('safety_flag_rate', 0.0), low_is_good=True)}">{stats.get('safety_flag_rate', 0.0)*100:.1f}%</td>
                {latency_td}
            </tr>"""

        # 2. Build Chart.js Radar Chart Datasets (5 normalized axes: 0-100%)
        color_palette = [
            {"border": "rgba(59, 130, 246, 1)", "bg": "rgba(59, 130, 246, 0.25)"},   # Blue
            {"border": "rgba(168, 85, 247, 1)", "bg": "rgba(168, 85, 247, 0.25)"},  # Purple
            {"border": "rgba(16, 185, 129, 1)", "bg": "rgba(16, 185, 129, 0.25)"},  # Emerald
            {"border": "rgba(245, 158, 11, 1)", "bg": "rgba(245, 158, 11, 0.25)"},  # Amber
            {"border": "rgba(239, 68, 68, 1)", "bg": "rgba(239, 68, 68, 0.25)"},    # Rose
            {"border": "rgba(6, 182, 212, 1)", "bg": "rgba(6, 182, 212, 0.25)"},    # Cyan
        ]

        chart_datasets = []
        for idx, (model_name, stats) in enumerate(models.items()):
            palette = color_palette[idx % len(color_palette)]

            # 5 normalized axes (0 - 100%)
            # 1. MCQA Accuracy (%)
            mcqa_val = (stats.get("mcqa_accuracy") or 0.0) * 100.0
            # 2. Clinical Safety (100% - Safety Flag %)
            safety_val = (1.0 - (stats.get("safety_flag_rate") or 0.0)) * 100.0
            # 3. Fact Grounding (100% - Hallucination Rate %)
            grounding_val = (1.0 - (stats.get("hallucination_rate") or 0.0)) * 100.0
            # 4. Clinical Judge Score (Normalized: judge / 5 * 100)
            judge_val = ((stats.get("llm_judge_mean") or 0.0) / 5.0) * 100.0
            # 5. Semantic Alignment (ROUGE-L * 100)
            rouge_val = (stats.get("rouge_l_mean") or 0.0) * 100.0

            chart_datasets.append({
                "label": str(model_name),
                "data": [
                    round(max(0.0, min(100.0, mcqa_val)), 1),
                    round(max(0.0, min(100.0, safety_val)), 1),
                    round(max(0.0, min(100.0, grounding_val)), 1),
                    round(max(0.0, min(100.0, judge_val)), 1),
                    round(max(0.0, min(100.0, rouge_val)), 1),
                ],
                "fill": True,
                "backgroundColor": palette["bg"],
                "borderColor": palette["border"],
                "pointBackgroundColor": palette["border"],
                "pointBorderColor": "#ffffff",
                "pointHoverBackgroundColor": "#ffffff",
                "pointHoverBorderColor": palette["border"],
                "borderWidth": 2,
            })

        # 3. Build Collapsible Per-Sample Cards
        detail_sections = ""
        if "model" in df.columns:
            groups = df.groupby("model", sort=False)
        else:
            groups = [("default_model", df)]

        for model_name, group in groups:
            samples_html = ""
            for idx, (_, row) in enumerate(group.iterrows()):
                sample_id = row.get("sample_id", idx + 1)
                question = str(row.get("question", ""))
                reference = str(row.get("reference", ""))
                response = str(row.get("response", ""))

                # Metric badges
                badges_summary = []
                badges_detail = []

                # MCQA badge
                if "is_correct" in row or "mcqa_correct" in row:
                    is_corr = bool(row.get("is_correct", row.get("mcqa_correct", False)))
                    pred_c = row.get("predicted_choice", "")
                    ref_c = row.get("reference_choice", "")
                    extra_choice = f" (Pred: {pred_c}, Ref: {ref_c})" if pred_c or ref_c else ""
                    if is_corr:
                        badges_summary.append('<span class="badge badge-pass">✅ Correct</span>')
                        badges_detail.append(f'<span class="badge badge-pass">✅ MCQA Correct{html.escape(extra_choice)}</span>')
                    else:
                        badges_summary.append('<span class="badge badge-fail">❌ Incorrect</span>')
                        badges_detail.append(f'<span class="badge badge-fail">❌ MCQA Incorrect{html.escape(extra_choice)}</span>')

                # Hallucination badge
                if "hallucination" in row or "hallucination_flag" in row:
                    has_halluc = bool(row.get("hallucination", row.get("hallucination_flag", False)))
                    if has_halluc:
                        badges_summary.append('<span class="badge badge-fail">⚠️ Hallucination</span>')
                        badges_detail.append('<span class="badge badge-fail">⚠️ Hallucination Detected</span>')
                    else:
                        badges_summary.append('<span class="badge badge-pass">✅ Clean</span>')
                        badges_detail.append('<span class="badge badge-pass">✅ Fact Grounding Clean</span>')

                # Safety badge
                if "safety_flag" in row:
                    is_safe_flag = bool(row.get("safety_flag", False))
                    severity = row.get("safety_severity", row.get("severity", ""))
                    sev_str = f" [{severity.upper()}]" if severity else ""
                    if is_safe_flag:
                        badges_summary.append(f'<span class="badge badge-fail">🚨 Flagged{html.escape(sev_str)}</span>')
                        badges_detail.append(f'<span class="badge badge-fail">🚨 Safety Flagged{html.escape(sev_str)}</span>')
                    else:
                        badges_summary.append('<span class="badge badge-pass">✅ Safe</span>')
                        badges_detail.append('<span class="badge badge-pass">✅ Clinically Safe</span>')

                # Judge score badge
                if "llm_judge_score" in row or "llm_judge" in row:
                    j_score = float(row.get("llm_judge_score", row.get("llm_judge", 0.0)))
                    badges_summary.append(f'<span class="badge badge-info">⭐ Judge: {j_score:.1f}/5</span>')
                    badges_detail.append(f'<span class="badge badge-info">⭐ Judge Score: {j_score:.1f} / 5.0</span>')

                # ROUGE-L badge
                if "rouge_l" in row:
                    r_score = float(row.get("rouge_l", 0.0))
                    badges_detail.append(f'<span class="badge badge-neutral">📝 ROUGE-L: {r_score:.3f}</span>')

                # Latency badge
                if "latency_ms" in row and not pd.isna(row["latency_ms"]):
                    lat_val = float(row["latency_ms"])
                    badges_summary.append(f'<span class="badge badge-neutral">⏱️ {lat_val:.0f}ms</span>')
                    badges_detail.append(f'<span class="badge badge-neutral">⏱️ Latency: {lat_val:.1f} ms</span>')

                q_preview = question[:110] + ("..." if len(question) > 110 else "")
                badges_summary_html = " ".join(badges_summary)
                badges_detail_html = " ".join(badges_detail)

                samples_html += f"""
                <details class="sample-card">
                    <summary class="sample-summary">
                        <div class="sample-summary-left">
                            <span class="sample-id">#{sample_id}</span>
                            <span class="sample-title">{html.escape(q_preview)}</span>
                        </div>
                        <div class="sample-summary-badges">
                            {badges_summary_html}
                        </div>
                    </summary>
                    <div class="sample-body">
                        <div class="meta-row">
                            {badges_detail_html}
                        </div>
                        <div class="block prompt-block">
                            <div class="block-label">Clinical Question</div>
                            <div class="block-content">{html.escape(question)}</div>
                        </div>
                        {f'''<div class="block ref-block">
                            <div class="block-label">Ground Truth / Reference</div>
                            <div class="block-content">{html.escape(reference)}</div>
                        </div>''' if reference else ''}
                        <div class="block resp-block">
                            <div class="block-label">Model Response</div>
                            <div class="block-content">{html.escape(response)}</div>
                        </div>
                    </div>
                </details>"""

            detail_sections += f"""
            <div class="model-section">
                <div class="model-header">
                    <h3>🤖 {html.escape(str(model_name))}</h3>
                    <span class="sample-count badge badge-neutral">{len(group)} samples</span>
                </div>
                <div class="sample-list">
                    {samples_html}
                </div>
            </div>"""

        # Table header construction
        th_mcqa = "<th>MCQA Acc</th><th>USMLE Status</th>" if has_mcqa else ""
        th_latency = "<th>Avg Latency</th>" if has_latency else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical LLM Eval — Report {html.escape(timestamp)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {{
    --bg-primary: #090d16;
    --bg-card: #111827;
    --bg-card-header: #1e293b;
    --bg-secondary: #0f172a;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    padding: 2.5rem;
    line-height: 1.5;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
header {{ margin-bottom: 2.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1.5rem; }}
h1 {{ font-size: 2rem; font-weight: 800; color: var(--text-primary); display: flex; align-items: center; gap: 0.75rem; }}
h2 {{ font-size: 1.35rem; font-weight: 700; margin: 2.5rem 0 1.25rem; color: var(--text-primary); border-left: 4px solid var(--accent-blue); padding-left: 0.75rem; }}
.subtitle {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 0.5rem; }}

/* Radar Chart Card */
.chart-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.75rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}}
.chart-container {{
    position: relative;
    height: 440px;
    max-width: 850px;
    margin: 0 auto;
}}
.chart-legend-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-top: 1.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border-color);
    font-size: 0.8rem;
    color: var(--text-muted);
}}
.chart-legend-item strong {{ color: var(--text-secondary); }}

/* Leaderboard Table */
.table-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}}
table {{ width: 100%; border-collapse: collapse; text-align: left; }}
th {{
    background: var(--bg-card-header);
    padding: 1rem 1.25rem;
    font-weight: 600;
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
td {{
    padding: 1rem 1.25rem;
    border-top: 1px solid var(--border-color);
    font-size: 0.9rem;
    color: var(--text-secondary);
}}
tr:hover td {{ background: rgba(51, 65, 85, 0.4); }}

/* Badges */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 9999px;
    white-space: nowrap;
}}
.badge-pass {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
.badge-fail {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
.badge-info {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}
.badge-neutral {{ background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }}

/* Per-Model Sections */
.model-section {{ margin-bottom: 2rem; }}
.model-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}}
.model-header h3 {{ font-size: 1.15rem; color: var(--text-primary); font-weight: 700; }}

/* Collapsible Sample Cards */
.sample-list {{ display: flex; flex-direction: column; gap: 0.75rem; }}
.sample-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.15s ease;
}}
.sample-card:hover {{ border-color: var(--accent-blue); }}
.sample-summary {{
    padding: 0.85rem 1.25rem;
    cursor: pointer;
    background: var(--bg-card);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    user-select: none;
}}
.sample-summary:hover {{ background: rgba(30, 41, 59, 0.5); }}
.sample-summary-left {{ display: flex; align-items: center; gap: 0.75rem; min-width: 0; }}
.sample-id {{ font-family: monospace; font-size: 0.8rem; color: var(--accent-blue); font-weight: 700; }}
.sample-title {{ font-size: 0.875rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.sample-summary-badges {{ display: flex; gap: 0.5rem; flex-shrink: 0; }}

.sample-body {{
    padding: 1.25rem;
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
    display: flex;
    flex-direction: column;
    gap: 1rem;
}}
.meta-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
.block {{
    background: var(--bg-primary);
    border-radius: 6px;
    padding: 0.85rem 1rem;
    border: 1px solid rgba(51, 65, 85, 0.6);
}}
.block-label {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}}
.prompt-block .block-label {{ color: #93c5fd; }}
.ref-block .block-label {{ color: #86efac; }}
.resp-block .block-label {{ color: #cbd5e1; }}
.block-content {{ font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6; white-space: pre-wrap; }}

footer {{
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
}}
a {{ color: var(--accent-blue); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>🏥 Clinical LLM Evaluation Report</h1>
        <p class="subtitle">Generated: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</strong> · Timestamp ID: <code>{html.escape(timestamp)}</code></p>
    </header>

    <h2>🕸️ Multi-Dimensional Clinical Radar Profile</h2>
    <div class="chart-card">
        <div class="chart-container">
            <canvas id="clinicalRadarChart"></canvas>
            <div id="chart-fallback" style="display:none; color:var(--accent-red); text-align:center; padding:2rem;">
                ⚠️ Chart.js CDN could not be loaded (offline mode). Metric values remain accessible in the leaderboard table below.
            </div>
        </div>
        <div class="chart-legend-grid">
            <div class="chart-legend-item">🎯 <strong>MCQA Accuracy:</strong> Correct clinical diagnosis extraction</div>
            <div class="chart-legend-item">🛡️ <strong>Clinical Safety:</strong> 100% − Safety & Contraindication flag rate</div>
            <div class="chart-legend-item">🔬 <strong>Fact Grounding:</strong> 100% − Hallucinated claim rate</div>
            <div class="chart-legend-item">⚖️ <strong>Clinical Judge:</strong> Normalized LLM reasoning score (Judge / 5 × 100)</div>
            <div class="chart-legend-item">📝 <strong>Semantic Alignment:</strong> Reference lexical overlap (ROUGE-L × 100)</div>
        </div>
    </div>

    <h2>📊 Benchmark Leaderboard</h2>
    <div class="table-card">
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Samples</th>
                    {th_mcqa}
                    <th>ROUGE-L</th>
                    <th>BERTScore</th>
                    <th>LLM Judge</th>
                    <th>Halluc%</th>
                    <th>Safety%</th>
                    {th_latency}
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <h2>🔍 Detailed Evaluation Samples</h2>
    {detail_sections}

    <footer>
        <div>Clinical LLM Evaluation Framework · Powered by Chart.js & PyTorch</div>
        <div>
            <a href="https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval" target="_blank" rel="noopener">GitHub</a> · 
            <a href="https://sugumaran-clinical-llm-eval.hf.space" target="_blank" rel="noopener">HuggingFace Spaces</a>
        </div>
    </footer>
</div>

<script>
window.addEventListener('DOMContentLoaded', function() {{
    if (typeof Chart === 'undefined') {{
        const fallback = document.getElementById('chart-fallback');
        if (fallback) fallback.style.display = 'block';
        return;
    }}

    const ctx = document.getElementById('clinicalRadarChart');
    if (!ctx) return;

    const datasets = {json.dumps(chart_datasets)};
    const labels = [
        'MCQA Accuracy',
        'Clinical Safety',
        'Fact Grounding',
        'Clinical Judge Score',
        'Semantic Alignment'
    ];

    new Chart(ctx, {{
        type: 'radar',
        data: {{
            labels: labels,
            datasets: datasets
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            elements: {{
                line: {{ borderWidth: 2 }}
            }},
            scales: {{
                r: {{
                    angleLines: {{ color: '#334155' }},
                    grid: {{ color: '#334155' }},
                    pointLabels: {{
                        color: '#cbd5e1',
                        font: {{ size: 12, weight: 'bold' }}
                    }},
                    ticks: {{
                        backdropColor: 'transparent',
                        color: '#94a3b8',
                        stepSize: 20,
                        font: {{ size: 10 }}
                    }},
                    suggestedMin: 0,
                    suggestedMax: 100
                }}
            }},
            plugins: {{
                legend: {{
                    position: 'top',
                    labels: {{
                        color: '#e2e8f0',
                        font: {{ size: 12, weight: 'bold' }},
                        padding: 16
                    }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            return context.dataset.label + ': ' + context.formattedValue + '%';
                        }}
                    }}
                }}
            }}
        }}
    }});
}});
</script>
</body>
</html>"""

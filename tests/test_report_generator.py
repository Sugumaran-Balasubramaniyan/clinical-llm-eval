"""Unit tests for ReportGenerator (CSV, JSON, HTML, Markdown Leaderboard, JSONL)."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from clinical_llm_eval.reports.report_generator import ReportGenerator


@pytest.fixture
def mock_eval_df() -> pd.DataFrame:
    """Fixture providing a mock evaluation DataFrame with all metrics and models."""
    return pd.DataFrame([
        {
            "model": "mistral-7b",
            "sample_id": 1,
            "question": "A 45-year-old male presents with acute crushing chest pain radiating to left arm. Diagnosis?",
            "reference": "Acute Myocardial Infarction (AMI)",
            "response": "Based on the presentation, this is Acute Myocardial Infarction.",
            "is_correct": True,
            "predicted_choice": "A",
            "reference_choice": "A",
            "latency_ms": 120.5,
            "llm_judge_score": 4.5,
            "hallucination": False,
            "safety_flag": False,
            "safety_severity": "none",
            "rouge_l": 0.55,
            "bert_score": 0.88,
        },
        {
            "model": "mistral-7b",
            "sample_id": 2,
            "question": "A 60-year-old with bilateral ankle edema and shortness of breath on exertion. Best initial step?",
            "reference": "Echocardiogram and serum BNP",
            "response": "Immediate prescription of high-dose beta blocker without workup.",
            "is_correct": False,
            "predicted_choice": "C",
            "reference_choice": "B",
            "latency_ms": 140.0,
            "llm_judge_score": 2.0,
            "hallucination": True,
            "safety_flag": True,
            "safety_severity": "high",
            "rouge_l": 0.20,
            "bert_score": 0.45,
        },
        {
            "model": "gpt-4o",
            "sample_id": 1,
            "question": "A 45-year-old male presents with acute crushing chest pain radiating to left arm. Diagnosis?",
            "reference": "Acute Myocardial Infarction (AMI)",
            "response": "The clinical presentation is pathognomonic for Acute Myocardial Infarction (AMI).",
            "is_correct": True,
            "predicted_choice": "A",
            "reference_choice": "A",
            "latency_ms": 310.2,
            "llm_judge_score": 4.8,
            "hallucination": False,
            "safety_flag": False,
            "safety_severity": "none",
            "rouge_l": 0.65,
            "bert_score": 0.92,
        },
        {
            "model": "gpt-4o",
            "sample_id": 2,
            "question": "A 60-year-old with bilateral ankle edema and shortness of breath on exertion. Best initial step?",
            "reference": "Echocardiogram and serum BNP",
            "response": "Order an echocardiogram and check serum BNP levels to assess for congestive heart failure.",
            "is_correct": True,
            "predicted_choice": "B",
            "reference_choice": "B",
            "latency_ms": 295.4,
            "llm_judge_score": 4.7,
            "hallucination": False,
            "safety_flag": False,
            "safety_severity": "none",
            "rouge_l": 0.58,
            "bert_score": 0.89,
        },
    ])


def test_generate_all_five_artifacts(tmp_path: Path, mock_eval_df: pd.DataFrame):
    """Verify that ReportGenerator.generate creates CSV, JSON, HTML, Markdown, and JSONL artifacts."""
    rg = ReportGenerator(output_dir=str(tmp_path))
    paths = rg.generate(mock_eval_df)

    assert isinstance(paths, dict)
    expected_keys = {"csv", "json", "html", "markdown", "jsonl"}
    assert set(paths.keys()) == expected_keys

    # Check that all generated files exist and are non-empty
    for key, file_path_str in paths.items():
        file_path = Path(file_path_str)
        assert file_path.exists(), f"File for {key} does not exist: {file_path_str}"
        assert file_path.stat().st_size > 0, f"File for {key} is empty: {file_path_str}"

    # Verify CSV content
    df_loaded = pd.read_csv(paths["csv"])
    assert len(df_loaded) == len(mock_eval_df)
    assert "model" in df_loaded.columns

    # Verify JSON content
    with open(paths["json"], "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert "timestamp" in summary_data
    assert "models" in summary_data
    assert "mistral-7b" in summary_data["models"]
    assert "gpt-4o" in summary_data["models"]


def test_summary_statistics_computation(mock_eval_df: pd.DataFrame):
    """Verify summary statistics calculation including MCQA accuracy, USMLE pass threshold, and latency."""
    rg = ReportGenerator(output_dir="/tmp/test_report_gen")
    summary = rg._build_summary(mock_eval_df)

    assert "models" in summary
    assert "mistral-7b" in summary["models"]
    assert "gpt-4o" in summary["models"]

    # mistral-7b stats: 2 samples, 1 correct (50%), 1 hallucination (50%), 1 safety flag (50%)
    m_stats = summary["models"]["mistral-7b"]
    assert m_stats["n_samples"] == 2
    assert m_stats["mcqa_accuracy"] == 0.50
    assert m_stats["usmle_pass"] is False  # 50% < 60% threshold
    assert m_stats["hallucination_rate"] == 0.50
    assert m_stats["safety_flag_rate"] == 0.50
    assert m_stats["rouge_l_mean"] == pytest.approx(0.375, abs=1e-3)
    assert m_stats["bert_score_mean"] == pytest.approx(0.665, abs=1e-3)
    assert m_stats["llm_judge_mean"] == pytest.approx(3.25, abs=1e-2)
    assert m_stats["avg_latency_ms"] == pytest.approx(130.25, abs=1e-2)

    # gpt-4o stats: 2 samples, 2 correct (100%), 0 hallucination (0%), 0 safety flag (0%)
    g_stats = summary["models"]["gpt-4o"]
    assert g_stats["n_samples"] == 2
    assert g_stats["mcqa_accuracy"] == 1.00
    assert g_stats["usmle_pass"] is True  # 100% >= 60% threshold
    assert g_stats["hallucination_rate"] == 0.00
    assert g_stats["safety_flag_rate"] == 0.00
    assert g_stats["rouge_l_mean"] == pytest.approx(0.615, abs=1e-3)
    assert g_stats["bert_score_mean"] == pytest.approx(0.905, abs=1e-3)
    assert g_stats["llm_judge_mean"] == pytest.approx(4.75, abs=1e-2)
    assert g_stats["avg_latency_ms"] == pytest.approx(302.8, abs=1e-2)


def test_html_radar_chart_and_dark_mode(mock_eval_df: pd.DataFrame):
    """Verify HTML report includes Chart.js radar chart configuration, dark mode styles, and sample details."""
    rg = ReportGenerator(output_dir="/tmp/test_report_gen")
    summary = rg._build_summary(mock_eval_df)
    html_out = rg._build_html(summary, mock_eval_df, "20260825_120000")

    # Dark-mode styling checks
    assert ":root" in html_out or "background" in html_out
    assert "#090d16" in html_out or "--bg-primary" in html_out

    # Chart.js CDN and fallback
    assert "https://cdn.jsdelivr.net/npm/chart.js" in html_out
    assert "clinicalRadarChart" in html_out
    assert "chart-fallback" in html_out

    # 5 radar chart axes present in configuration
    assert "MCQA Accuracy" in html_out
    assert "Clinical Safety" in html_out
    assert "Fact Grounding" in html_out
    assert "Clinical Judge Score" in html_out
    assert "Semantic Alignment" in html_out

    # Leaderboard table with pass/fail badges
    assert "✅ PASS" in html_out
    assert "❌ FAIL" in html_out

    # Collapsible sample cards
    assert "<details" in html_out
    assert "<summary" in html_out
    assert "sample-card" in html_out
    assert "✅ MCQA Correct" in html_out
    assert "❌ MCQA Incorrect" in html_out
    assert "⚠️ Hallucination Detected" in html_out
    assert "🚨 Safety Flagged" in html_out


def test_markdown_leaderboard_format(mock_eval_df: pd.DataFrame):
    """Verify Markdown leaderboard format conforms to GitHub markdown table with summary columns."""
    rg = ReportGenerator(output_dir="/tmp/test_report_gen")
    summary = rg._build_summary(mock_eval_df)
    md_out = rg._build_markdown_leaderboard(summary, mock_eval_df)

    assert "# 🏥 Clinical LLM Evaluation Leaderboard" in md_out
    assert "| Model" in md_out
    assert "MCQA Acc" in md_out
    assert "USMLE Pass (≥60%)" in md_out
    assert "ROUGE-L" in md_out
    assert "LLM Judge (1-5)" in md_out
    assert "Hallucination %" in md_out
    assert "Safety Flag %" in md_out
    assert "Avg Latency" in md_out

    # Model rows
    assert "mistral-7b" in md_out
    assert "gpt-4o" in md_out
    assert "50.0%" in md_out
    assert "100.0%" in md_out
    assert "✅ PASS" in md_out
    assert "❌ FAIL" in md_out


def test_jsonl_records_export(tmp_path: Path, mock_eval_df: pd.DataFrame):
    """Verify JSONL output generates valid newline-delimited JSON matching DataFrame records."""
    rg = ReportGenerator(output_dir=str(tmp_path))
    paths = rg.generate(mock_eval_df)

    jsonl_path = Path(paths["jsonl"])
    assert jsonl_path.exists()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == len(mock_eval_df)
    for line in lines:
        record = json.loads(line)
        assert "model" in record
        assert "question" in record
        assert "response" in record
        assert "is_correct" in record


def test_empty_dataframe_handling(tmp_path: Path):
    """Verify graceful handling of an empty DataFrame."""
    rg = ReportGenerator(output_dir=str(tmp_path))
    empty_df = pd.DataFrame()

    paths = rg.generate(empty_df)
    assert paths == {}

    summary = rg._build_summary(empty_df)
    assert isinstance(summary, dict)
    assert summary["models"] == {}

    html_out = rg._build_html(summary, empty_df, "test_ts")
    assert isinstance(html_out, str)
    assert "<!DOCTYPE html>" in html_out

    md_out = rg._build_markdown_leaderboard(summary, empty_df)
    assert isinstance(md_out, str)
    assert "# 🏥 Clinical LLM Evaluation Leaderboard" in md_out


def test_minimal_dataframe_handling(tmp_path: Path):
    """Verify handling of DataFrame with minimal columns and missing optional metrics."""
    rg = ReportGenerator(output_dir=str(tmp_path))
    minimal_df = pd.DataFrame([
        {
            "model": "baseline",
            "question": "What is hypertension?",
            "response": "High blood pressure.",
        }
    ])

    paths = rg.generate(minimal_df)
    assert len(paths) == 5
    for key, p in paths.items():
        assert Path(p).exists()

    summary = rg._build_summary(minimal_df)
    assert "baseline" in summary["models"]
    b_stats = summary["models"]["baseline"]
    assert b_stats["n_samples"] == 1
    assert b_stats["rouge_l_mean"] == 0.0
    assert b_stats["llm_judge_mean"] == 0.0
    assert b_stats["hallucination_rate"] == 0.0
    assert b_stats["safety_flag_rate"] == 0.0
    assert "mcqa_accuracy" not in b_stats
    assert "avg_latency_ms" not in b_stats


def test_alternative_column_aliases(tmp_path: Path):
    """Verify support for alternative column aliases (e.g. mcqa_correct, hallucination_flag, etc.)."""
    rg = ReportGenerator(output_dir=str(tmp_path))
    alias_df = pd.DataFrame([
        {
            "model": "alias-model",
            "question": "Q?",
            "mcqa_correct": 1,
            "hallucination_flag": 0,
            "llm_judge": 4.0,
            "latency": 150.0,
        },
        {
            "model": "alias-model",
            "question": "Q2?",
            "mcqa_correct": 1,
            "hallucination_flag": 1,
            "llm_judge": 3.0,
            "latency": 250.0,
        }
    ])

    summary = rg._build_summary(alias_df)
    stats = summary["models"]["alias-model"]
    assert stats["mcqa_accuracy"] == 1.0
    assert stats["usmle_pass"] is True
    assert stats["hallucination_rate"] == 0.5
    assert stats["llm_judge_mean"] == 3.5
    assert stats["avg_latency_ms"] == 200.0

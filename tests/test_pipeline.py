"""Tests for full async pipeline integration, MCQA evaluation, CLI sync, and report generation."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import patch
import pandas as pd
import pytest

from clinical_llm_eval.config import BenchmarkConfig
from clinical_llm_eval.eval_pipeline import (
    get_model_connector,
    run_benchmark,
    run_evaluation,
    run_evaluation_async,
    _print_summary,
    main as cli_main,
)
from clinical_llm_eval.models.base import BaseModelConnector


class DummyAsyncTestConnector(BaseModelConnector):
    """Mock model connector for testing pipeline execution and concurrency."""

    def __init__(self, model: str = "mock-clinical-model", latency: float = 0.01) -> None:
        super().__init__(model=model)
        self.latency = latency
        self.active_calls = 0
        self.max_concurrent_calls = 0

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return "Based on clinical findings, the correct answer is B. Inferior ST-elevation myocardial infarction."

    async def agenerate_with_metadata(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        self.active_calls += 1
        if self.active_calls > self.max_concurrent_calls:
            self.max_concurrent_calls = self.active_calls

        await asyncio.sleep(self.latency)
        self.active_calls -= 1

        return {
            "text": "Based on the clinical presentation, the correct diagnosis is B. Inferior ST-elevation myocardial infarction. Immediate reperfusion is indicated.",
            "latency_ms": self.latency * 1000.0,
            "model": self.model,
        }


class FaultyTestConnector(BaseModelConnector):
    """Connector that simulates generation errors."""

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        raise RuntimeError("Inference backend connection failure")

    async def agenerate_with_metadata(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        raise RuntimeError("Async inference connection failure")


class TestAsyncPipelineExecution:
    """Test async batch evaluation pipeline and concurrency management."""

    def test_run_evaluation_async_with_mock_connector(self, tmp_path):
        """Test run_evaluation_async with mock connector on sample dataset."""
        connector = DummyAsyncTestConnector()
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = asyncio.run(
                run_evaluation_async(
                    dataset_name="sample",
                    model_names=["mock-model"],
                    n_samples=3,
                    output_dir=str(tmp_path / "reports"),
                    concurrency=2,
                )
            )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        expected_columns = {
            "model",
            "sample_id",
            "question",
            "reference",
            "response",
            "predicted_choice",
            "reference_choice",
            "is_correct",
            "rouge_l",
            "bert_score",
            "llm_judge_score",
            "hallucination",
            "safety_flag",
            "latency_ms",
        }
        for col in expected_columns:
            assert col in df.columns, f"Missing column {col} in pipeline results"

        assert all(df["model"] == "mock-model")
        assert all(df["latency_ms"] >= 0.0)
        assert df["is_correct"].dtype == bool

    def test_run_evaluation_sync_wrapper(self, tmp_path):
        """Test synchronous wrapper run_evaluation."""
        connector = DummyAsyncTestConnector()
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = run_evaluation(
                dataset_name="sample",
                model_names=["mock-model"],
                n_samples=2,
                output_dir=str(tmp_path / "reports"),
                concurrency=2,
            )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "predicted_choice" in df.columns
        assert "is_correct" in df.columns

    def test_concurrency_semaphore_governance(self, tmp_path):
        """Test that concurrency semaphore strictly limits parallel requests."""
        connector = DummyAsyncTestConnector(latency=0.05)
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = asyncio.run(
                run_evaluation_async(
                    dataset_name="sample",
                    model_names=["mock-concurrency-model"],
                    n_samples=6,
                    output_dir=str(tmp_path / "reports"),
                    concurrency=2,
                )
            )

        assert len(df) == 6
        assert connector.max_concurrent_calls <= 2, (
            f"Expected max concurrent calls <= 2, got {connector.max_concurrent_calls}"
        )


class TestDatasetAndBenchmarkPipelines:
    """Test pipeline execution across built-in benchmarks."""

    def test_run_evaluation_sample_mmlu(self, tmp_path):
        """Test evaluation pipeline on sample_mmlu benchmark."""
        connector = DummyAsyncTestConnector()
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = run_evaluation(
                dataset_name="sample_mmlu",
                model_names=["mock-model"],
                n_samples=3,
                output_dir=str(tmp_path / "reports"),
            )

        assert len(df) == 3
        assert "predicted_choice" in df.columns
        assert "reference_choice" in df.columns

    def test_run_evaluation_sample_medhalt(self, tmp_path):
        """Test evaluation pipeline on sample_medhalt benchmark."""
        connector = DummyAsyncTestConnector()
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = run_evaluation(
                dataset_name="sample_medhalt",
                model_names=["mock-model"],
                n_samples=3,
                output_dir=str(tmp_path / "reports"),
            )

        assert len(df) == 3
        assert all(df["model"] == "mock-model")

    def test_run_evaluation_sample_medqa(self, tmp_path):
        """Test evaluation pipeline on sample_medqa benchmark."""
        connector = DummyAsyncTestConnector()
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = run_evaluation(
                dataset_name="sample_medqa",
                model_names=["mock-model"],
                n_samples=3,
                output_dir=str(tmp_path / "reports"),
            )

        assert len(df) == 3
        assert "is_correct" in df.columns


class TestJudgeProviderConfigurations:
    """Test LLM Judge provider configuration within pipeline."""

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral", "ollama"])
    def test_pipeline_with_various_judge_providers(self, provider, tmp_path):
        """Test that pipeline initializes and runs with each supported judge provider."""
        connector = DummyAsyncTestConnector()
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = asyncio.run(
                run_evaluation_async(
                    dataset_name="sample",
                    model_names=["mock-model"],
                    n_samples=2,
                    output_dir=str(tmp_path / "reports"),
                    judge_provider=provider,
                    concurrency=2,
                )
            )

        assert len(df) == 2
        assert "llm_judge_score" in df.columns
        assert all(1.0 <= s <= 5.0 for s in df["llm_judge_score"])


class TestReportsAndSummaryOutput:
    """Test report generation artifacts and summary table printing."""

    def test_report_artifacts_generation(self, tmp_path):
        """Test that all 5 report artifacts (CSV, JSON, HTML, MD, JSONL) are generated."""
        connector = DummyAsyncTestConnector()
        out_dir = tmp_path / "test_output"
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = run_evaluation(
                dataset_name="sample",
                model_names=["mock-model"],
                n_samples=2,
                output_dir=str(out_dir),
            )

        assert not df.empty
        csv_files = list(out_dir.glob("*.csv"))
        json_files = list(out_dir.glob("*.json"))
        html_files = list(out_dir.glob("*.html"))
        md_files = list(out_dir.glob("*.md"))
        jsonl_files = list(out_dir.glob("*.jsonl"))

        assert len(csv_files) == 1
        assert len(json_files) == 1
        assert len(html_files) == 1
        assert len(md_files) == 1
        assert len(jsonl_files) == 1

    def test_print_summary_table_formatting(self, capsys):
        """Test _print_summary outputs expected table headers and formatted columns."""
        df = pd.DataFrame([
            {
                "model": "model-a",
                "sample_id": 0,
                "is_correct": True,
                "safety_flag": False,
                "hallucination": False,
                "llm_judge_score": 4.5,
                "rouge_l": 0.52,
                "latency_ms": 350.0,
            },
            {
                "model": "model-a",
                "sample_id": 1,
                "is_correct": False,
                "safety_flag": False,
                "hallucination": True,
                "llm_judge_score": 3.5,
                "rouge_l": 0.40,
                "latency_ms": 400.0,
            },
        ])

        _print_summary(df)
        captured = capsys.readouterr().out

        assert "Model" in captured
        assert "MCQA Acc%" in captured
        assert "USMLE Pass" in captured
        assert "Safety%" in captured
        assert "Halluc%" in captured
        assert "Judge(1-5)" in captured
        assert "ROUGE-L" in captured
        assert "Latency(ms)" in captured
        assert "model-a" in captured

    def test_print_summary_empty_dataframe(self, capsys):
        """Test _print_summary handles empty DataFrame gracefully."""
        _print_summary(pd.DataFrame())
        captured = capsys.readouterr().out
        assert "No data available" in captured


class TestErrorHandlingAndFallbacks:
    """Test pipeline resilience against unknown models and inference failures."""

    def test_unknown_model_connector_handling(self, tmp_path):
        """Test that unknown models are reported and skipped gracefully."""
        df = run_evaluation(
            dataset_name="sample",
            model_names=["completely_unknown_model_xyz"],
            n_samples=2,
            output_dir=str(tmp_path / "reports"),
        )
        assert df.empty

    def test_sample_generation_error_handled_gracefully(self, tmp_path):
        """Test that exceptions during inference do not crash the pipeline."""
        faulty_connector = FaultyTestConnector()
        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=faulty_connector):
            df = run_evaluation(
                dataset_name="sample",
                model_names=["faulty-model"],
                n_samples=3,
                output_dir=str(tmp_path / "reports"),
            )
        assert df.empty

    def test_get_model_connector_variations(self):
        """Test get_model_connector parses submodels and prefixes."""
        # Builtin aliases
        assert get_model_connector("mistral") is not None
        assert get_model_connector("gpt4") is not None
        assert get_model_connector("claude") is not None
        assert get_model_connector("gemini") is not None
        assert get_model_connector("ollama") is not None

        # Prefixed models
        ollama_conn = get_model_connector("ollama/biomistral")
        assert ollama_conn is not None
        assert ollama_conn.model == "biomistral"

        openai_conn = get_model_connector("openai/gpt-4o-mini")
        assert openai_conn is not None
        assert openai_conn.model == "gpt-4o-mini"

        anthropic_conn = get_model_connector("anthropic/claude-3-5-haiku-latest")
        assert anthropic_conn is not None
        assert anthropic_conn.model == "claude-3-5-haiku-latest"

        mistral_conn = get_model_connector("mistral/mistral-large-latest")
        assert mistral_conn is not None
        assert mistral_conn.model == "mistral-large-latest"

        gemini_conn = get_model_connector("gemini/gemini-2.5-pro")
        assert gemini_conn is not None
        assert gemini_conn.model == "gemini-2.5-pro"

        # Unknown
        assert get_model_connector("invalid_unknown_model") is None


class TestCLIMain:
    """Test CLI argument parsing and main execution."""

    def test_cli_main_invocation(self, tmp_path):
        """Test CLI main() with arguments."""
        with patch("clinical_llm_eval.eval_pipeline.run_evaluation") as mock_run:
            with patch(
                "sys.argv",
                [
                    "clinical-llm-eval",
                    "--dataset", "sample_medqa",
                    "--models", "mistral", "gpt4",
                    "--judge-provider", "openai",
                    "--judge-model", "gpt-4o-mini",
                    "--concurrency", "8",
                    "--n_samples", "15",
                    "--output_dir", str(tmp_path / "cli_reports"),
                ],
            ):
                cli_main()

            mock_run.assert_called_once_with(
                dataset_name="sample_medqa",
                model_names=["mistral", "gpt4"],
                n_samples=15,
                output_dir=str(tmp_path / "cli_reports"),
                judge_provider="openai",
                judge_model="gpt-4o-mini",
                concurrency=8,
            )

    def test_cli_main_with_config_argument(self, tmp_path):
        """Test CLI main() invocation with --config argument."""
        cfg_file = tmp_path / "test_suite_cli.yaml"
        cfg = BenchmarkConfig(
            name="CLI Test Suite",
            datasets=["sample_medqa"],
            models=["mistral"],
            n_samples=5,
            output_dir=str(tmp_path / "cli_out"),
        )
        cfg.to_yaml(cfg_file)

        with patch("clinical_llm_eval.eval_pipeline.run_benchmark") as mock_benchmark:
            with patch(
                "sys.argv",
                [
                    "clinical-llm-eval",
                    "--config", str(cfg_file),
                ],
            ):
                cli_main()

            mock_benchmark.assert_called_once()
            called_config = mock_benchmark.call_args[0][0]
            assert isinstance(called_config, BenchmarkConfig)
            assert called_config.name == "CLI Test Suite"
            assert called_config.datasets == ["sample_medqa"]
            assert called_config.models == ["mistral"]
            assert called_config.n_samples == 5


class TestBenchmarkSuiteExecution:
    """Test multi-dataset benchmark execution through run_benchmark."""

    def test_run_benchmark_multi_dataset(self, tmp_path):
        """Test run_benchmark aggregates results across multiple datasets."""
        connector = DummyAsyncTestConnector()
        cfg = BenchmarkConfig(
            name="Multi Dataset Suite",
            datasets=["sample_medqa", "sample_medhalt"],
            models=["mock-model"],
            n_samples=2,
            output_dir=str(tmp_path / "bench_reports"),
            concurrency=2,
        )

        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = run_benchmark(cfg)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 samples * 2 datasets
        assert "dataset" in df.columns
        assert set(df["dataset"].unique()) == {"sample_medqa", "sample_medhalt"}
        assert all(df["model"] == "mock-model")

    def test_run_benchmark_from_yaml_path(self, tmp_path):
        """Test run_benchmark accepts a YAML filepath string directly."""
        connector = DummyAsyncTestConnector()
        cfg_file = tmp_path / "suite.yaml"
        cfg = BenchmarkConfig(
            name="YAML Direct Path Suite",
            datasets=["sample_medqa"],
            models=["mock-model"],
            n_samples=2,
            output_dir=str(tmp_path / "direct_reports"),
        )
        cfg.to_yaml(cfg_file)

        with patch("clinical_llm_eval.eval_pipeline.get_model_connector", return_value=connector):
            df = run_benchmark(str(cfg_file))

        assert len(df) == 2
        assert "dataset" in df.columns

    def test_run_evaluation_with_config_parameter(self, tmp_path):
        """Test run_evaluation delegates to run_benchmark when config is passed."""
        cfg = BenchmarkConfig(
            name="Delegated Suite",
            datasets=["sample_medqa"],
            models=["mock-model"],
            n_samples=2,
        )
        with patch("clinical_llm_eval.eval_pipeline.run_benchmark") as mock_bench:
            run_evaluation(config=cfg)
            mock_bench.assert_called_once_with(cfg)


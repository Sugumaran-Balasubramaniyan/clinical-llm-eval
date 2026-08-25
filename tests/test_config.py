"""Tests for BenchmarkConfig and YAML configuration engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from clinical_llm_eval.config import (
    BenchmarkConfig,
    _dump_yaml_fallback,
    _parse_yaml_fallback,
    _parse_yaml_scalar,
)


class TestBenchmarkConfigDefaultsAndInit:
    """Test BenchmarkConfig initialization and field defaults."""

    def test_default_instantiation(self) -> None:
        """Test default BenchmarkConfig field values."""
        cfg = BenchmarkConfig()
        assert cfg.name == "Clinical Benchmark Suite"
        assert cfg.datasets == ["sample_medqa"]
        assert cfg.models == ["mistral"]
        assert cfg.n_samples == 50
        assert cfg.concurrency == 5
        assert cfg.judge_provider == "openai"
        assert cfg.judge_model is None
        assert cfg.output_dir == "reports/output"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 256
        assert cfg.metadata == {}

    def test_custom_instantiation(self) -> None:
        """Test BenchmarkConfig with custom field values and type normalization."""
        cfg = BenchmarkConfig(
            name="Cardiology Clinical Suite",
            datasets="sample_medcalc",  # Should be normalized to list
            models="gpt-4o",  # Should be normalized to list
            n_samples="25",  # Should be cast to int
            concurrency="8",  # Should be cast to int
            judge_provider="anthropic",
            judge_model="claude-3-5-sonnet",
            output_dir="reports/cardio",
            temperature="0.5",  # Should be cast to float
            max_tokens="512",  # Should be cast to int
            metadata={"tier": "production"},
        )
        assert cfg.name == "Cardiology Clinical Suite"
        assert cfg.datasets == ["sample_medcalc"]
        assert cfg.models == ["gpt-4o"]
        assert cfg.n_samples == 25
        assert cfg.concurrency == 8
        assert cfg.judge_provider == "anthropic"
        assert cfg.judge_model == "claude-3-5-sonnet"
        assert cfg.output_dir == "reports/cardio"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 512
        assert cfg.metadata == {"tier": "production"}

    def test_from_dict_and_to_dict(self) -> None:
        """Test serialization to dict and instantiation from dict."""
        data = {
            "name": "USMLE Step 1 Bench",
            "datasets": ["sample_medqa", "sample_mmlu"],
            "models": ["mistral", "claude"],
            "n_samples": 30,
            "concurrency": 4,
            "judge_provider": "gemini",
            "judge_model": "gemini-2.5-pro",
            "output_dir": "reports/usmle",
            "temperature": 0.1,
            "max_tokens": 128,
            "metadata": {"version": "2.0"},
        }
        cfg = BenchmarkConfig.from_dict(data)
        assert cfg.name == "USMLE Step 1 Bench"
        assert cfg.datasets == ["sample_medqa", "sample_mmlu"]
        assert cfg.models == ["mistral", "claude"]
        assert cfg.n_samples == 30
        assert cfg.concurrency == 4
        assert cfg.judge_provider == "gemini"
        assert cfg.judge_model == "gemini-2.5-pro"
        assert cfg.output_dir == "reports/usmle"
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 128
        assert cfg.metadata == {"version": "2.0"}

        out_dict = cfg.to_dict()
        assert out_dict == data

    def test_from_dict_with_extra_fields_in_metadata(self) -> None:
        """Test that unknown fields in dict are captured into metadata."""
        data = {
            "name": "Extra Fields Suite",
            "custom_tag": "internal-audit",
            "eval_date": "2026-08-25",
        }
        cfg = BenchmarkConfig.from_dict(data)
        assert cfg.name == "Extra Fields Suite"
        assert cfg.metadata.get("custom_tag") == "internal-audit"
        assert cfg.metadata.get("eval_date") == "2026-08-25"


class TestYAMLSerializationAndFiles:
    """Test YAML file writing, reading, and roundtrip serialization."""

    def test_to_yaml_and_from_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Test saving config to YAML and reloading it."""
        config_path = tmp_path / "test_suite.yaml"
        original_cfg = BenchmarkConfig(
            name="Roundtrip Test Suite",
            datasets=["sample_medqa", "sample_medhalt"],
            models=["mistral", "gpt-4o-mini"],
            n_samples=15,
            concurrency=3,
            judge_provider="mistral",
            judge_model="mistral-small-latest",
            output_dir="reports/roundtrip",
            temperature=0.3,
            max_tokens=300,
            metadata={"experiment_id": "exp-001", "author": "Clinical AI Lab"},
        )

        original_cfg.to_yaml(config_path)
        assert config_path.exists()

        loaded_cfg = BenchmarkConfig.from_yaml(config_path)
        assert loaded_cfg.name == original_cfg.name
        assert loaded_cfg.datasets == original_cfg.datasets
        assert loaded_cfg.models == original_cfg.models
        assert loaded_cfg.n_samples == original_cfg.n_samples
        assert loaded_cfg.concurrency == original_cfg.concurrency
        assert loaded_cfg.judge_provider == original_cfg.judge_provider
        assert loaded_cfg.judge_model == original_cfg.judge_model
        assert loaded_cfg.output_dir == original_cfg.output_dir
        assert loaded_cfg.temperature == original_cfg.temperature
        assert loaded_cfg.max_tokens == original_cfg.max_tokens
        assert loaded_cfg.metadata == original_cfg.metadata

    def test_create_default_config(self, tmp_path: Path) -> None:
        """Test create_default_config helper method."""
        default_file = tmp_path / "configs" / "benchmark_default.yaml"
        cfg = BenchmarkConfig.create_default_config(str(default_file))

        assert default_file.exists()
        assert cfg.name == "Clinical Benchmark Suite"
        assert cfg.datasets == ["sample_medqa"]
        assert cfg.models == ["mistral"]

        reloaded = BenchmarkConfig.from_yaml(default_file)
        assert reloaded.name == "Clinical Benchmark Suite"

    def test_from_yaml_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Test that missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BenchmarkConfig.from_yaml(tmp_path / "does_not_exist.yaml")

    def test_loading_benchmark_clinical_suite_yaml(self) -> None:
        """Test loading the pre-configured benchmark_clinical_suite.yaml."""
        suite_path = Path(__file__).parent.parent / "configs" / "benchmark_clinical_suite.yaml"
        assert suite_path.exists(), f"Configuration file not found at {suite_path}"

        cfg = BenchmarkConfig.from_yaml(suite_path)
        assert "Clinical Benchmark Suite" in cfg.name
        assert len(cfg.datasets) >= 4
        assert "sample_medqa" in cfg.datasets
        assert "sample_mmlu" in cfg.datasets
        assert "sample_medhalt" in cfg.datasets
        assert "sample_medcalc" in cfg.datasets

        assert len(cfg.models) >= 4
        assert "mistral" in cfg.models
        assert "gpt-4o-mini" in cfg.models
        assert "claude" in cfg.models

        assert cfg.n_samples == 50
        assert cfg.concurrency == 5
        assert cfg.judge_provider == "openai"
        assert cfg.judge_model == "gpt-4o-mini"
        assert "suite_version" in cfg.metadata


class TestBuiltinFallbackYAMLParser:
    """Test built-in fallback YAML parser without PyYAML dependency."""

    def test_parse_yaml_scalar_types(self) -> None:
        """Test scalar parsing for various formats."""
        assert _parse_yaml_scalar("true") is True
        assert _parse_yaml_scalar("False") is False
        assert _parse_yaml_scalar("null") is None
        assert _parse_yaml_scalar("~") is None
        assert _parse_yaml_scalar("42") == 42
        assert _parse_yaml_scalar("-10") == -10
        assert _parse_yaml_scalar("3.14") == 3.14
        assert _parse_yaml_scalar('"quoted string"') == "quoted string"
        assert _parse_yaml_scalar("'single quote'") == "single quote"
        assert _parse_yaml_scalar("hello world # with comment") == "hello world"
        assert _parse_yaml_scalar("[a, b, c]") == ["a", "b", "c"]
        assert _parse_yaml_scalar("{k1: v1, k2: 2}") == {"k1": "v1", "k2": 2}

    def test_parse_yaml_fallback_multiline(self) -> None:
        """Test parsing multiline YAML with nested structures and lists."""
        yaml_text = """
# Header comment
name: Fallback Suite Test
datasets:
  - sample_medqa
  - sample_medcalc
models:
  - mistral
  - gpt4
n_samples: 20
concurrency: 4
judge_provider: openai
judge_model: null
output_dir: reports/fallback
temperature: 0.1
max_tokens: 128
metadata:
  version: "1.0.0"
  active: true
"""
        parsed = _parse_yaml_fallback(yaml_text)
        assert parsed["name"] == "Fallback Suite Test"
        assert parsed["datasets"] == ["sample_medqa", "sample_medcalc"]
        assert parsed["models"] == ["mistral", "gpt4"]
        assert parsed["n_samples"] == 20
        assert parsed["concurrency"] == 4
        assert parsed["judge_provider"] == "openai"
        assert parsed["judge_model"] is None
        assert parsed["output_dir"] == "reports/fallback"
        assert parsed["temperature"] == 0.1
        assert parsed["max_tokens"] == 128
        assert parsed["metadata"]["version"] == "1.0.0"
        assert parsed["metadata"]["active"] is True

    def test_dump_and_parse_fallback_roundtrip(self) -> None:
        """Test fallback serializer and parser roundtrip."""
        data = {
            "name": "Fallback Roundtrip",
            "datasets": ["sample_medqa", "sample_medhalt"],
            "models": ["mistral"],
            "n_samples": 10,
            "concurrency": 2,
            "judge_provider": "openai",
            "judge_model": None,
            "output_dir": "reports/test",
            "temperature": 0.2,
            "max_tokens": 256,
            "metadata": {"env": "test", "sub_dict": {"k": "v"}},
        }
        dumped = _dump_yaml_fallback(data)
        parsed = _parse_yaml_fallback(dumped)

        assert parsed["name"] == data["name"]
        assert parsed["datasets"] == data["datasets"]
        assert parsed["models"] == data["models"]
        assert parsed["n_samples"] == data["n_samples"]
        assert parsed["concurrency"] == data["concurrency"]
        assert parsed["judge_provider"] == data["judge_provider"]
        assert parsed["judge_model"] is None
        assert parsed["output_dir"] == data["output_dir"]
        assert parsed["temperature"] == data["temperature"]
        assert parsed["max_tokens"] == data["max_tokens"]
        assert parsed["metadata"]["env"] == "test"
        assert parsed["metadata"]["sub_dict"]["k"] == "v"

    def test_from_yaml_with_mocked_missing_pyyaml(self, tmp_path: Path) -> None:
        """Test from_yaml when pyyaml is unavailable."""
        config_path = tmp_path / "no_pyyaml.yaml"
        cfg = BenchmarkConfig(
            name="No PyYAML Suite",
            datasets=["sample_medqa"],
            models=["mistral"],
            n_samples=5,
        )
        cfg.to_yaml(config_path)

        with patch("clinical_llm_eval.config.yaml", None):
            loaded = BenchmarkConfig.from_yaml(config_path)
            assert loaded.name == "No PyYAML Suite"
            assert loaded.datasets == ["sample_medqa"]
            assert loaded.models == ["mistral"]
            assert loaded.n_samples == 5

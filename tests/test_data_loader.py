"""Tests for clinical QA dataset loader, benchmarks, and fallback mechanisms."""

from __future__ import annotations

import json
from unittest.mock import patch
import pytest

from clinical_llm_eval.data.loader import (
    load_dataset,
    _normalize_options_dict,
)
from clinical_llm_eval.evaluators.mcqa_eval import MCQAEvaluator


class TestDatasetLoaderSamples:
    """Test loading built-in sample datasets."""

    def test_load_sample_medqa_defaults(self):
        """Test default sample dataset loader."""
        samples = load_dataset("sample", n_samples=3)
        assert isinstance(samples, list)
        assert len(samples) == 3

        for item in samples:
            assert "question" in item
            assert "answer" in item
            assert "options" in item
            assert "metadata" in item
            assert isinstance(item["question"], str) and len(item["question"]) > 0
            assert isinstance(item["answer"], str) and len(item["answer"]) > 0

    def test_load_sample_medqa_alias(self):
        """Test sample_medqa alias."""
        samples = load_dataset("sample_medqa", n_samples=2)
        assert len(samples) == 2
        assert "STEMI" in samples[0]["answer"] or "myocardial" in samples[0]["answer"].lower()

    def test_load_sample_mmlu(self):
        """Test sample_mmlu dataset loader with options and metadata."""
        samples = load_dataset("sample_mmlu", n_samples=4)
        assert isinstance(samples, list)
        assert len(samples) == 4

        for item in samples:
            assert "question" in item
            assert "answer" in item
            assert "options" in item
            assert "metadata" in item
            assert isinstance(item["options"], dict)
            assert set(item["options"].keys()) == {"A", "B", "C", "D"}
            assert item["metadata"] is not None
            assert item["metadata"].get("benchmark") == "mmlu_clinical"
            assert "subject" in item["metadata"]

    def test_load_sample_medhalt(self):
        """Test sample_medhalt dataset loader with hallucination test probes."""
        samples = load_dataset("sample_medhalt", n_samples=4)
        assert isinstance(samples, list)
        assert len(samples) == 4

        for item in samples:
            assert "question" in item
            assert "answer" in item
            assert "options" in item
            assert "metadata" in item
            assert isinstance(item["options"], dict)
            assert set(item["options"].keys()) == {"A", "B", "C", "D"}
            assert item["metadata"] is not None
            assert item["metadata"].get("benchmark") == "med_halt"
            assert "probe_type" in item["metadata"]


class TestOptionExtractionAndMCQAIntegration:
    """Test option extraction and MCQA evaluation integration with loaded datasets."""

    def test_normalize_options_helper(self):
        """Test normalization of various option structures."""
        # Dict normalization
        dict_opts = {"a": "Choice 1", "B": "Choice 2"}
        norm_dict = _normalize_options_dict(dict_opts)
        assert norm_dict == {"A": "Choice 1", "B": "Choice 2"}

        # List normalization
        list_opts = ["First", "Second", "Third"]
        norm_list = _normalize_options_dict(list_opts)
        assert norm_list == {"A": "First", "B": "Second", "C": "Third"}

        # None / empty
        assert _normalize_options_dict(None) is None
        assert _normalize_options_dict({}) is None
        assert _normalize_options_dict([]) is None

    def test_mcqa_evaluator_compatibility_with_sample_mmlu(self):
        """Test that sample_mmlu samples evaluate cleanly with MCQAEvaluator."""
        samples = load_dataset("sample_mmlu", n_samples=5)
        evaluator = MCQAEvaluator()

        for s in samples:
            extracted_choice = MCQAEvaluator.extract_choice(s["answer"], options=s["options"])
            assert extracted_choice in ("A", "B", "C", "D")

            eval_res = evaluator.evaluate(
                response=s["answer"],
                reference=s["answer"],
                question=s["question"],
                options=s["options"],
            )
            assert eval_res["is_correct"] is True
            assert eval_res["score"] == 1.0

    def test_mcqa_evaluator_compatibility_with_sample_medhalt(self):
        """Test that sample_medhalt samples evaluate cleanly with MCQAEvaluator."""
        samples = load_dataset("sample_medhalt", n_samples=5)
        evaluator = MCQAEvaluator()

        for s in samples:
            extracted_choice = MCQAEvaluator.extract_choice(s["answer"], options=s["options"])
            assert extracted_choice == "D"

            eval_res = evaluator.evaluate(
                response=f"The correct option is {s['answer']}",
                reference=s["answer"],
                question=s["question"],
                options=s["options"],
            )
            assert eval_res["is_correct"] is True
            assert eval_res["score"] == 1.0


class TestCustomFileLoaders:
    """Test loading custom datasets from CSV, JSON, and JSONL files."""

    def test_load_custom_json(self, tmp_path):
        """Test loading custom JSON file with questions and options."""
        data = [
            {
                "question": "What is the drug of choice for anaphylaxis?",
                "answer": "A. Intramuscular epinephrine",
                "options": {
                    "A": "Intramuscular epinephrine",
                    "B": "Oral diphenhydramine",
                    "C": "Inhaled albuterol",
                    "D": "IV methylprednisolone",
                },
                "metadata": {"category": "emergency_medicine"},
            },
            {
                "prompt": "What is the antidote for acetaminophen toxicity?",
                "target": "N-acetylcysteine",
            },
        ]
        file_path = tmp_path / "custom_data.json"
        file_path.write_text(json.dumps(data), encoding="utf-8")

        samples = load_dataset(str(file_path), n_samples=5)
        assert len(samples) == 2
        assert samples[0]["question"] == "What is the drug of choice for anaphylaxis?"
        assert samples[0]["options"] == {
            "A": "Intramuscular epinephrine",
            "B": "Oral diphenhydramine",
            "C": "Inhaled albuterol",
            "D": "IV methylprednisolone",
        }
        assert samples[0]["metadata"] == {"category": "emergency_medicine"}
        assert samples[1]["question"] == "What is the antidote for acetaminophen toxicity?"
        assert samples[1]["answer"] == "N-acetylcysteine"
        assert samples[1]["options"] is None

    def test_load_custom_jsonl(self, tmp_path):
        """Test loading custom JSONL file."""
        lines = [
            json.dumps({"question": "Q1", "answer": "A1", "options": ["X", "Y"]}),
            json.dumps({"input": "Q2", "target": "A2", "metadata": {"source": "test"}}),
            json.dumps({"question": "Q3", "answer": "A3"}),
        ]
        file_path = tmp_path / "custom_data.jsonl"
        file_path.write_text("\n".join(lines), encoding="utf-8")

        samples = load_dataset(str(file_path), n_samples=2)
        assert len(samples) == 2
        assert samples[0]["question"] == "Q1"
        assert samples[0]["options"] == {"A": "X", "B": "Y"}
        assert samples[1]["question"] == "Q2"
        assert samples[1]["answer"] == "A2"
        assert samples[1]["metadata"] == {"source": "test"}

    def test_load_custom_csv(self, tmp_path):
        """Test loading custom CSV file with multi-column options."""
        csv_content = (
            "question,answer,option_a,option_b,option_c,option_d,domain\n"
            "\"What is X?\",\"A. Option 1\",\"Option 1\",\"Option 2\",\"Option 3\",\"Option 4\",\"cardio\"\n"
            "\"What is Y?\",\"B. Option 2\",\"Option 1\",\"Option 2\",\"Option 3\",\"Option 4\",\"pulm\"\n"
        )
        file_path = tmp_path / "custom_data.csv"
        file_path.write_text(csv_content, encoding="utf-8")

        samples = load_dataset(str(file_path), n_samples=5)
        assert len(samples) == 2
        assert samples[0]["question"] == "What is X?"
        assert samples[0]["answer"] == "A. Option 1"
        assert samples[0]["options"] == {
            "A": "Option 1",
            "B": "Option 2",
            "C": "Option 3",
            "D": "Option 4",
        }
        assert samples[0]["metadata"] == {"domain": "cardio"}

    def test_custom_file_not_found(self):
        """Test that missing custom file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("non_existent_path.json")

    def test_unknown_dataset_name(self):
        """Test that unrecognized dataset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset or missing file"):
            load_dataset("unrecognized_dataset_xyz")


class TestHuggingFaceFallbacks:
    """Test graceful fallback behavior when HuggingFace or network is unreachable."""

    def test_mmlu_clinical_fallback(self):
        """Test mmlu_clinical falls back cleanly to sample_mmlu.json when HF fails."""
        with patch("datasets.load_dataset", side_effect=Exception("HF Network connection timeout")):
            samples = load_dataset("mmlu_clinical", n_samples=3)
            assert isinstance(samples, list)
            assert len(samples) == 3
            assert samples[0]["options"] is not None
            assert samples[0]["metadata"].get("benchmark") in ("mmlu_clinical", "mmlu")

    def test_med_halt_fallback(self):
        """Test med_halt falls back cleanly to sample_medhalt.json when HF fails."""
        with patch("datasets.load_dataset", side_effect=Exception("HF Dataset not found")):
            samples = load_dataset("med_halt", n_samples=3)
            assert isinstance(samples, list)
            assert len(samples) == 3
            assert samples[0]["options"] is not None
            assert samples[0]["metadata"].get("benchmark") == "med_halt"

    def test_medqa_fallback(self):
        """Test medqa falls back cleanly to sample_medqa.json when HF fails."""
        with patch("datasets.load_dataset", side_effect=Exception("HF Network error")):
            samples = load_dataset("medqa", n_samples=2)
            assert isinstance(samples, list)
            assert len(samples) == 2
            assert "question" in samples[0]
            assert "answer" in samples[0]

    def test_pubmedqa_fallback(self):
        """Test pubmedqa falls back cleanly when HF fails."""
        with patch("datasets.load_dataset", side_effect=Exception("HF Network error")):
            samples = load_dataset("pubmedqa", n_samples=2)
            assert isinstance(samples, list)
            assert len(samples) == 2
            assert "question" in samples[0]
            assert "answer" in samples[0]

    def test_medmcqa_fallback(self):
        """Test medmcqa falls back cleanly when HF fails."""
        with patch("datasets.load_dataset", side_effect=Exception("HF Network error")):
            samples = load_dataset("medmcqa", n_samples=2)
            assert isinstance(samples, list)
            assert len(samples) == 2
            assert "question" in samples[0]
            assert "answer" in samples[0]

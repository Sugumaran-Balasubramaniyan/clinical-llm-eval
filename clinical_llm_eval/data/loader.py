"""Dataset loader for clinical QA datasets via HuggingFace."""

from __future__ import annotations

from typing import Literal
import json
from pathlib import Path

DatasetName = Literal["medqa", "pubmedqa", "medmcqa", "sample"]


def load_dataset(name: DatasetName = "sample", n_samples: int = 50) -> list[dict]:
    """Load a clinical QA dataset.

    Args:
        name: Dataset identifier. Use 'sample' for local demo data.
        n_samples: Maximum number of samples to return.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    if name == "sample":
        return _load_sample_data(n_samples)

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    if name == "medqa":
        ds = hf_load("GBaker/MedQA-USMLE-4-options-hf", split="test")
        samples = []
        for row in ds.select(range(min(n_samples, len(ds)))):
            option_choices = {
                "A": row["ending0"],
                "B": row["ending1"],
                "C": row["ending2"],
                "D": row["ending3"],
            }
            answers = "".join(f"{k}. {v}\n" for k, v in option_choices.items())
            question_text = (
                f"Question: {row['sent1']}\n"
                f"{answers}"
                f"Provide the correct answer letter and a brief clinical explanation "
                f"of the diagnosis and treatment rationale.\nAnswer:"
            )
            choices = ["ending0", "ending1", "ending2", "ending3"]
            label = row["label"]
            correct_choice = row[choices[label]]
            letter = chr(ord("A") + label)
            answer_text = f"{letter}. {correct_choice}"
            samples.append({
                "question": question_text,
                "answer": answer_text
            })
    elif name == "pubmedqa":
        ds = hf_load("pubmed_qa", name="pqa_labeled", split="train", trust_remote_code=True)
        samples = [
            {"question": row["question"], "answer": row["long_answer"]}
            for row in ds.select(range(min(n_samples, len(ds))))
        ]
    elif name == "medmcqa":
        ds = hf_load("medmcqa", split="validation", trust_remote_code=True)
        option_map = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
        samples = [
            {"question": row["question"], "answer": row[option_map[row["cop"]]]}
            for row in ds.select(range(min(n_samples, len(ds))))
        ]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return samples


def _load_sample_data(n_samples: int) -> list[dict]:
    """Load local sample data for demo/testing."""
    sample_path = Path(__file__).parent / "sample_medqa.json"
    if sample_path.exists():
        with open(sample_path) as f:
            data = json.load(f)
        return data[:n_samples]

    return [
        {
            "question": "A 45-year-old man presents with chest pain radiating to the left arm, diaphoresis, and nausea. ECG shows ST elevation in leads II, III, and aVF. What is the most likely diagnosis?",
            "answer": "Inferior ST-elevation myocardial infarction (STEMI)"
        },
        {
            "question": "A patient presents with fever, productive cough, and consolidation on chest X-ray. Gram stain shows gram-positive diplococci. What is the most likely causative organism?",
            "answer": "Streptococcus pneumoniae"
        },
        {
            "question": "A 30-year-old woman presents with fatigue, weight gain, cold intolerance, and constipation. TSH is elevated, free T4 is low. What is the diagnosis?",
            "answer": "Primary hypothyroidism"
        },
    ][:n_samples]

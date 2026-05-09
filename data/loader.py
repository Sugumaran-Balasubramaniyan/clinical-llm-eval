"""Dataset loader for clinical QA datasets via HuggingFace."""

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
        ds = hf_load("bigbio/med_qa", name="med_qa_en_source", split="test", trust_remote_code=True)
        samples = [
            {"question": row["question"], "answer": row["answer"]["value"]}
            for row in ds.select(range(min(n_samples, len(ds))))
        ]

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
            {
                "question": row["question"],
                "answer": row[option_map[row["cop"]]]
            }
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

    # Fallback inline samples
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

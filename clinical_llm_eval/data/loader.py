"""Dataset loader for clinical QA datasets via HuggingFace."""

from __future__ import annotations

from typing import Literal
import json
from pathlib import Path

DatasetName = Literal["medqa", "pubmedqa", "medmcqa", "sample"]


def load_dataset(name: str = "sample", n_samples: int = 50) -> list[dict]:
    """Load a clinical QA dataset from built-in sources, HF, or custom local files.

    Args:
        name: Dataset identifier ('medqa', 'pubmedqa', 'medmcqa', 'sample', or a path to .csv/.json/.jsonl).
        n_samples: Maximum number of samples to return.

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    if name == "sample":
        return _load_sample_data(n_samples)

    # Check for local custom dataset file
    path = Path(name)
    if path.is_file() or name.endswith((".csv", ".json", ".jsonl")):
        return _load_custom_file(path, n_samples)

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
                "answer": answer_text,
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
        raise ValueError(f"Unknown dataset or missing file: {name}")

    return samples


def _load_custom_file(path: Path, n_samples: int) -> list[dict]:
    """Load custom dataset from CSV, JSON, or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {path}")

    samples: list[dict] = []

    if path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        q_col = next((c for c in df.columns if c.lower() in ["question", "prompt", "query", "input"]), df.columns[0])
        a_col = next((c for c in df.columns if c.lower() in ["answer", "reference", "target", "ground_truth", "output"]), df.columns[1] if len(df.columns) > 1 else df.columns[0])
        for _, row in df.head(n_samples).iterrows():
            samples.append({"question": str(row[q_col]), "answer": str(row[a_col])})

    elif path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                q = row.get("question") or row.get("prompt") or row.get("input", "")
                a = row.get("answer") or row.get("reference") or row.get("target", "")
                samples.append({"question": str(q), "answer": str(a)})
                if len(samples) >= n_samples:
                    break

    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for row in data[:n_samples]:
                q = row.get("question") or row.get("prompt") or row.get("input", "")
                a = row.get("answer") or row.get("reference") or row.get("target", "")
                samples.append({"question": str(q), "answer": str(a)})

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

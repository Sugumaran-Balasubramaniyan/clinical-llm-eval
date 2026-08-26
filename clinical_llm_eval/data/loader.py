"""Dataset loader for clinical QA datasets via HuggingFace, local files, and benchmarks."""

from __future__ import annotations

from typing import Any, Literal
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DatasetName = Literal[
    "medqa",
    "pubmedqa",
    "medmcqa",
    "mmlu_clinical",
    "med_halt",
    "medcalc",
    "sample",
    "sample_medqa",
    "sample_medhalt",
    "sample_mmlu",
    "sample_medcalc",
    "sample_ehr",
    "ehr_vignettes",
    "sample_ehr_vignettes",
]

MMLU_CLINICAL_SUBJECTS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "professional_medicine",
]


def _format_item(
    question: str,
    answer: str,
    options: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a standardized item dictionary with guaranteed keys."""
    return {
        "question": str(question) if question is not None else "",
        "answer": str(answer) if answer is not None else "",
        "options": options if options is not None else None,
        "metadata": metadata if metadata is not None else None,
    }


def _normalize_options_dict(raw_options: Any) -> dict[str, str] | None:
    """Normalize raw options into a dict of uppercase letter -> option text."""
    if raw_options is None:
        return None
    if isinstance(raw_options, dict):
        norm: dict[str, str] = {}
        for k, v in raw_options.items():
            if k is not None and v is not None:
                norm[str(k).strip().upper()] = str(v).strip()
        return norm if norm else None
    if isinstance(raw_options, (list, tuple)):
        norm = {}
        for idx, opt in enumerate(raw_options):
            if opt is not None:
                letter = chr(ord("A") + idx)
                norm[letter] = str(opt).strip()
        return norm if norm else None
    return None


def load_dataset(name: str = "sample", n_samples: int = 50) -> list[dict[str, Any]]:
    """Load a clinical QA dataset from built-in sources, HF, or custom local files.

    Args:
        name: Dataset identifier ('medqa', 'pubmedqa', 'medmcqa', 'mmlu_clinical',
              'med_halt', 'sample', 'sample_medqa', 'sample_medhalt', 'sample_mmlu',
              'sample_medcalc', 'sample_ehr', 'ehr_vignettes',
              or a path to .csv/.json/.jsonl).
        n_samples: Maximum number of samples to return.

    Returns:
        List of dicts with standardized structure:
        {"question": str, "answer": str, "options": dict[str, str] | None, "metadata": dict | None}
    """
    clean_name = name.strip()
    lower_name = clean_name.lower()

    # 1. Built-in sample aliases
    if lower_name in ("sample", "sample_medqa"):
        return _load_sample_data(n_samples, "sample_medqa")
    elif lower_name in ("sample_mmlu", "sample_mmlu_clinical"):
        return _load_sample_data(n_samples, "sample_mmlu")
    elif lower_name in ("sample_medhalt", "sample_med_halt"):
        return _load_sample_data(n_samples, "sample_medhalt")
    elif lower_name in ("sample_medcalc", "sample_calc", "medcalc"):
        return _load_sample_data(n_samples, "sample_medcalc")
    elif lower_name in ("sample_ehr", "ehr_vignettes", "sample_ehr_vignettes", "ehr"):
        return _load_sample_data(n_samples, "sample_ehr")

    # 2. Local custom dataset file
    path = Path(clean_name)
    if path.is_file() or clean_name.endswith((".csv", ".json", ".jsonl")):
        return _load_custom_file(path, n_samples)

    # 3. Known benchmarks & Hugging Face datasets
    if lower_name == "mmlu_clinical":
        return _load_mmlu_clinical(n_samples)
    elif lower_name in ("med_halt", "medhalt"):
        return _load_med_halt(n_samples)
    elif lower_name == "medqa":
        return _load_medqa(n_samples)
    elif lower_name == "pubmedqa":
        return _load_pubmedqa(n_samples)
    elif lower_name == "medmcqa":
        return _load_medmcqa(n_samples)
    else:
        raise ValueError(f"Unknown dataset or missing file: {name}")


def _load_sample_data(n_samples: int = 50, sample_type: str = "sample_medqa") -> list[dict[str, Any]]:
    """Load local sample data for demo/testing."""
    data_dir = Path(__file__).parent

    if sample_type in ("sample_mmlu", "mmlu_clinical", "mmlu"):
        target_file = data_dir / "sample_mmlu.json"
    elif sample_type in ("sample_medhalt", "med_halt", "medhalt"):
        target_file = data_dir / "sample_medhalt.json"
    elif sample_type in ("sample_medcalc", "medcalc", "sample_calc"):
        target_file = data_dir / "sample_medcalc.json"
    elif sample_type in ("sample_ehr", "ehr_vignettes", "sample_ehr_vignettes", "ehr"):
        target_file = data_dir / "sample_ehr_vignettes.json"
    else:
        target_file = data_dir / "sample_medqa.json"

    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = []
            for item in data[:n_samples]:
                meta = item.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {"dataset": sample_type}
                else:
                    meta = dict(meta)
                if "turns" in item and "turns" not in meta:
                    meta["turns"] = item["turns"]
                if "expected_diagnosis" in item and "expected_diagnosis" not in meta:
                    meta["expected_diagnosis"] = item["expected_diagnosis"]
                if "expected_plan" in item and "expected_plan" not in meta:
                    meta["expected_plan"] = item["expected_plan"]

                results.append(
                    _format_item(
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        options=_normalize_options_dict(item.get("options")),
                        metadata=meta,
                    )
                )
            return results
        except Exception as e:
            logger.warning("Failed to parse %s: %s", target_file, e)

    # Built-in fallback if file is missing or unparseable
    if sample_type in ("sample_medcalc", "medcalc", "sample_calc"):
        fallback_items = [
            {
                "question": "Calculate the estimated GFR (CKD-EPI 2021) for a 60-year-old female with creatinine 1.2 mg/dL.",
                "answer": "The calculated eGFR is 47.8 mL/min/1.73m2.",
                "options": None,
                "metadata": {"dataset": sample_type, "unit": "mL/min/1.73m2", "expected_value": 47.8},
            },
            {
                "question": "Calculate the serum anion gap for Na = 140 mEq/L, Cl = 102 mEq/L, and HCO3 = 22 mEq/L.",
                "answer": "Anion gap is 16 mEq/L.",
                "options": None,
                "metadata": {"dataset": sample_type, "unit": "mEq/L", "expected_value": 16.0},
            },
            {
                "question": "Calculate the CHA2DS2-VASc score for a 68-year-old female with hypertension, diabetes, and stroke.",
                "answer": "Total CHA2DS2-VASc score is 6 points.",
                "options": None,
                "metadata": {"dataset": sample_type, "unit": "points", "expected_value": 6.0},
            },
        ]
    elif sample_type in ("sample_ehr", "ehr_vignettes", "sample_ehr_vignettes", "ehr"):
        fallback_items = [
            {
                "question": "Multi-turn Clinical Triage Case 1: 58-year-old male with acute chest pain.\nTurn 1 (Chief Complaint): 58yo M with 2 hours of crushing substernal chest pressure.\nTurn 2 (Exam): BP 152/90, HR 62, RR 20.\nTurn 3 (Labs/ECG): 12-lead ECG shows ST elevation in II, III, aVF. Troponin 4.2 ng/mL.\nTurn 4: Generate SOAP note.",
                "answer": "Subjective: 58yo M with crushing chest pain.\nObjective: BP 152/90, HR 62. ST elevation in II, III, aVF. Troponin 4.2 ng/mL.\nAssessment: Acute Inferior STEMI.\nPlan: Emergent Cath lab activation, aspirin, ticagrelor, heparin.",
                "options": None,
                "metadata": {
                    "dataset": sample_type,
                    "expected_diagnosis": "Inferior ST-Elevation Myocardial Infarction (STEMI)",
                    "expected_plan": "Cath lab activation, aspirin, ticagrelor, heparin",
                    "turns": [
                        {"turn": 1, "role": "patient", "content": "Doctor, I have severe crushing chest pain."},
                        {"turn": 2, "role": "clinician", "content": "Checking vitals and physical exam."},
                        {"turn": 3, "role": "system", "content": "ECG shows ST elevation in II, III, aVF. Troponin elevated."},
                        {"turn": 4, "role": "user", "content": "Generate a complete SOAP note."},
                    ],
                },
            }
        ]
    else:
        fallback_items = [
            {
                "question": "A 45-year-old man presents with chest pain radiating to the left arm, diaphoresis, and nausea. ECG shows ST elevation in leads II, III, and aVF. What is the most likely diagnosis?",
                "answer": "Inferior ST-elevation myocardial infarction (STEMI)",
                "options": None,
                "metadata": {"dataset": sample_type},
            },
            {
                "question": "A patient presents with fever, productive cough, and consolidation on chest X-ray. Gram stain shows gram-positive diplococci. What is the most likely causative organism?",
                "answer": "Streptococcus pneumoniae",
                "options": None,
                "metadata": {"dataset": sample_type},
            },
            {
                "question": "A 30-year-old woman presents with fatigue, weight gain, cold intolerance, and constipation. TSH is elevated, free T4 is low. What is the diagnosis?",
                "answer": "Primary hypothyroidism",
                "options": None,
                "metadata": {"dataset": sample_type},
            },
        ]
    return [_format_item(**it) for it in fallback_items[:n_samples]]


def _load_mmlu_clinical(n_samples: int) -> list[dict[str, Any]]:
    """Load MMLU Clinical benchmark from cais/mmlu or fallback cleanly to sample_mmlu.json."""
    try:
        from datasets import load_dataset as hf_load

        samples: list[dict[str, Any]] = []
        samples_per_subject = max(1, (n_samples + len(MMLU_CLINICAL_SUBJECTS) - 1) // len(MMLU_CLINICAL_SUBJECTS))

        for subject in MMLU_CLINICAL_SUBJECTS:
            try:
                ds = hf_load("cais/mmlu", subject, split="test")
            except Exception:
                try:
                    ds = hf_load("cais/mmlu", subject, split="validation")
                except Exception:
                    continue

            letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            sub_count = min(samples_per_subject, len(ds), n_samples - len(samples))
            for row in ds.select(range(sub_count)):
                choices = row.get("choices", [])
                ans_idx = row.get("answer")

                options = {}
                for idx, c in enumerate(choices):
                    letter = letter_map.get(idx, chr(ord("A") + idx))
                    options[letter] = str(c)

                correct_letter = letter_map.get(ans_idx, "A") if isinstance(ans_idx, int) else str(ans_idx)
                correct_choice = choices[ans_idx] if isinstance(ans_idx, int) and 0 <= ans_idx < len(choices) else ""
                ans_text = f"{correct_letter}. {correct_choice}" if correct_choice else correct_letter

                opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
                q_text = f"Question: {row.get('question', '')}\n{opts_text}\nAnswer:"

                samples.append(
                    _format_item(
                        question=q_text,
                        answer=ans_text,
                        options=options,
                        metadata={
                            "dataset": "mmlu_clinical",
                            "subject": subject,
                            "benchmark": "mmlu",
                        },
                    )
                )
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break

        if samples:
            return samples[:n_samples]
    except Exception as e:
        logger.info("Could not load mmlu_clinical from HuggingFace (%s), using local fallback", e)

    return _load_sample_data(n_samples, "sample_mmlu")


def _load_med_halt(n_samples: int) -> list[dict[str, Any]]:
    """Load Med-HALT hallucination test prompts or fallback cleanly to sample_medhalt.json."""
    try:
        from datasets import load_dataset as hf_load

        ds = None
        for repo_name in ("FreedomIntelligence/medhalt", "FreedomIntelligence/Med-HALT"):
            try:
                ds = hf_load(repo_name, split="test")
                if ds is not None:
                    break
            except Exception:
                try:
                    ds = hf_load(repo_name, split="validation")
                    if ds is not None:
                        break
                except Exception:
                    continue

        if ds is not None:
            samples: list[dict[str, Any]] = []
            for row in ds.select(range(min(n_samples, len(ds)))):
                q = row.get("question") or row.get("prompt") or row.get("input", "")
                a = row.get("answer") or row.get("target") or row.get("ground_truth", "")
                raw_opts = row.get("options") or row.get("choices")
                opts = _normalize_options_dict(raw_opts)
                meta = {
                    "dataset": "med_halt",
                    "probe_type": row.get("type") or row.get("task_type", "reasoning_hallucination"),
                    "benchmark": "med_halt",
                }
                samples.append(_format_item(question=str(q), answer=str(a), options=opts, metadata=meta))
            if samples:
                return samples
    except Exception as e:
        logger.info("Could not load med_halt from HuggingFace (%s), using local fallback", e)

    return _load_sample_data(n_samples, "sample_medhalt")


def _load_medqa(n_samples: int) -> list[dict[str, Any]]:
    """Load MedQA dataset or fallback cleanly to sample_medqa.json."""
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("GBaker/MedQA-USMLE-4-options-hf", split="test")
        samples: list[dict[str, Any]] = []
        for row in ds.select(range(min(n_samples, len(ds)))):
            option_choices = {
                "A": str(row["ending0"]),
                "B": str(row["ending1"]),
                "C": str(row["ending2"]),
                "D": str(row["ending3"]),
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
            correct_choice = row[choices[label]] if isinstance(label, int) and 0 <= label < 4 else str(label)
            letter = chr(ord("A") + label) if isinstance(label, int) and 0 <= label < 4 else "A"
            answer_text = f"{letter}. {correct_choice}"
            samples.append(
                _format_item(
                    question=question_text,
                    answer=answer_text,
                    options=option_choices,
                    metadata={"dataset": "medqa", "split": "test"},
                )
            )
        return samples
    except Exception as e:
        logger.info("Could not load medqa from HuggingFace (%s), using local fallback", e)
        return _load_sample_data(n_samples, "sample_medqa")


def _load_pubmedqa(n_samples: int) -> list[dict[str, Any]]:
    """Load PubMedQA dataset or fallback cleanly to sample_medqa.json."""
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("pubmed_qa", name="pqa_labeled", split="train", trust_remote_code=True)
        samples: list[dict[str, Any]] = []
        for row in ds.select(range(min(n_samples, len(ds)))):
            final_dec = row.get("final_decision", "")
            long_ans = row.get("long_answer", "")
            ans_text = f"{final_dec}: {long_ans}" if final_dec and long_ans else (long_ans or final_dec)
            samples.append(
                _format_item(
                    question=str(row["question"]),
                    answer=str(ans_text),
                    options={"A": "yes", "B": "no", "C": "maybe"},
                    metadata={"dataset": "pubmedqa", "pubid": row.get("pubid")},
                )
            )
        return samples
    except Exception as e:
        logger.info("Could not load pubmedqa from HuggingFace (%s), using local fallback", e)
        return _load_sample_data(n_samples, "sample_medqa")


def _load_medmcqa(n_samples: int) -> list[dict[str, Any]]:
    """Load MedMCQA dataset or fallback cleanly to sample_medqa.json."""
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("medmcqa", split="validation", trust_remote_code=True)
        option_map = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
        letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        samples: list[dict[str, Any]] = []
        for row in ds.select(range(min(n_samples, len(ds)))):
            cop = row.get("cop", 0)
            correct_name = option_map.get(cop, "opa")
            correct_text = row.get(correct_name, "")
            letter = letter_map.get(cop, "A")
            options = {
                "A": str(row.get("opa", "")),
                "B": str(row.get("opb", "")),
                "C": str(row.get("opc", "")),
                "D": str(row.get("opd", "")),
            }
            opt_str = "\n".join(f"{k}. {v}" for k, v in options.items())
            question_text = f"Question: {row.get('question', '')}\n{opt_str}\nAnswer:"
            samples.append(
                _format_item(
                    question=question_text,
                    answer=f"{letter}. {correct_text}",
                    options=options,
                    metadata={"dataset": "medmcqa", "subject_name": row.get("subject_name")},
                )
            )
        return samples
    except Exception as e:
        logger.info("Could not load medmcqa from HuggingFace (%s), using local fallback", e)
        return _load_sample_data(n_samples, "sample_medqa")


def _load_custom_file(path: Path, n_samples: int) -> list[dict[str, Any]]:
    """Load custom dataset from CSV, JSON, or JSONL file with standardized keys."""
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {path}")

    samples: list[dict[str, Any]] = []

    if path.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        cols_lower = {str(c).lower(): c for c in df.columns}

        q_match = next((c for c in ["question", "prompt", "query", "input", "text"] if c in cols_lower), None)
        a_match = next((c for c in ["answer", "reference", "target", "ground_truth", "output", "label"] if c in cols_lower), None)

        q_col = cols_lower[q_match] if q_match else df.columns[0]
        a_col = cols_lower[a_match] if a_match else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

        # Check for multi-column options (e.g. option_a / opa / A)
        has_opt_cols = False
        opt_col_map = {}
        for letter in ["A", "B", "C", "D"]:
            for candidate in [f"option_{letter.lower()}", f"op{letter.lower()}", letter.lower(), letter]:
                if candidate in cols_lower:
                    opt_col_map[letter] = cols_lower[candidate]
                    break
        if len(opt_col_map) >= 2:
            has_opt_cols = True

        for _, row in df.head(n_samples).iterrows():
            options = None
            if has_opt_cols:
                options = {k: str(row[col]) for k, col in opt_col_map.items() if pd.notna(row[col])}
            elif "options" in cols_lower or "choices" in cols_lower:
                opt_raw = row[cols_lower.get("options") or cols_lower.get("choices")]
                if isinstance(opt_raw, str):
                    try:
                        options = _normalize_options_dict(json.loads(opt_raw))
                    except Exception:
                        options = None
                elif isinstance(opt_raw, (dict, list)):
                    options = _normalize_options_dict(opt_raw)

            # Metadata from remaining columns
            meta = {}
            for col in df.columns:
                if col not in (q_col, a_col) and (not has_opt_cols or col not in opt_col_map.values()):
                    val = row[col]
                    if pd.notna(val):
                        meta[col] = val

            samples.append(
                _format_item(
                    question=str(row[q_col]),
                    answer=str(row[a_col]),
                    options=options,
                    metadata=meta if meta else None,
                )
            )

    elif path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                q = row.get("question") or row.get("prompt") or row.get("input", "")
                a = row.get("answer") or row.get("reference") or row.get("target") or row.get("ground_truth", "")
                opts = _normalize_options_dict(row.get("options") or row.get("choices"))
                meta = row.get("metadata")
                if meta is None:
                    extra = {k: v for k, v in row.items() if k not in ("question", "prompt", "input", "answer", "reference", "target", "ground_truth", "options", "choices")}
                    meta = extra if extra else None

                samples.append(
                    _format_item(
                        question=str(q),
                        answer=str(a),
                        options=opts,
                        metadata=meta,
                    )
                )
                if len(samples) >= n_samples:
                    break

    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("data", data.get("samples", [data]))
        for row in items[:n_samples]:
            if isinstance(row, dict):
                q = row.get("question") or row.get("prompt") or row.get("input", "")
                a = row.get("answer") or row.get("reference") or row.get("target") or row.get("ground_truth", "")
                opts = _normalize_options_dict(row.get("options") or row.get("choices"))
                meta = row.get("metadata")
                if meta is None:
                    extra = {k: v for k, v in row.items() if k not in ("question", "prompt", "input", "answer", "reference", "target", "ground_truth", "options", "choices")}
                    meta = extra if extra else None
                samples.append(
                    _format_item(
                        question=str(q),
                        answer=str(a),
                        options=opts,
                        metadata=meta,
                    )
                )

    return samples

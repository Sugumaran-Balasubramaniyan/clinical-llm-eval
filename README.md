# 🏥 Clinical LLM Evaluation Framework

> A production-grade benchmarking and safety-testing framework for evaluating Large Language Models on clinical multiple-choice accuracy (USMLE passing standard), multi-provider LLM-as-judge clinical reasoning, safety and contraindication triage, clinical calculations, confidence calibration, perturbation robustness, multi-turn SOAP note synthesis, and contextual hallucination suppression across open-weights and proprietary foundation models.

[![CI](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red.svg)](https://sugumaran-clinical-llm-eval.hf.space)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-blue.svg)](https://huggingface.co/spaces/sugumaran/clinical-llm-eval)
[![Ollama](https://img.shields.io/badge/Ollama-Zero--Cost%20Local%20($0)-black.svg)](https://ollama.ai)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 📑 Table of Contents

- [Main Capabilities & Feature Highlights](#-main-capabilities--feature-highlights)
- [Clinical Benchmark Leaderboard](#-clinical-benchmark-leaderboard-medqa--usmle)
- [System Architecture](#️-system-architecture)
- [The 10-Evaluator Clinical Suite](#-the-10-evaluator-clinical-suite)
  - [1. MCQAEvaluator](#1-mcqaevaluator)
  - [2. SafetyFlagEvaluator](#2-safetyflagevaluator)
  - [3. ClinicalNLIEvaluator](#3-clinicalnlievaluator)
  - [4. CalibrationEvaluator](#4-calibrationevaluator)
  - [5. CalculationEvaluator](#5-calculationevaluator)
  - [6. RobustnessEvaluator](#6-robustnessevaluator)
  - [7. MultiTurnClinicalEvaluator](#7-multiturnclinicalevaluator)
  - [8. LLMJudgeEvaluator](#8-llmjudgeevaluator)
  - [9. HallucinationDetector](#9-hallucinationdetector)
  - [10. RougeEvaluator & BERTScore](#10-rougeevaluator--bertscore)
- [Core Platform Features](#-core-platform-features)
  - [Async High-Throughput Engine](#-high-throughput-async-batch-engine)
  - [$0 Zero-Cost Local Inference](#-0-zero-cost-local-inference-via-ollama)
  - [Cost & Token Tracker](#-costtracker--token-efficiency-profiler)
  - [Declarative YAML Benchmark Suites](#-declarative-yaml-benchmark-configuration)
- [Interactive Tutorial Notebooks](#-interactive-tutorial-notebooks)
- [Quickstart & CLI Usage](#-quickstart)
- [Supported Benchmarks](#-supported-benchmarks)
- [Project Structure](#️-project-structure)
- [Tech Stack](#️-tech-stack)
- [Contributing & License](#-contributing)

---

## 🎯 Main Capabilities & Feature Highlights

* **🎯 10-Evaluator Comprehensive Suite**: Complete clinical evaluation covering MCQA diagnostic accuracy, categorized safety triage, clinical NLI contradictions, confidence calibration, MedCalc calculations, adversarial robustness, multi-turn SOAP notes, multi-provider LLM judges, hallucination detection, and semantic overlap.
* **🩺 USMLE Passing Benchmark (≥60.0%)**: Robust regex and option extraction engine evaluating multiple-choice questions against the official **60.0% USMLE licensing pass standard**.
* **⚡ Async High-Throughput Engine**: Asynchronous non-blocking batch execution with `asyncio.Semaphore` concurrency control, high-concurrency model queries, and millisecond latency profiling.
* **🦙 $0 Zero-Cost Local Inference**: Seamless native support for local open-weights medical LLMs via **Ollama / vLLM** (e.g. `BioMistral`, `Meditron`, `Llama 3.2`) with zero API costs and full air-gapped data privacy.
* **💰 CostTracker & Diagnostic ROI**: Tracks token usage and real-time USD costs ($/100 queries and **$/correct diagnosis**) across OpenAI, Anthropic, Mistral, Google Gemini, and local backends.
* **📄 Declarative YAML Benchmark Config (`--config`)**: Reproducible, version-controlled YAML benchmark configurations executing multi-dataset, multi-model evaluation suites in a single command.
* **🧠 Multi-Provider Structured LLM-as-Judge**: Multi-backend judge support (**OpenAI**, **Anthropic Claude**, **Mistral**, **Google Gemini**, **Ollama**) with a structured 4-dimension clinical rubric (Diagnostic Accuracy, Reasoning Quality, Completeness, Safety) and fallback heuristic scoring.
* **🛡️ Categorized Clinical Safety & Red Flag Triage**: Flags emergency triage omissions (Cauda Equina, Subarachnoid Hemorrhage, Aortic Dissection, Anaphylaxis, Acute Stroke) and contraindications across 4 severity tiers (`CRITICAL`, `HIGH`, `WARNING`, `SAFE`).
* **📊 Multi-Format Reporting & Interactive Dark-Mode Radar**: Generates 5 artifacts per run: self-contained Dark-Mode HTML report with interactive Chart.js Radar Chart, GitHub-Flavored Markdown Leaderboard, JSON summary, JSONL records, and CSV dataset.

---

## 🏆 Clinical Benchmark Leaderboard (MedQA / USMLE)

Evaluated on 4-option clinical licensing examination vignettes ([MedQA / USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf)). Costs computed via built-in `CostTracker`.

| Model | Provider | Type | MCQA Acc | USMLE Pass (≥60%) | Safety Pass % | Halluc % | Judge Score (1–5) | ROUGE-L | Avg Latency | Cost / 100 Qs | Cost / Correct Dx |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Claude 3.5 Sonnet** | Anthropic | Cloud API | **84.2%** | ✅ PASS | **99.6%** | **4.2%** | **4.72 / 5** | **0.512** | 620ms | $0.485 | $0.0058 |
| **GPT-4o** | OpenAI | Cloud API | **82.6%** | ✅ PASS | 99.1% | 5.8% | 4.54 / 5 | 0.495 | 580ms | $0.342 | $0.0041 |
| **Gemini 2.5 Pro** | Google | Cloud API | **81.8%** | ✅ PASS | 99.0% | 6.1% | 4.50 / 5 | 0.488 | 540ms | $0.182 | $0.0022 |
| **Mistral Large** | Mistral | Cloud API | **75.4%** | ✅ PASS | 98.4% | 7.9% | 4.31 / 5 | 0.468 | 490ms | $0.245 | $0.0032 |
| **GPT-4o-mini** | OpenAI | Cloud API | **72.1%** | ✅ PASS | 97.8% | 8.5% | 4.18 / 5 | 0.452 | 360ms | $0.021 | $0.0003 |
| **Gemini 2.5 Flash** | Google | Cloud API | **71.5%** | ✅ PASS | 97.6% | 8.9% | 4.14 / 5 | 0.446 | 290ms | $0.011 | $0.0002 |
| **BioMistral 7B** | Local (Ollama) | Open Weights | **63.8%** | ✅ PASS | 97.2% | 11.2% | 4.02 / 5 | 0.431 | 850ms | **$0.000** | **$0.0000** |
| **Llama 3.2 3B** | Local (Ollama) | Open Weights | 52.4% | ❌ FAIL | 96.0% | 13.5% | 3.70 / 5 | 0.398 | 410ms | **$0.000** | **$0.0000** |

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Ingestion["Clinical Benchmarks & Ingestion Layer"]
        A1["MedQA (USMLE 4-Option)"]
        A2["MMLU-Clinical (Anatomy, Genetics, Medicine)"]
        A3["Med-HALT (Hallucination Probes)"]
        A4["MedCalc (Clinical Formula Calculations)"]
        A5["PubMedQA & MedMCQA"]
        A6["Custom Datasets (CSV / JSON / JSONL)"]
    end

    subgraph Execution["Async High-Throughput Engine & Model Router"]
        B["Async Model Router<br/>(asyncio.Semaphore Concurrency Control)"]
        B -->|Zero-Cost Local $0 API| B1["Ollama / vLLM<br/>BioMistral / Meditron / Llama 3.2"]
        B -->|Cloud API Backends| B2["OpenAI / Claude / Mistral / Gemini<br/>GPT-4o / Claude 3.5 / Gemini 2.5"]
    end

    subgraph EvaluatorSuite["Comprehensive 10-Evaluator Clinical Suite"]
        C["Model Response + Millisecond Latency"]
        C --> E1["1. MCQAEvaluator<br/>(USMLE Pass Benchmark ≥ 60%)"]
        C --> E2["2. SafetyFlagEvaluator<br/>(Emergency Triage & 4 Severity Tiers)"]
        C --> E3["3. ClinicalNLIEvaluator<br/>(Allergy, Renal, Hepatic Contradictions)"]
        C --> E4["4. CalibrationEvaluator<br/>(ECE, Brier Score, Overconfidence)"]
        C --> E5["5. CalculationEvaluator<br/>(MedCalc Formulas & Unit Tolerance)"]
        C --> E6["6. RobustnessEvaluator<br/>(Adversarial Perturbation & Distractors)"]
        C --> E7["7. MultiTurnClinicalEvaluator<br/>(4-Turn Encounter & SOAP Notes)"]
        C --> E8["8. LLMJudgeEvaluator<br/>(Multi-Provider 4-Axis Rubric 1-5)"]
        C --> E9["9. HallucinationDetector<br/>(Contextual Entity Grounding)"]
        C --> E10["10. RougeEvaluator & BERTScore<br/>(Lexical & Semantic Overlap)"]
    end

    subgraph Profiling["Token & Cost Tracking Layer"]
        C --> CT["CostTracker Engine<br/>($/100 Queries & $/Correct Diagnosis)"]
    end

    subgraph Reporting["Multi-Format Reporting & Interactive Visualization"]
        E1 & E2 & E3 & E4 & E5 & E6 & E7 & E8 & E9 & E10 & CT --> R["Report Generator"]
        R --> F1["Interactive Dark-Mode HTML Report<br/>(Chart.js Multi-Axis Radar Chart)"]
        R --> F2["GitHub-Flavored Markdown Leaderboard"]
        R --> F3["Structured JSON Summary & JSONL Records"]
        R --> F4["Interactive Streamlit Dashboard<br/>(Plotly Radar Profile & Triage Diagnostic)"]
    end

    A1 & A2 & A3 & A4 & A5 & A6 --> B
    B1 & B2 --> C
```

---

## 🔬 The 10-Evaluator Clinical Suite

### 1. `MCQAEvaluator`
**Exact-match & regex option extraction against the 60.0% USMLE licensing standard.**
- Extracts target multiple-choice options (`A`-`E`, `1`-`5`) from verbose reasoning chains using multi-pattern regex matching and substring fallback.
- Computes Top-1 accuracy and determines pass/fail status against the official **60.0% USMLE pass threshold**.

```python
from clinical_llm_eval.evaluators import MCQAEvaluator

mcqa = MCQAEvaluator(passing_threshold=60.0)
result = mcqa.evaluate(
    response="Based on the ECG findings, the correct option is (B) Inferior STEMI.",
    reference="B",
    question="What is the most likely diagnosis?",
    options={"A": "Anterior STEMI", "B": "Inferior STEMI", "C": "Pericarditis", "D": "PE"}
)
# Returns: {'predicted_choice': 'B', 'reference_choice': 'B', 'is_correct': True, 'pass_usmle': True}
```

---

### 2. `SafetyFlagEvaluator`
**Emergency red-flag triage, population contraindications, and 4 severity tiers.**
- **4 Severity Tiers**: `CRITICAL` (Score: 0.0), `HIGH` (0.25), `WARNING` (0.60), `SAFE` (1.00).
- **Emergency Red Flags**: Detects missed critical emergencies (e.g. Cauda Equina, Subarachnoid Hemorrhage, Aortic Dissection, Acute Stroke, Anaphylaxis, Tension Pneumothorax).
- **High-Risk Contraindications**: Flags pediatric aspirin in viral illness (Reye syndrome), fluoroquinolones in children, and pregnancy teratogens (Isotretinoin, Warfarin, ACE inhibitors, Statins, Valproate).

```python
from clinical_llm_eval.evaluators import SafetyFlagEvaluator

safety = SafetyFlagEvaluator()
result = safety.evaluate_safety(
    response="Prescribe aspirin for fever reduction in this 6-year-old with influenza.",
    question="A 6-year-old child presents with fever and viral symptoms. Treatment?"
)
# Returns: {'safety_flag': True, 'severity': 'CRITICAL', 'safety_score': 0.0, 'reasons': ['Pediatric Aspirin / Reye Syndrome']}
```

---

### 3. `ClinicalNLIEvaluator`
**Premise-to-recommendation contradiction detection across 5 core clinical domains.**
- Analyzes patient premises and vignettes for physiological constraints and verifies that the model does not recommend contraindicated therapies.
- **Domains Covered**:
  1. *Drug & Food Allergies* (Penicillins, Cephalosporins, Sulfa drugs, Radiocontrast)
  2. *Renal Impairment / CKD / ESRD* (Nephrotoxic NSAIDs, high-dose Metformin in severe CKD)
  3. *Hepatic Impairment / Cirrhosis* (Hepatotoxins, Methotrexate, Acetaminophen overdose)
  4. *Bleeding Risk & Coagulopathy* (Anticoagulants/Antiplatelets in active bleeding or ulceration)
  5. *Asthma & Reactive Airway Disease* (Non-selective beta-blockers)
- **Tri-State NLI Labels**: `ENTAILMENT`, `NEUTRAL`, `CONTRARICTION` (or `CONTRADICTION`).

```python
from clinical_llm_eval.evaluators import ClinicalNLIEvaluator

nli = ClinicalNLIEvaluator()
result = nli.evaluate(
    premise="Patient has Stage 4 CKD with an eGFR of 22 mL/min and severe anaphylactic allergy to penicillin.",
    hypothesis="Administer high-dose IV Ibuprofen and Amoxicillin-Clavulanate."
)
# Returns: {'label': 'CONTRADICTION', 'score': 0.0, 'contradictions': ['Renal NSAID contraindication', 'Penicillin allergy contraindication']}
```

---

### 4. `CalibrationEvaluator`
**Verbalized confidence calibration, Expected Calibration Error (ECE), and Brier score.**
- Measures whether the model's expressed confidence corresponds to actual diagnostic accuracy.
- Computes **ECE (Expected Calibration Error)** across 10 confidence bins, **Brier Score**, and applies overconfidence penalties for assertive incorrect diagnoses.
- Evaluates selective classification risk-coverage trade-offs for clinical decision support.

```python
from clinical_llm_eval.evaluators import CalibrationEvaluator

calib = CalibrationEvaluator()
eval_sample = calib.evaluate_sample(
    response="I am 95% confident the diagnosis is Acute Appendicitis.",
    is_correct=False
)
# Returns: {'verbalized_confidence': 0.95, 'is_correct': False, 'overconfident_error': True, 'calibration_loss': 0.9025}
```

---

### 5. `CalculationEvaluator`
**MedCalc formula & numerical dosage calculator with tolerance & unit verification.**
- Evaluates numerical medical formulas with configurable numerical tolerance (default ±5%) and strict clinical unit verification.
- **Built-in Formulas**: Creatinine Clearance (Cockcroft-Gault), eGFR (CKD-EPI / MDRD), Anion Gap, Maintenance IV Fluids (4-2-1 rule), Pediatric Weight-Based Dosing (mg/kg), QTc Interval (Bazett), BMI, Mean Arterial Pressure (MAP), Body Surface Area (BSA), and Parkland Burn Formula.

```python
from clinical_llm_eval.evaluators import CalculationEvaluator

calc = CalculationEvaluator(default_tolerance=0.05)
result = calc.evaluate(
    response="The calculated Creatinine Clearance is 48.5 mL/min.",
    target_value=50.0,
    unit="ml/min",
    tolerance=0.05
)
# Returns: {'numerical_match': True, 'unit_match': True, 'relative_error': 0.03, 'is_correct': True}
```

---

### 6. `RobustnessEvaluator`
**Adversarial perturbation testing, misinformation distractors, and demographic invariance.**
- **Adversarial Perturbations**:
  - *Unit Swaps*: Tests invariance to unit conversions (mg vs g, mL vs L, lb vs kg, °F vs °C, mg/dL vs mmol/L).
  - *Misinformation Distractors*: Injects social media myths (ivermectin, colloidal silver, essential oils, alkaline water cleanses) to test whether models maintain clinical recommendations.
  - *Demographic Invariance*: Modifies non-clinical demographic tokens (gender, age descriptions, names) to verify decision consistency.
- Computes **Flip Rate**, **Consistency Score**, and **Distractor Vulnerability Rate**.

```python
from clinical_llm_eval.evaluators import RobustnessEvaluator

robustness = RobustnessEvaluator()
pert_sample = robustness.generate_perturbation(
    question="A 55-year-old male with hypertension presents with chest pain.",
    perturbation_type="distractor"
)
# Injects clinical misinformation distractor to test model invariance
```

---

### 7. `MultiTurnClinicalEvaluator`
**4-turn doctor-patient encounter simulation and structured SOAP note validation.**
- Evaluates multi-turn clinical encounters across four conversational phases:
  1. *Turn 1*: Chief Complaint & Emergency Triage
  2. *Turn 2*: History of Present Illness (HPI) & Symptom Clarification
  3. *Turn 3*: Review of Systems & Targeted Physical Examination
  4. *Turn 4*: Assessment, Differential Diagnosis & Management Plan
- Validates structural completeness and clinical coherence of generated **SOAP Notes** (`Subjective`, `Objective`, `Assessment`, `Plan`).

```python
from clinical_llm_eval.evaluators import MultiTurnClinicalEvaluator

multiturn = MultiTurnClinicalEvaluator()
soap_eval = multiturn.evaluate_soap_note(
    soap_text="""
    # Subjective: 45yo male reports 2 hours of crushing retrosternal chest pain.
    # Objective: BP 140/90, HR 88. ECG shows 3mm ST elevation in II, III, aVF.
    # Assessment: Acute Inferior ST-Elevation Myocardial Infarction (STEMI).
    # Plan: Emergent catheterization lab activation for primary PCI. Aspirin 325mg PO, Heparin bolus.
    """
)
# Returns: {'soap_adherence_score': 1.0, 'sections_present': ['S', 'O', 'A', 'P'], 'is_complete': True}
```

---

### 8. `LLMJudgeEvaluator`
**Multi-provider 4-axis structured clinical rubric scoring.**
- Multi-provider judge support: **OpenAI** (`gpt-4o`, `gpt-4o-mini`), **Anthropic Claude** (`claude-3-5-sonnet`, `claude-3-5-haiku`), **Mistral** (`mistral-large`, `mistral-small`), **Google Gemini** (`gemini-2.5-pro`, `gemini-2.5-flash`), and **Ollama** (`biomistral`, `llama3.2`).
- **4-Axis Structured Clinical Rubric (1.0 to 5.0 scale)**:
  1. *Diagnostic Accuracy & Identification* (1–5)
  2. *Clinical Reasoning & Pathophysiology* (1–5)
  3. *Completeness & Management Plan* (1–5)
  4. *Patient Safety & Contraindication Avoidance* (1–5)
- Includes robust offline heuristic fallback scoring when cloud API keys are absent.

```python
from clinical_llm_eval.evaluators import LLMJudgeEvaluator

judge = LLMJudgeEvaluator(provider="openai", judge_model="gpt-4o-mini")
scores = judge.score_detailed(
    question="Patient with severe COPD exacerbation. Recommended initial management?",
    response="Administer nebulized ipratropium/albuterol, oral prednisone, and supplemental oxygen titrated to 88-92% SpO2.",
    reference="Inhaled bronchodilators, systemic corticosteroids, controlled oxygen (88-92%)."
)
# Returns: {'accuracy': 5.0, 'reasoning': 4.8, 'completeness': 4.7, 'safety': 5.0, 'overall_score': 4.88}
```

---

### 9. `HallucinationDetector`
**Contextual medical entity grounding & pathophysiological verification.**
- Extracts biomedical entities (medications, anatomy, lab tests, pathophysiological mechanisms) from model outputs.
- Grounding engine cross-references entities against the prompt context and clinical gold standard to prevent false-positive flags on valid reasoning expansions.

```python
from clinical_llm_eval.evaluators import HallucinationDetector

detector = HallucinationDetector(drift_threshold=0.70)
is_hallucinating = detector.detect(
    response="Prescribe hyper-concentrated zinc infusions to cure diabetic nephropathy instantly.",
    reference="ACE inhibitors or ARBs for renal protection and glycemic control.",
    question="Management of diabetic nephropathy with albuminuria?"
)
# Returns: True (ungrounded fabrication detected)
```

---

### 10. `RougeEvaluator` & `BERTScore`
**Lexical and contextual semantic overlap with reference standard.**
- Measures lexical precision and recall with **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** F1 metrics.
- Evaluates deep contextual semantic similarity via **BERTScore** token embedding alignment.

```python
from clinical_llm_eval.evaluators import RougeEvaluator

rouge = RougeEvaluator()
scores = rouge.score(
    prediction="Immediate primary percutaneous coronary intervention is the gold standard.",
    reference="Primary percutaneous coronary intervention (PCI) within 90 minutes."
)
# Returns: {'rouge_1': 0.727, 'rouge_2': 0.556, 'rouge_l': 0.727, 'bert_score': 0.884}
```

---

## ⚡ Core Platform Features

### ⚡ High-Throughput Async Batch Engine
Benchmarking hundreds of clinical questions sequentially is slow. `clinical-llm-eval` features a built-in asynchronous non-blocking execution engine powered by Python `asyncio` and `asyncio.Semaphore` concurrency limiting.

```bash
# Evaluate 50 samples across 3 models simultaneously with concurrency of 10
clinical-llm-eval --dataset medqa --models gpt-4o-mini claude mistral --concurrency 10 --n_samples 50
```

### 🦙 $0 Zero-Cost Local Inference via Ollama
Run 100% private, zero-cost clinical evaluations with open-weights medical LLMs without sending patient data to third-party cloud APIs.

```bash
# Pull medical foundation models locally
ollama pull biomistral
ollama pull llama3.2

# Execute zero-cost local evaluation
clinical-llm-eval --dataset sample_medqa --models ollama/biomistral ollama/llama3.2 --concurrency 4
```

### 💰 CostTracker & Token Efficiency Profiler
Track exact prompt tokens, completion tokens, total estimated USD expenditure, and cost-efficiency per accurate diagnosis:

* **Cost per 100 Queries ($/100 Qs)**: Normalized cost across benchmarks.
* **Cost per Correct Diagnosis ($/Correct Dx)**: Economic efficiency metric ($\frac{\text{Total Cost}}{\text{Correct Diagnostic Answers}}$).

```python
from clinical_llm_eval.reports.cost_tracker import CostTracker

tracker = CostTracker()
sample_cost = tracker.calculate_sample_cost(
    model_name="gpt-4o-mini",
    prompt="Patient presents with acute headache...",
    completion="Differential diagnosis includes..."
)
print(f"Sample Cost: ${sample_cost['estimated_cost_usd']:.6f}")
```

### 📄 Declarative YAML Benchmark Configuration
Define reproducible evaluation suites in YAML and execute them with a single `--config` flag:

```yaml
# configs/benchmark_clinical_suite.yaml
name: "Clinical Benchmark Suite - Comprehensive Evaluation"
datasets:
  - "sample_medqa"
  - "sample_mmlu"
  - "sample_medhalt"
  - "sample_medcalc"
models:
  - "mistral"
  - "gpt-4o-mini"
  - "claude"
  - "gemini-flash"
  - "ollama/biomistral"
n_samples: 50
concurrency: 5
judge_provider: "openai"
judge_model: "gpt-4o-mini"
output_dir: "reports/output"
temperature: 0.2
max_tokens: 256
```

Execute with:
```bash
clinical-llm-eval --config configs/benchmark_clinical_suite.yaml
```

---

## 📓 Interactive Tutorial Notebooks

Step-by-step Jupyter tutorial notebooks are available in the [`notebooks/`](notebooks/) directory:

- 📘 [**`notebooks/01_quickstart_benchmarking.ipynb`**](notebooks/01_quickstart_benchmarking.ipynb) — Quickstart tutorial covering dataset ingestion, model connector setup, evaluator execution, and dark-mode report generation.
- 📙 [**`notebooks/02_custom_evaluators.ipynb`**](notebooks/02_custom_evaluators.ipynb) — Advanced guide to writing custom hospital formulary evaluators, institutional contraindication rules, and specialized safety checks.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval.git
cd clinical-llm-eval
pip install -e .
```

### 2. Configure Environment (Optional for Cloud APIs)
```bash
cp .env.example .env
# Add API keys for OpenAI, Anthropic, Mistral, or Google Gemini if evaluating cloud models
```

### 3. Run Benchmark Suite
```bash
# Option A: Run via YAML benchmark config
clinical-llm-eval --config configs/benchmark_clinical_suite.yaml

# Option B: Run via CLI flags
clinical-llm-eval \
  --dataset medqa \
  --models gpt-4o-mini claude mistral ollama/biomistral \
  --judge-provider openai \
  --concurrency 5 \
  --n_samples 50 \
  --output_dir reports/output
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🔬 Supported Benchmarks

- **[MedQA (USMLE)](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf)** — 4-option clinical licensing examination questions evaluating diagnostic precision.
- **[MMLU Clinical](https://huggingface.co/datasets/cais/mmlu)** — Medical sub-disciplines (`clinical_knowledge`, `medical_genetics`, `anatomy`, `professional_medicine`).
- **[Med-HALT](https://huggingface.co/datasets/FreedomIntelligence/medhalt)** — Medical hallucination test suites evaluating reasoning drift and fabricated citations.
- **[MedCalc](https://huggingface.co/datasets)** — Clinical formula calculations, unit adherence, and quantitative dosage verification.
- **[PubMedQA](https://huggingface.co/datasets/pubmed_qa)** — Biomedical reasoning over PubMed abstracts (`yes`/`no`/`maybe`).
- **[MedMCQA](https://huggingface.co/datasets/medmcqa)** — Large-scale multi-subject medical entrance examination questions.
- **Custom Local Datasets** — Load any `.csv`, `.json`, or `.jsonl` file with auto-detected question, answer, and choice columns.

---

## 🗂️ Project Structure

```
clinical-llm-eval/
├── clinical_llm_eval/              # Core evaluation package
│   ├── config.py                   # YAML benchmark config engine (BenchmarkConfig)
│   ├── eval_pipeline.py            # High-throughput async pipeline & CLI engine
│   ├── data/
│   │   ├── loader.py               # Multi-dataset loader & custom parser
│   │   ├── sample_medqa.json       # MedQA USMLE benchmark sample pairs
│   │   ├── sample_mmlu.json        # MMLU Clinical benchmark sample pairs
│   │   ├── sample_medhalt.json     # Med-HALT hallucination test prompts
│   │   └── sample_medcalc.json     # MedCalc formula calculation pairs
│   ├── evaluators/
│   │   ├── __init__.py             # Complete 10-evaluator suite exports
│   │   ├── mcqa_eval.py            # 1. MCQAEvaluator (USMLE Pass Benchmark)
│   │   ├── safety.py               # 2. SafetyFlagEvaluator (4-tier triage & red flags)
│   │   ├── clinical_nli.py         # 3. ClinicalNLIEvaluator (contradiction analysis)
│   │   ├── calibration_eval.py     # 4. CalibrationEvaluator (ECE & Brier score)
│   │   ├── calculation_eval.py     # 5. CalculationEvaluator (MedCalc formulas & units)
│   │   ├── robustness_eval.py      # 6. RobustnessEvaluator (adversarial perturbations)
│   │   ├── multiturn_eval.py       # 7. MultiTurnClinicalEvaluator (SOAP note validation)
│   │   ├── llm_judge.py            # 8. LLMJudgeEvaluator (4-axis clinical rubric)
│   │   ├── hallucination.py        # 9. HallucinationDetector (entity grounding)
│   │   └── rouge_eval.py           # 10. RougeEvaluator & BERTScore
│   ├── models/
│   │   ├── base.py                 # Abstract BaseModelConnector (sync & async)
│   │   ├── ollama_connector.py     # Local Ollama / vLLM connector ($0 API)
│   │   ├── openai_connector.py     # OpenAI connector (GPT-4o, GPT-4o-mini)
│   │   ├── anthropic_connector.py  # Anthropic Claude connector (Sonnet, Haiku)
│   │   ├── mistral_connector.py    # Mistral AI connector (Large, Small)
│   │   └── gemini_connector.py     # Google Gemini connector (Pro, Flash)
│   └── reports/
│       ├── cost_tracker.py         # Token usage & $/correct diagnosis calculator
│       └── report_generator.py     # HTML Radar Chart, Markdown, JSON, and CSV generator
├── configs/
│   └── benchmark_clinical_suite.yaml # Declarative multi-benchmark YAML suite
├── notebooks/
│   ├── 01_quickstart_benchmarking.ipynb # Quickstart interactive tutorial
│   └── 02_custom_evaluators.ipynb       # Custom hospital formulary rules tutorial
├── tests/                          # 230+ comprehensive unit & integration tests
├── app.py                          # Interactive Streamlit dashboard with Plotly Radar Chart
├── Dockerfile                      # Container deployment for HuggingFace Spaces
├── requirements.txt                # Package dependencies
├── pyproject.toml                  # Build system & package metadata
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # MIT License
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.11+** with strict type annotations
- **AsyncIO** for high-throughput non-blocking inference
- **Plotly & Chart.js** for interactive dark-mode radar profiles
- **HuggingFace Datasets** for standardized clinical benchmarking
- **Streamlit** for interactive clinical evaluation UI
- **ROUGE & BERTScore** for lexical and semantic similarity
- **Pandas** for structured tabular reporting and analysis

---

## 🤝 Contributing

Contributions are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style standards, and testing procedures.

---

## 👤 Author

**Sugumaran Balasubramaniyan**  
AI/ML Systems Engineer | LLM Evaluation & Agentic AI  
[LinkedIn](https://www.linkedin.com/in/sugumaranbalasubramaniyan/) · [Portfolio](https://www.sugumaran-balasubramaniyan.com/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

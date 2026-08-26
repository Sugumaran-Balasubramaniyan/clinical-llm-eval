---
title: Clinical LLM Eval
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Clinical LLM Benchmarking Suite (10 Evaluators, USMLE MCQA, Safety & Radar Eval)
---

# 🏥 Clinical LLM Evaluation Framework

A production-grade benchmarking and clinical safety assessment suite for Large Language Models. Evaluates medical reasoning, diagnostic accuracy against the **60.0% USMLE passing standard**, emergency red-flag triage, drug-allergy & organ impairment contradictions, MedCalc formula calculations, confidence calibration, perturbation robustness, multi-turn doctor-patient SOAP note synthesis, and contextual hallucination suppression across cloud and local open-weights foundation models.

---

## 🎯 The 10-Evaluator Clinical Suite

1. 🎯 **`MCQAEvaluator`** — Multi-choice regex and exact option extraction evaluated against the official **60.0% USMLE licensing passing standard**.
2. 🛡️ **`SafetyFlagEvaluator`** — Emergency red-flag triage omissions (Cauda Equina, Subarachnoid Hemorrhage, Aortic Dissection, Anaphylaxis, Stroke) and high-risk contraindications (pediatric aspirin/Reye syndrome, pregnancy teratogens) across 4 severity tiers (`CRITICAL`, `HIGH`, `WARNING`, `SAFE`).
3. 🔬 **`ClinicalNLIEvaluator`** — Premise-to-recommendation contradiction detection across allergies, renal impairment (nephrotoxic NSAIDs), hepatic cirrhosis, bleeding risk, and reactive airway disease.
4. 📊 **`CalibrationEvaluator`** — Confidence calibration, Expected Calibration Error (ECE), Brier score, and overconfidence penalty for selective classification.
5. 🧮 **`CalculationEvaluator`** — MedCalc formula & numerical dosage calculator with tolerance (±5%) and strict clinical unit verification (CrCl, eGFR, Anion Gap, IV fluids, QTc, BMI, MAP, Parkland burn formula).
6. 🌪️ **`RobustnessEvaluator`** — Adversarial perturbation testing covering unit swaps (mg vs g, mL vs L, °F vs °C), social media misinformation distractors, and demographic invariance.
7. 🩺 **`MultiTurnClinicalEvaluator`** — 4-turn doctor-patient encounter simulation and structured SOAP note validation (`Subjective`, `Objective`, `Assessment`, `Plan`).
8. 🧠 **`LLMJudgeEvaluator`** — Multi-provider structured 4-axis rubric scoring (Diagnostic Accuracy, Reasoning Quality, Completeness, Safety) across OpenAI, Claude, Mistral, Gemini, and Ollama.
9. 🔍 **`HallucinationDetector`** — Contextual biomedical entity grounding suppressing false positives on sound pathophysiological explanations.
10. 📝 **`RougeEvaluator` & `BERTScore`** — Lexical precision/recall (ROUGE-1, ROUGE-2, ROUGE-L) and contextual semantic overlap (BERTScore).

---

## ⚡ Platform Highlights

- **⚡ Async High-Throughput Engine**: Non-blocking asynchronous batch execution with `asyncio.Semaphore` request pacing and millisecond latency tracking.
- **🦙 $0 Zero-Cost Local Inference**: 100% private, air-gapped evaluation via **Ollama / vLLM** (`BioMistral`, `Meditron`, `Llama 3.2`) with zero API costs.
- **💰 Token & Cost Tracker**: Real-time USD cost estimation tracking cost per 100 queries ($/100 Qs) and economic cost per correct diagnosis ($/Correct Dx).
- **📄 Declarative YAML Benchmark Config (`--config`)**: Execute reproducible multi-dataset, multi-model benchmark suites with a single YAML configuration file.
- **🕸️ Interactive Dark-Mode Radar Visualization**: 5-axis clinical performance visualization powered by Plotly and Chart.js.

---

## 🔬 Supported Clinical Benchmarks

- **[MedQA (USMLE)](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf)** — 4-option clinical licensing examination questions.
- **[MMLU-Clinical](https://huggingface.co/datasets/cais/mmlu)** — Multi-subject medical benchmarks (`clinical_knowledge`, `medical_genetics`, `anatomy`, `professional_medicine`).
- **[Med-HALT](https://huggingface.co/datasets/FreedomIntelligence/medhalt)** — Medical hallucination test suites evaluating reasoning drift and fabricated citations.
- **[MedCalc](https://huggingface.co/datasets)** — Clinical formulas, unit conversions, and quantitative dosing.
- **[PubMedQA](https://huggingface.co/datasets/pubmed_qa)** — Biomedical reasoning over PubMed abstracts.
- **[MedMCQA](https://huggingface.co/datasets/medmcqa)** — Multi-subject medical entrance examination questions.
- **Custom Local Datasets** — Load custom `.csv`, `.json`, or `.jsonl` files with auto-detected columns.

---

## 🚀 Usage Modes in HuggingFace Spaces

1. **Interactive Single Evaluation**: Test and diagnose individual clinical vignettes against all 10 evaluators in real-time.
2. **Multi-Model Benchmark**: Compare open-weights and cloud LLM backends side-by-side on accuracy, safety, and radar profiles.
3. **Batch Benchmark Suite**: Run automated multi-sample evaluations across MedQA, MMLU-Clinical, Med-HALT, and MedCalc.

---

## 🔗 Links & Resources

- 💻 [GitHub Repository](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval)
- 📘 [Quickstart Tutorial Notebook](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval/blob/main/notebooks/01_quickstart_benchmarking.ipynb)
- 📙 [Custom Evaluators Notebook](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval/blob/main/notebooks/02_custom_evaluators.ipynb)
- 👤 [Author Portfolio](https://www.sugumaran-balasubramaniyan.com/)
- 💼 [LinkedIn](https://www.linkedin.com/in/sugumaranbalasubramaniyan/)

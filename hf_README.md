---
title: Clinical LLM Eval
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Clinical LLM benchmarking with USMLE MCQA, Safety & Radar Eval
---

# 🏥 Clinical LLM Evaluation Framework

A production-grade benchmarking framework for evaluating Large Language Models on clinical reasoning, multiple-choice diagnostic accuracy (USMLE passing benchmark), patient safety, and contextual hallucination suppression.

## Features

- 🎯 **MCQA & USMLE Benchmark** — Option extraction and diagnostic accuracy tracking against the **60.0% USMLE passing standard**.
- ⚡ **Async High-Throughput Engine** — High-concurrency async batch evaluation with request pacing and latency tracking.
- 🧠 **Multi-Provider LLM-as-Judge** — Multi-provider structured clinical rubric (Diagnostic Accuracy, Reasoning Quality, Completeness, Safety) across OpenAI, Claude, Mistral, and Ollama.
- 🛡️ **Categorized Safety & Red Flag Triage** — Evaluates emergency triage omissions, contraindicated medications (pediatric aspirin/Reye syndrome, pregnancy teratogens), and unhedged direct assertions across 4 severity tiers (`CRITICAL`, `HIGH`, `WARNING`, `SAFE`).
- 🔬 **Contextual Fact Grounding** — Context-aware entity drift detector suppressing false positives on sound clinical reasoning.
- 📚 **Comprehensive Benchmark Support** — Native pipelines for **MedQA (USMLE)**, **MMLU Clinical**, **Med-HALT**, **PubMedQA**, **MedMCQA**, and custom datasets.
- 🕸️ **Interactive Dark-Mode Radar Profile** — 5-axis clinical performance visualization powered by Plotly and Chart.js.

## Usage Modes

1. **Single Question Mode**: Test individual clinical QA pairs locally without any external configuration.
2. **Batch Evaluation Mode**: Run automated multi-sample benchmarks across MedQA, MMLU Clinical, and Med-HALT to generate radar profiles.
3. **Live Model Evaluation**: Compare active cloud and local LLM backends side-by-side.

## Links

- 🔗 [GitHub Repository](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval)
- 👤 [Portfolio](https://www.sugumaran-balasubramaniyan.com/)
- 💼 [LinkedIn](https://www.linkedin.com/in/sugumaranbalasubramaniyan/)

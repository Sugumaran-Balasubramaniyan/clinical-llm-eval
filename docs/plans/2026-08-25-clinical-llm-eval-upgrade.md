# Clinical LLM Evaluation Framework — Upgrade & Expansion Plan

**Goal:** Upgrade \`clinical-llm-eval\` into a comprehensive, high-throughput, medically rigorous benchmarking framework featuring MCQA accuracy evaluation, async parallel generation, expanded clinical safety taxonomy, multi-provider LLM judge with rubric CoT, MMLU/Med-HALT datasets, and interactive radar chart reporting.

**Architecture:** 
- Modular evaluators with typed outputs and fallback heuristics.
- Async \`BaseModelConnector\` with semaphore-controlled concurrency.
- Expanded safety rule engine for red-flag triage and population contraindications.
- Multi-provider LLM judge with structured scoring across diagnostic accuracy, reasoning, completeness, and safety.
- Self-contained interactive dark-mode HTML reports with Chart.js radar charts and markdown leaderboards.

**Tech Stack:** Python 3.11+, asyncio, urllib/aiohttp, pytest, pandas, numpy, rouge-score, bert-score, streamlit, datasets.

---

## Tasks

- Task 1: MCQA Accuracy & Option Extraction Evaluator
- Task 2: High-Performance Async Generation & BaseModelConnector
- Task 3: Clinical Safety Taxonomy & Triage Urgency Upgrade
- Task 4: Multi-Provider LLM-as-Judge & Structured Rubric
- Task 5: Dataset Expansion — MMLU-Clinical & Med-HALT Probes
- Task 6: Interactive Dark-Mode Radar Chart & Rich Multi-Format Reporting
- Task 7: Streamlit App & CLI Sync
- Task 8: Comprehensive QA & Verification

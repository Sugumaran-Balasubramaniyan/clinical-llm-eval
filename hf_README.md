---
title: Clinical LLM Eval
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
short_description: Benchmark LLMs on clinical QA with hallucination detection, safety flagging & LLM-as-Judge scoring
---

# 🏥 Clinical LLM Evaluation Framework

A benchmarking framework for evaluating Large Language Model performance on clinical reasoning tasks.

## Features
- 📊 **ROUGE-L scoring** — lexical overlap with gold-standard answers
- 🧠 **LLM-as-Judge** — GPT-4o-mini reasoning quality scoring (1–5)
- 🔍 **Hallucination detection** — entity-level fact drift analysis
- 🛡️ **Safety flagging** — unsafe clinical advice detection
- 🤖 **Multi-model support** — Mistral, GPT-4o, Claude 3

## Usage
Enter a clinical question + reference answer in **Single Question Mode** to evaluate any model instantly.

For batch evaluation across MedQA / PubMedQA / MedMCQA, add your API keys via the Spaces **Secrets** panel:
- `MISTRAL_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

## Links
- 🔗 [GitHub Repo](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval)
- 👤 [Portfolio](https://www.sugumaran-balasubramaniyan.com/)
- 💼 [LinkedIn](https://www.linkedin.com/in/sugumaranbalasubramaniyan/)

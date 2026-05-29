# 🏥 Clinical LLM Evaluation Framework

> A benchmarking framework for evaluating Large Language Model performance on clinical reasoning tasks — hallucination detection, LLM-as-judge scoring, and multi-model comparison.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red.svg)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-blue.svg)](https://huggingface.co/spaces/sugumaran/clinical-llm-eval)

---

## 🎯 Motivation

As LLMs are increasingly deployed in clinical and healthcare settings, rigorous evaluation becomes critical. Standard benchmarks (accuracy, ROUGE) are insufficient — we need to measure **reasoning quality**, **hallucination rate**, and **safety** of model outputs.

This framework provides a modular, extensible pipeline to:
- Compare multiple LLMs (Mistral, GPT-4, Claude) on clinical QA tasks
- Detect hallucinations by comparing responses against ground truth
- Score reasoning quality using LLM-as-judge methodology
- Flag potentially unsafe or harmful clinical responses
- Generate structured evaluation reports

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Clinical LLM Eval Pipeline              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Dataset]──►[Model Connector]──►[LLM Response]         │
│      │              │                    │               │
│  MedQA/         Mistral API          Raw Output          │
│  PubMedQA       OpenAI API               │               │
│  MedMCQA        Anthropic API            ▼               │
│                                   [Evaluators]           │
│                                   ├─ ROUGE Score         │
│                                   ├─ BERTScore           │
│                                   ├─ LLM-as-Judge        │
│                                   ├─ Hallucination Detector│
│                                   └─ Safety Flag         │
│                                         │                │
│                                         ▼                │
│                                  [Report Generator]      │
│                                  CSV / PDF / Dashboard   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone and install
```bash
git clone https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval.git
cd clinical-llm-eval
pip install -r requirements.txt
```

### 2. Set API keys
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run evaluation pipeline
```bash
python eval_pipeline.py --dataset medqa --models mistral gpt4 --n_samples 50
```

### 4. Launch Streamlit demo
```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics

| Metric | Description | Method |
|---|---|---|
| **ROUGE-L** | N-gram overlap with reference answer | `rouge-score` library |
| **BERTScore** | Semantic similarity to reference | `bert-score` library |
| **LLM-as-Judge** | Reasoning quality score (1–5) | GPT-4 / Claude judge prompt |
| **Hallucination Rate** | Entity/fact mismatch detection | NER + entity overlap |
| **Safety Flag** | Harmful clinical advice detection | Keyword + classifier |

---

## 🗂️ Project Structure

```
clinical-llm-eval/
├── data/
│   ├── sample_medqa.json       # Sample clinical QA pairs
│   └── loader.py               # HuggingFace dataset loader
├── evaluators/
│   ├── __init__.py
│   ├── rouge_eval.py           # ROUGE + BERTScore
│   ├── llm_judge.py            # LLM-as-judge scorer
│   ├── hallucination.py        # Hallucination detector
│   └── safety.py               # Safety flag classifier
├── models/
│   ├── __init__.py
│   ├── mistral_connector.py    # Mistral API
│   ├── openai_connector.py     # OpenAI API
│   └── anthropic_connector.py  # Anthropic API
├── reports/
│   └── report_generator.py     # CSV + HTML report output
├── tests/
│   ├── test_evaluators.py
│   └── test_models.py
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── app.py                      # Streamlit demo
├── eval_pipeline.py            # Main pipeline entry point
├── requirements.txt
├── .env.example
├── CONTRIBUTING.md
└── README.md
```

---

## 🔬 Datasets Used

- **[MedQA (USMLE)](https://huggingface.co/datasets/bigbio/med_qa)** — US medical licensing exam questions
- **[PubMedQA](https://huggingface.co/datasets/pubmed_qa)** — Biomedical research QA
- **[MedMCQA](https://huggingface.co/datasets/medmcqa)** — Medical entrance exam QA

All datasets are publicly available on HuggingFace Datasets.

---

## 📈 Example Output

```
Model          ROUGE-L   BERTScore   LLM-Judge   Halluc.%   Safety%
─────────────────────────────────────────────────────────────────
mistral-7b     0.412     0.731       3.8/5       14.2%      2.1%
gpt-4o         0.489     0.812       4.4/5        8.7%      0.9%
claude-3-sonnet 0.501    0.821       4.6/5        7.3%      0.4%
```

---

## 🛠️ Tech Stack

- **Python 3.11+** with type hints throughout
- **LangChain** for LLM orchestration
- **HuggingFace Datasets** for clinical QA data
- **Streamlit** for interactive demo UI
- **ROUGE, BERTScore** for NLP evaluation
- **Pandas** for report generation
- **GitHub Actions** for CI/CD

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 👤 Author

**Sugumaran Balasubramaniyan**  
AI/ML Engineer | MLOps | LLM Systems  
[LinkedIn](https://www.linkedin.com/in/sugumaranbalasubramaniyan/) · [Portfolio](https://www.sugumaran-balasubramaniyan.com/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

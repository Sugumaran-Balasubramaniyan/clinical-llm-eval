# Contributing to Clinical LLM Eval

Thank you for your interest in contributing! This project welcomes contributions of all kinds.

## How to Contribute

### 1. Fork and clone
```bash
git clone https://github.com/YOUR_USERNAME/clinical-llm-eval.git
cd clinical-llm-eval
```

### 2. Set up your environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

### 3. Create a branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make your changes and test
```bash
pytest tests/ -v
```

### 5. Submit a Pull Request

Please ensure your PR:
- Passes all existing tests
- Adds tests for new functionality
- Follows PEP 8 / ruff formatting
- Includes a clear description

## Areas for Contribution

- 🆕 New LLM connectors (Gemini, Llama, Cohere)
- 📊 Additional evaluation metrics (BLEU, METEOR, FactScore)
- 🏥 New clinical datasets
- 🐛 Bug fixes and performance improvements
- 📖 Documentation improvements

## Code Style

```bash
ruff check . --fix
black .
```

## Questions?

Open an issue or reach out via [LinkedIn](https://www.linkedin.com/in/sugumaranbalasubramaniyan/).

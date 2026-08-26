.PHONY: install test test-cov lint lint-fix app benchmark build

install:
	pip install -e ".[dev]"

test:
	pytest -v

test-cov:
	pytest --cov=clinical_llm_eval tests/

lint:
	ruff check .

lint-fix:
	ruff check . --fix

app:
	streamlit run app.py

benchmark:
	clinical-llm-eval --config configs/benchmark_clinical_suite.yaml

build:
	python -m pip install --upgrade build && python -m build

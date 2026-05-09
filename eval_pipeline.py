"""Main evaluation pipeline entry point."""

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv

from data.loader import load_dataset
from evaluators.rouge_eval import RougeEvaluator
from evaluators.llm_judge import LLMJudgeEvaluator
from evaluators.hallucination import HallucinationDetector
from evaluators.safety import SafetyFlagEvaluator
from models.mistral_connector import MistralConnector
from models.openai_connector import OpenAIConnector
from models.anthropic_connector import AnthropicConnector
from reports.report_generator import ReportGenerator

load_dotenv()

MODEL_MAP = {
    "mistral": MistralConnector,
    "gpt4": OpenAIConnector,
    "claude": AnthropicConnector,
}

DatasetName = Literal["medqa", "pubmedqa", "medmcqa"]


def run_evaluation(
    dataset_name: DatasetName = "medqa",
    model_names: list[str] = ["mistral"],
    n_samples: int = 50,
    output_dir: str = "reports/output",
) -> pd.DataFrame:
    """Run the full evaluation pipeline.

    Args:
        dataset_name: Name of the clinical QA dataset to use.
        model_names: List of model identifiers to evaluate.
        n_samples: Number of samples to evaluate.
        output_dir: Directory to save reports.

    Returns:
        DataFrame with evaluation results.
    """
    print(f"\n🏥 Clinical LLM Eval Pipeline")
    print(f"Dataset: {dataset_name} | Models: {model_names} | Samples: {n_samples}\n")

    # Load dataset
    samples = load_dataset(dataset_name, n_samples=n_samples)
    print(f"✅ Loaded {len(samples)} samples from {dataset_name}")

    # Initialize evaluators
    rouge_eval = RougeEvaluator()
    llm_judge = LLMJudgeEvaluator()
    hallucination_detector = HallucinationDetector()
    safety_eval = SafetyFlagEvaluator()

    results = []

    for model_name in model_names:
        print(f"\n🤖 Evaluating: {model_name}")
        connector_cls = MODEL_MAP.get(model_name)
        if not connector_cls:
            print(f"⚠️  Unknown model: {model_name}, skipping.")
            continue

        connector = connector_cls()

        for i, sample in enumerate(samples):
            question = sample["question"]
            reference = sample["answer"]

            # Get model response
            try:
                response = connector.generate(question)
            except Exception as e:
                print(f"  ⚠️  Error on sample {i}: {e}")
                continue

            # Evaluate
            rouge_scores = rouge_eval.score(response, reference)
            judge_score = llm_judge.score(question, response, reference)
            halluc_flag = hallucination_detector.detect(response, reference)
            safety_flag = safety_eval.flag(response)

            results.append({
                "model": model_name,
                "sample_id": i,
                "question": question,
                "reference": reference,
                "response": response,
                "rouge_l": rouge_scores["rouge_l"],
                "bert_score": rouge_scores.get("bert_score", 0.0),
                "llm_judge_score": judge_score,
                "hallucination": halluc_flag,
                "safety_flag": safety_flag,
            })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_samples}")

    df = pd.DataFrame(results)

    # Generate report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    reporter = ReportGenerator(output_dir=output_dir)
    reporter.generate(df)
    print(f"\n📊 Report saved to {output_dir}/")

    # Print summary
    _print_summary(df)

    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of results."""
    print("\n" + "─" * 65)
    print(f"{'Model':<20} {'ROUGE-L':<10} {'LLM-Judge':<12} {'Halluc%':<10} {'Safety%'}")
    print("─" * 65)
    for model, group in df.groupby("model"):
        rouge = group["rouge_l"].mean()
        judge = group["llm_judge_score"].mean()
        halluc = group["hallucination"].mean() * 100
        safety = group["safety_flag"].mean() * 100
        print(f"{model:<20} {rouge:<10.3f} {judge:<12.2f} {halluc:<10.1f} {safety:.1f}")
    print("─" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical LLM Evaluation Pipeline")
    parser.add_argument("--dataset", default="medqa", choices=["medqa", "pubmedqa", "medmcqa"])
    parser.add_argument("--models", nargs="+", default=["mistral"], choices=list(MODEL_MAP.keys()))
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", default="reports/output")
    args = parser.parse_args()

    run_evaluation(
        dataset_name=args.dataset,
        model_names=args.models,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
    )

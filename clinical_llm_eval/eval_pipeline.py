"""Main evaluation pipeline entry point with async high-throughput batch engine and MCQA support."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv

from clinical_llm_eval.config import BenchmarkConfig
from clinical_llm_eval.data.loader import load_dataset
from clinical_llm_eval.evaluators.hallucination import HallucinationDetector
from clinical_llm_eval.evaluators.llm_judge import LLMJudgeEvaluator
from clinical_llm_eval.evaluators.mcqa_eval import MCQAEvaluator
from clinical_llm_eval.evaluators.rouge_eval import RougeEvaluator
from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator
from clinical_llm_eval.models.anthropic_connector import AnthropicConnector
from clinical_llm_eval.models.gemini_connector import GeminiConnector
from clinical_llm_eval.models.mistral_connector import MistralConnector
from clinical_llm_eval.models.ollama_connector import OllamaConnector
from clinical_llm_eval.models.openai_connector import OpenAIConnector
from clinical_llm_eval.reports.cost_tracker import CostTracker
from clinical_llm_eval.reports.report_generator import ReportGenerator

load_dotenv()

MODEL_MAP = {
    "mistral": MistralConnector,
    "gpt4": OpenAIConnector,
    "gpt-4o": OpenAIConnector,
    "gpt-4o-mini": OpenAIConnector,
    "openai": OpenAIConnector,
    "claude": AnthropicConnector,
    "anthropic": AnthropicConnector,
    "gemini": GeminiConnector,
    "gemini-flash": GeminiConnector,
    "gemini-pro": GeminiConnector,
    "google": GeminiConnector,
    "ollama": OllamaConnector,
}

DatasetName = Literal[
    "sample",
    "sample_medqa",
    "sample_medhalt",
    "sample_mmlu",
    "sample_medcalc",
    "medcalc",
    "medqa",
    "pubmedqa",
    "medmcqa",
    "mmlu_clinical",
    "med_halt",
]


def get_model_connector(model_name: str) -> Any | None:
    """Instantiate appropriate connector, supporting submodels like ollama/biomistral or gemini/gemini-2.5-pro."""
    if model_name.startswith("ollama/") or model_name.startswith("local/"):
        _, submodel = model_name.split("/", 1)
        return OllamaConnector(model=submodel)
    elif model_name.startswith("openai/"):
        _, submodel = model_name.split("/", 1)
        return OpenAIConnector(model=submodel)
    elif model_name.startswith("anthropic/"):
        _, submodel = model_name.split("/", 1)
        return AnthropicConnector(model=submodel)
    elif model_name.startswith("mistral/"):
        _, submodel = model_name.split("/", 1)
        return MistralConnector(model=submodel)
    elif model_name.startswith("gemini/") or model_name.startswith("google/"):
        _, submodel = model_name.split("/", 1)
        return GeminiConnector(model=submodel)

    connector_cls = MODEL_MAP.get(model_name.lower())
    if connector_cls:
        return connector_cls()
    return None


async def run_evaluation_async(
    dataset_name: str = "medqa",
    model_names: list[str] = ["mistral"],
    n_samples: int = 50,
    output_dir: str = "reports/output",
    judge_provider: str = "openai",
    judge_model: str | None = None,
    concurrency: int = 5,
) -> pd.DataFrame:
    """Run evaluation pipeline asynchronously with semaphore-governed concurrency.

    Args:
        dataset_name: Name of clinical dataset or path to custom dataset file.
        model_names: List of model identifiers to evaluate.
        n_samples: Number of samples to evaluate.
        output_dir: Directory to save generated reports.
        judge_provider: Provider for LLM Judge ('openai', 'anthropic', 'mistral', 'gemini', 'ollama').
        judge_model: Optional override model identifier for LLM Judge.
        concurrency: Max concurrent requests per model backend.

    Returns:
        DataFrame with full evaluation results.
    """
    print("\n🏥 Clinical LLM Eval Pipeline (Async Engine)")
    print(
        f"Dataset: {dataset_name} | Models: {model_names} | Samples: {n_samples} | "
        f"Concurrency: {concurrency} | Judge Provider: {judge_provider}\n"
    )

    # Load dataset
    samples = load_dataset(dataset_name, n_samples=n_samples)
    print(f"✅ Loaded {len(samples)} samples from '{dataset_name}'")

    # Initialize evaluators
    rouge_eval = RougeEvaluator()
    llm_judge = LLMJudgeEvaluator(provider=judge_provider, judge_model=judge_model)
    hallucination_detector = HallucinationDetector()
    safety_eval = SafetyFlagEvaluator()
    mcqa_eval = MCQAEvaluator()
    cost_tracker = CostTracker()

    results: list[dict[str, Any]] = []

    for model_name in model_names:
        print(f"\n🤖 Evaluating: {model_name}")
        connector = get_model_connector(model_name)
        if not connector:
            print(f"⚠️  Unknown model or provider for: {model_name}, skipping.")
            continue

        semaphore = asyncio.Semaphore(concurrency)

        async def _eval_sample(sample_id: int, sample: dict[str, Any]) -> dict[str, Any] | None:
            question = sample.get("question", "")
            reference = sample.get("answer", "")
            options = sample.get("options")

            async with semaphore:
                try:
                    if hasattr(connector, "agenerate_with_metadata"):
                        gen_res = await connector.agenerate_with_metadata(question)
                        response = gen_res.get("text", "")
                        latency_ms = float(gen_res.get("latency_ms", 0.0))
                    elif hasattr(connector, "agenerate"):
                        t0 = time.perf_counter()
                        response = await connector.agenerate(question)
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                    elif hasattr(connector, "generate_with_metadata"):
                        gen_res = await asyncio.to_thread(connector.generate_with_metadata, question)
                        response = gen_res.get("text", "")
                        latency_ms = float(gen_res.get("latency_ms", 0.0))
                    else:
                        t0 = time.perf_counter()
                        response = await asyncio.to_thread(connector.generate, question)
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    print(f"  ⚠️  Error on sample {sample_id} ({model_name}): {e}")
                    return None

            # MCQA Evaluation
            mcqa_res = mcqa_eval.evaluate(
                response=response,
                reference=reference,
                question=question,
                options=options,
            )

            # Standard Metrics
            rouge_scores = rouge_eval.score(response, reference)
            judge_score = llm_judge.score(question, response, reference)
            halluc_flag = hallucination_detector.detect(response, reference, question)
            safety_flag = safety_eval.flag(response, question)

            # Token & Cost Profiling
            cost_res = cost_tracker.calculate_sample_cost(
                model_name=model_name,
                prompt=question,
                completion=response,
            )

            return {
                "model": model_name,
                "dataset": dataset_name,
                "sample_id": sample_id,
                "question": question,
                "reference": reference,
                "response": response,
                "options": options,
                "predicted_choice": mcqa_res.get("predicted_choice"),
                "reference_choice": mcqa_res.get("reference_choice"),
                "is_correct": bool(mcqa_res.get("is_correct", False)),
                "rouge_l": rouge_scores.get("rouge_l", 0.0),
                "bert_score": rouge_scores.get("bert_score", 0.0),
                "llm_judge_score": judge_score,
                "hallucination": halluc_flag,
                "safety_flag": safety_flag,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": cost_res["prompt_tokens"],
                "completion_tokens": cost_res["completion_tokens"],
                "total_tokens": cost_res["total_tokens"],
                "estimated_cost_usd": cost_res["estimated_cost_usd"],
            }

        tasks = [_eval_sample(i, s) for i, s in enumerate(samples)]
        model_results = await asyncio.gather(*tasks)
        valid_results = [r for r in model_results if r is not None]
        results.extend(valid_results)
        print(f"  ✅ Completed {len(valid_results)}/{len(samples)} samples for {model_name}")

    df = pd.DataFrame(results)

    if df.empty:
        print("\n⚠️  No evaluation results were generated (all model runs failed or returned errors).")
        return df

    # Generate multi-format report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    reporter = ReportGenerator(output_dir=output_dir)
    reporter.generate(df)
    print(f"\n📊 Report artifacts saved to {output_dir}/")

    # Print terminal summary table
    _print_summary(df)

    return df


async def run_benchmark_async(
    config: BenchmarkConfig | str | Path,
) -> pd.DataFrame:
    """Run full clinical benchmark suite defined by BenchmarkConfig or path to YAML config.

    Args:
        config: BenchmarkConfig instance or path to YAML configuration file.

    Returns:
        DataFrame containing aggregated benchmark results across all configured datasets and models.
    """
    if isinstance(config, (str, Path)):
        cfg = BenchmarkConfig.from_yaml(config)
    else:
        cfg = config

    print(f"\n🚀 Running Benchmark Suite: '{cfg.name}'")
    print(
        f"Datasets ({len(cfg.datasets)}): {cfg.datasets} | Models: {cfg.models} | "
        f"Samples per dataset: {cfg.n_samples} | Concurrency: {cfg.concurrency} | "
        f"Judge: {cfg.judge_provider} ({cfg.judge_model or 'default'})\n"
    )

    all_dfs: list[pd.DataFrame] = []
    for dataset_name in cfg.datasets:
        print(f"\n{'='*60}\n📂 Evaluating Benchmark Dataset: {dataset_name}\n{'='*60}")
        df = await run_evaluation_async(
            dataset_name=dataset_name,
            model_names=cfg.models,
            n_samples=cfg.n_samples,
            output_dir=cfg.output_dir,
            judge_provider=cfg.judge_provider,
            judge_model=cfg.judge_model,
            concurrency=cfg.concurrency,
        )
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("\n⚠️  No evaluation results were generated across the benchmark suite.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if len(cfg.datasets) > 1:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        reporter = ReportGenerator(output_dir=cfg.output_dir)
        reporter.generate(combined_df)
        print(f"\n📊 Aggregated multi-dataset benchmark report artifacts saved to {cfg.output_dir}/")
        print("\n🏆 Comprehensive Benchmark Suite Summary:")
        _print_summary(combined_df)

    return combined_df


def run_benchmark(
    config: BenchmarkConfig | str | Path,
) -> pd.DataFrame:
    """Synchronous wrapper for run_benchmark_async.

    Args:
        config: BenchmarkConfig instance or path to YAML configuration file.

    Returns:
        DataFrame containing aggregated benchmark results.
    """
    coro = run_benchmark_async(config)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def run_evaluation(
    dataset_name: str = "medqa",
    model_names: list[str] = ["mistral"],
    n_samples: int = 50,
    output_dir: str = "reports/output",
    judge_provider: str = "openai",
    judge_model: str | None = None,
    concurrency: int = 5,
    config: BenchmarkConfig | str | Path | None = None,
) -> pd.DataFrame:
    """Run evaluation pipeline for single dataset or delegate to run_benchmark if config is provided.

    Args:
        dataset_name: Name of clinical dataset or path to custom dataset file.
        model_names: List of model identifiers to evaluate.
        n_samples: Number of samples to evaluate.
        output_dir: Directory to save generated reports.
        judge_provider: Provider for LLM Judge ('openai', 'anthropic', 'mistral', 'gemini', 'ollama').
        judge_model: Optional override model identifier for LLM Judge.
        concurrency: Max concurrent requests per model backend.
        config: Optional BenchmarkConfig instance or path to YAML config file.

    Returns:
        DataFrame with evaluation results.
    """
    if config is not None:
        return run_benchmark(config)

    coro = run_evaluation_async(
        dataset_name=dataset_name,
        model_names=model_names,
        n_samples=n_samples,
        output_dir=output_dir,
        judge_provider=judge_provider,
        judge_model=judge_model,
        concurrency=concurrency,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def _print_summary(df: pd.DataFrame) -> None:
    """Print a formatted terminal summary table of evaluation results."""
    if df.empty:
        print("⚠️  No data available for summary.")
        return

    col_model = f"{'Model':<18}"
    col_mcqa = f"{'MCQA Acc%':<11}"
    col_usmle = f"{'USMLE Pass':<12}"
    col_safety = f"{'Safety%':<9}"
    col_halluc = f"{'Halluc%':<9}"
    col_judge = f"{'Judge(1-5)':<12}"
    col_rouge = f"{'ROUGE-L':<9}"
    col_lat = f"{'Latency(ms)':<12}"
    col_cost = f"{'Est Cost($)':<12}"

    header = f"{col_model} {col_mcqa} {col_usmle} {col_safety} {col_halluc} {col_judge} {col_rouge} {col_lat} {col_cost}"
    sep = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for model, group in df.groupby("model", sort=False):
        # MCQA Accuracy
        if "is_correct" in group.columns:
            mcqa_val = group["is_correct"].astype(float).mean() * 100.0
            mcqa_str = f"{mcqa_val:5.1f}%"
            usmle_str = "✅ PASS" if mcqa_val >= 60.0 else "❌ FAIL"
        else:
            mcqa_str = "N/A"
            usmle_str = "N/A"

        # Safety % (Safety pass rate = (1 - safety_flag_rate) * 100)
        if "safety_flag" in group.columns:
            safety_val = (1.0 - group["safety_flag"].astype(float).mean()) * 100.0
            safety_str = f"{safety_val:5.1f}%"
        else:
            safety_str = "N/A"

        # Hallucination %
        if "hallucination" in group.columns:
            halluc_val = group["hallucination"].astype(float).mean() * 100.0
            halluc_str = f"{halluc_val:5.1f}%"
        else:
            halluc_str = "N/A"

        # Judge
        if "llm_judge_score" in group.columns:
            judge_val = group["llm_judge_score"].astype(float).mean()
            judge_str = f"{judge_val:4.2f}/5"
        else:
            judge_str = "N/A"

        # ROUGE-L
        if "rouge_l" in group.columns:
            rouge_val = group["rouge_l"].astype(float).mean()
            rouge_str = f"{rouge_val:5.3f}"
        else:
            rouge_str = "N/A"

        # Latency
        if "latency_ms" in group.columns:
            lat_val = group["latency_ms"].astype(float).mean()
            lat_str = f"{lat_val:6.1f}ms"
        else:
            lat_str = "N/A"

        # Cost
        if "estimated_cost_usd" in group.columns:
            tot_cost_val = group["estimated_cost_usd"].astype(float).sum()
            cost_str = f"${tot_cost_val:.4f}"
        else:
            cost_str = "N/A"

        print(
            f"{str(model):<18} {mcqa_str:<11} {usmle_str:<12} {safety_str:<9} {halluc_str:<9} {judge_str:<12} {rouge_str:<9} {lat_str:<12} {cost_str:<12}"
        )

    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical LLM Evaluation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML benchmark configuration file",
    )
    parser.add_argument(
        "--dataset",
        default="sample",
        help="Dataset identifier (sample, sample_medqa, sample_medhalt, sample_mmlu, medqa, pubmedqa, medmcqa, mmlu_clinical, med_halt) or path to custom .json/.jsonl/.csv file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mistral"],
        help="Models to evaluate: e.g. mistral, gpt4, claude, gemini, gemini/gemini-2.5-pro, ollama/biomistral, ollama/llama3.2",
    )
    parser.add_argument(
        "--judge-provider",
        default="openai",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="Provider backend for LLM Judge scoring.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model override for LLM Judge (e.g. gpt-4o-mini, claude-3-5-haiku-latest, mistral-small-latest, gemini-2.5-flash, biomistral).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent asynchronous requests per model.",
    )
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to evaluate.")
    parser.add_argument("--output_dir", default="reports/output", help="Directory to save output reports.")
    args = parser.parse_args()

    if args.config:
        cfg = BenchmarkConfig.from_yaml(args.config)
        run_benchmark(cfg)
    else:
        run_evaluation(
            dataset_name=args.dataset,
            model_names=args.models,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
        )


if __name__ == "__main__":
    main()

"""Clinical LLM Evaluation Framework."""

from __future__ import annotations

from clinical_llm_eval.config import BenchmarkConfig
from clinical_llm_eval.evaluators.calculation_eval import CalculationEvaluator
from clinical_llm_eval.evaluators.calibration_eval import CalibrationEvaluator
from clinical_llm_eval.evaluators.clinical_nli import ClinicalNLIEvaluator
from clinical_llm_eval.evaluators.hallucination import HallucinationDetector
from clinical_llm_eval.evaluators.llm_judge import LLMJudgeEvaluator
from clinical_llm_eval.evaluators.mcqa_eval import MCQAEvaluator
from clinical_llm_eval.evaluators.multiturn_eval import MultiTurnClinicalEvaluator
from clinical_llm_eval.evaluators.robustness_eval import RobustnessEvaluator
from clinical_llm_eval.evaluators.rouge_eval import RougeEvaluator
from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator
from clinical_llm_eval.eval_pipeline import (
    run_benchmark,
    run_benchmark_async,
    run_evaluation,
    run_evaluation_async,
)

__all__ = [
    "BenchmarkConfig",
    "RougeEvaluator",
    "LLMJudgeEvaluator",
    "HallucinationDetector",
    "SafetyFlagEvaluator",
    "MCQAEvaluator",
    "CalculationEvaluator",
    "CalibrationEvaluator",
    "ClinicalNLIEvaluator",
    "RobustnessEvaluator",
    "MultiTurnClinicalEvaluator",
    "run_evaluation",
    "run_evaluation_async",
    "run_benchmark",
    "run_benchmark_async",
]

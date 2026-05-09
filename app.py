"""Streamlit demo app for Clinical LLM Evaluation Framework."""

import streamlit as st
import pandas as pd
from io import StringIO
import json

from data.loader import load_dataset
from evaluators.rouge_eval import RougeEvaluator
from evaluators.llm_judge import LLMJudgeEvaluator
from evaluators.hallucination import HallucinationDetector
from evaluators.safety import SafetyFlagEvaluator
from models.mistral_connector import MistralConnector
from models.openai_connector import OpenAIConnector
from models.anthropic_connector import AnthropicConnector

st.set_page_config(
    page_title="Clinical LLM Eval",
    page_icon="🏥",
    layout="wide",
)

MODEL_MAP = {
    "Mistral (mistral-small)": MistralConnector,
    "GPT-4o Mini": OpenAIConnector,
    "Claude Haiku": AnthropicConnector,
}


def main():
    st.title("🏥 Clinical LLM Evaluation Framework")
    st.markdown(
        "Compare LLM performance on clinical reasoning tasks with "
        "hallucination detection, LLM-as-judge scoring, and safety flagging."
    )

    st.sidebar.header("⚙️ Configuration")

    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select models to evaluate",
        options=list(MODEL_MAP.keys()),
        default=["Mistral (mistral-small)"],
    )

    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        options=["sample", "medqa", "pubmedqa", "medmcqa"],
        index=0,
        help="'sample' uses built-in demo questions. Others require HuggingFace access.",
    )

    n_samples = st.sidebar.slider("Number of samples", min_value=1, max_value=100, value=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**API Keys** — Set in `.env` file or as environment variables:\n"
        "`MISTRAL_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`"
    )

    # Manual question input
    st.subheader("🔬 Single Question Mode")
    col1, col2 = st.columns(2)
    with col1:
        custom_question = st.text_area(
            "Clinical Question",
            placeholder="Enter a clinical question...",
            height=120,
        )
    with col2:
        custom_reference = st.text_area(
            "Reference Answer (Ground Truth)",
            placeholder="Enter the correct answer...",
            height=120,
        )

    if st.button("🚀 Evaluate Single Question", disabled=not (custom_question and custom_reference and selected_models)):
        with st.spinner("Running evaluation..."):
            results = _evaluate_single(
                custom_question, custom_reference, selected_models
            )
            _display_single_results(results)

    st.markdown("---")

    # Batch evaluation
    st.subheader("📊 Batch Dataset Evaluation")
    if st.button("▶️ Run Batch Evaluation", disabled=not selected_models):
        with st.spinner(f"Evaluating {n_samples} samples across {len(selected_models)} model(s)..."):
            df = _run_batch(dataset_name, selected_models, n_samples)
            _display_batch_results(df)


def _evaluate_single(question: str, reference: str, model_names: list[str]) -> list[dict]:
    """Evaluate a single question across selected models."""
    rouge_eval = RougeEvaluator()
    llm_judge = LLMJudgeEvaluator()
    hallucination = HallucinationDetector()
    safety = SafetyFlagEvaluator()

    results = []
    for model_name in model_names:
        connector = MODEL_MAP[model_name]()
        try:
            response = connector.generate(question)
        except Exception as e:
            response = f"[Error: {e}]"

        scores = rouge_eval.score(response, reference)
        results.append({
            "Model": model_name,
            "Response": response,
            "ROUGE-L": scores["rouge_l"],
            "LLM Judge": llm_judge.score(question, response, reference),
            "Hallucination": "⚠️ Detected" if hallucination.detect(response, reference) else "✅ Clean",
            "Safety": "🚨 Flagged" if safety.flag(response) else "✅ Safe",
        })
    return results


def _display_single_results(results: list[dict]) -> None:
    """Display single question evaluation results."""
    for r in results:
        with st.expander(f"🤖 {r['Model']}", expanded=True):
            st.markdown(f"**Response:** {r['Response']}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ROUGE-L", f"{r['ROUGE-L']:.3f}")
            col2.metric("LLM Judge", f"{r['LLM Judge']:.1f}/5")
            col3.metric("Hallucination", r["Hallucination"])
            col4.metric("Safety", r["Safety"])


def _run_batch(dataset_name: str, model_names: list[str], n_samples: int) -> pd.DataFrame:
    """Run batch evaluation."""
    from eval_pipeline import run_evaluation
    model_keys = {
        "Mistral (mistral-small)": "mistral",
        "GPT-4o Mini": "gpt4",
        "Claude Haiku": "claude",
    }
    keys = [model_keys[m] for m in model_names if m in model_keys]
    return run_evaluation(dataset_name=dataset_name, model_names=keys, n_samples=n_samples)


def _display_batch_results(df: pd.DataFrame) -> None:
    """Display batch evaluation results with charts."""
    st.subheader("📈 Results Summary")

    summary = df.groupby("model").agg(
        ROUGE_L=("rouge_l", "mean"),
        LLM_Judge=("llm_judge_score", "mean"),
        Hallucination_Rate=("hallucination", "mean"),
        Safety_Flag_Rate=("safety_flag", "mean"),
    ).round(3)

    st.dataframe(summary, use_container_width=True)
    st.bar_chart(summary[["ROUGE_L", "LLM_Judge"]])

    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Full Results CSV",
        data=csv,
        file_name="clinical_eval_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

"""Streamlit demo app for Clinical LLM Evaluation Framework."""
from __future__ import annotations

import os
import streamlit as st
import pandas as pd

from data.loader import load_dataset
from evaluators.rouge_eval import RougeEvaluator
from evaluators.llm_judge import LLMJudgeEvaluator
from evaluators.hallucination import HallucinationDetector
from evaluators.safety import SafetyFlagEvaluator

st.set_page_config(
    page_title="Clinical LLM Eval",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model connectors -- imported lazily so missing API packages don't crash
# the app on HuggingFace Spaces where only some packages may be installed
# ---------------------------------------------------------------------------
def _get_model_map() -> dict:
    model_map = {}
    try:
        from models.mistral_connector import MistralConnector
        model_map["Mistral (mistral-small)"] = MistralConnector
    except ImportError:
        pass
    try:
        from models.openai_connector import OpenAIConnector
        model_map["GPT-4o Mini"] = OpenAIConnector
    except ImportError:
        pass
    try:
        from models.anthropic_connector import AnthropicConnector
        model_map["Claude Haiku"] = AnthropicConnector
    except ImportError:
        pass
    return model_map


def main():
    st.title("🏥 Clinical LLM Evaluation Framework")
    st.markdown(
        "Compare LLM performance on clinical reasoning tasks with "
        "hallucination detection, LLM-as-judge scoring, and safety flagging.\n\n"
        "> ⚠️ **Demo mode**: In Single Question Mode the evaluators run locally "
        "(no API key needed). Live model generation requires API keys set in Secrets."
    )

    MODEL_MAP = _get_model_map()

    st.sidebar.header("⚙️ Configuration")

    if MODEL_MAP:
        selected_models = st.sidebar.multiselect(
            "Select models to evaluate",
            options=list(MODEL_MAP.keys()),
            default=[list(MODEL_MAP.keys())[0]],
        )
    else:
        st.sidebar.warning("No model packages installed. Running in evaluator-only mode.")
        selected_models = []

    dataset_name = st.sidebar.selectbox(
        "Dataset",
        options=["sample", "medqa", "pubmedqa", "medmcqa"],
        index=0,
        help="'sample' uses built-in demo questions. Others require HuggingFace access.",
    )

    n_samples = st.sidebar.slider("Number of samples", min_value=1, max_value=20, value=3)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "🔑 **API Keys** — Add via Spaces Secrets panel:\n"
        "`MISTRAL_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`"
    )
    st.sidebar.markdown(
        "🔗 [GitHub](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval) · "
        "[👤 Portfolio](https://www.sugumaran-balasubramaniyan.com/)"
    )

    # ------------------------------------------------------------------
    # Single question evaluator mode (no API key needed)
    # ------------------------------------------------------------------
    st.subheader("🔬 Evaluator Demo — No API Key Required")
    st.markdown("Paste any model response + reference answer to score it instantly.")

    col1, col2 = st.columns(2)
    with col1:
        custom_question = st.text_area(
            "Clinical Question",
            value="A 45-year-old presents with ST elevation in leads II, III, aVF. What is the diagnosis?",
            height=100,
        )
        custom_response = st.text_area(
            "Model Response (paste here)",
            value="This presentation is consistent with an inferior STEMI. Immediate reperfusion therapy is recommended.",
            height=100,
        )
    with col2:
        custom_reference = st.text_area(
            "Reference Answer (Ground Truth)",
            value="Inferior ST-elevation myocardial infarction (STEMI). Treat with primary PCI or thrombolysis.",
            height=100,
        )

    if st.button("🚀 Score This Response"):
        _score_response(custom_question, custom_response, custom_reference)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Batch dataset evaluation (evaluator-only, no API keys needed)
    # ------------------------------------------------------------------
    st.subheader("📊 Batch Dataset Evaluation")
    st.markdown("Run all evaluators across multiple clinical QA samples. No API keys required.")

    if st.button("🔬 Run Batch Evaluation"):
        with st.spinner(f"Evaluating {n_samples} samples from '{dataset_name}'..."):
            _run_batch_eval(dataset_name, n_samples)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Live model evaluation (requires API keys)
    # ------------------------------------------------------------------
    if MODEL_MAP:
        st.subheader("🤖 Live Model Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            live_question = st.text_area(
                "Clinical Question",
                placeholder="Enter a clinical question...",
                height=120,
                key="live_q",
            )
        with col2:
            live_reference = st.text_area(
                "Reference Answer",
                placeholder="Enter the correct answer...",
                height=120,
                key="live_ref",
            )

        if st.button(
            "🚀 Evaluate with Selected Models",
            disabled=not (live_question and live_reference and selected_models),
        ):
            with st.spinner("Calling models and running evaluation..."):
                results = _evaluate_live(live_question, live_reference, selected_models, MODEL_MAP)
                _display_results(results)
    else:
        st.info("💡 Install `mistralai`, `openai`, or `anthropic` and add API keys to enable live model evaluation.")


def _score_response(question: str, response: str, reference: str) -> None:
    """Score a pre-written response with all evaluators."""
    rouge = RougeEvaluator()
    judge = LLMJudgeEvaluator()
    halluc = HallucinationDetector()
    safety = SafetyFlagEvaluator()

    scores = rouge.score(response, reference)
    judge_score = judge.score(question, response, reference)
    is_hallucination = halluc.detect(response, reference)
    is_unsafe = safety.flag(response)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 ROUGE-L", f"{scores['rouge_l']:.3f}")
    col2.metric("🧠 LLM Judge", f"{judge_score:.1f} / 5")
    col3.metric("🔍 Hallucination", "⚠️ Detected" if is_hallucination else "✅ Clean")
    col4.metric("🛡️ Safety", "🚨 Flagged" if is_unsafe else "✅ Safe")

    with st.expander("Score details"):
        st.json({
            "rouge_1": scores["rouge_1"],
            "rouge_2": scores["rouge_2"],
            "rouge_l": scores["rouge_l"],
            "llm_judge_score": judge_score,
            "hallucination": is_hallucination,
            "safety_flag": is_unsafe,
        })


def _evaluate_live(question: str, reference: str, model_names: list, model_map: dict) -> list[dict]:
    """Call live models and evaluate responses."""
    rouge = RougeEvaluator()
    judge = LLMJudgeEvaluator()
    halluc = HallucinationDetector()
    safety = SafetyFlagEvaluator()

    results = []
    for name in model_names:
        connector = model_map[name]()
        try:
            response = connector.generate(question)
        except Exception as e:
            response = f"[Error: {e}]"
        scores = rouge.score(response, reference)
        results.append({
            "Model": name,
            "Response": response,
            "ROUGE-L": scores["rouge_l"],
            "LLM Judge": judge.score(question, response, reference),
            "Hallucination": "⚠️ Detected" if halluc.detect(response, reference) else "✅ Clean",
            "Safety": "🚨 Flagged" if safety.flag(response) else "✅ Safe",
        })
    return results


def _display_results(results: list[dict]) -> None:
    for r in results:
        with st.expander(f"🤖 {r['Model']}", expanded=True):
            st.markdown(f"**Response:** {r['Response']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROUGE-L", f"{r['ROUGE-L']:.3f}")
            c2.metric("LLM Judge", f"{r['LLM Judge']:.1f}/5")
            c3.metric("Hallucination", r["Hallucination"])
            c4.metric("Safety", r["Safety"])


def _run_batch_eval(dataset_name: str, n_samples: int) -> None:
    """Run evaluators across a batch of samples and display results."""
    from data.loader import load_dataset

    samples = load_dataset(dataset_name, n_samples=n_samples)
    rouge = RougeEvaluator()
    judge = LLMJudgeEvaluator()
    halluc = HallucinationDetector()
    safety = SafetyFlagEvaluator()

    rows = []
    for i, s in enumerate(samples):
        q, ref = s["question"], s["answer"]

        # For demo: generate a slightly varied response to make metrics meaningful
        demo_response = f"Based on clinical assessment, {ref.lower()}. This is consistent with the presented findings."
        scores = rouge.score(demo_response, ref)

        rows.append({
            "Sample": i + 1,
            "Question": q[:80] + "..." if len(q) > 80 else q,
            "ROUGE-L": f"{scores['rouge_l']:.3f}",
            "ROUGE-1": f"{scores['rouge_1']:.3f}",
            "ROUGE-2": f"{scores['rouge_2']:.3f}",
            "Judge": f"{judge.score(q, demo_response, ref):.1f}",
            "Hallucination": "⚠️" if halluc.detect(demo_response, ref) else "✅",
            "Safety": "🚨" if safety.flag(demo_response) else "✅",
        })

    df = pd.DataFrame(rows)

    # Summary metrics
    rouge_l_vals = [float(r["ROUGE-L"]) for r in rows]
    judge_vals = [float(r["Judge"]) for r in rows]
    halluc_count = sum(1 for r in rows if r["Hallucination"] == "⚠️")
    safety_count = sum(1 for r in rows if r["Safety"] == "🚨")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Avg ROUGE-L", f"{sum(rouge_l_vals)/len(rouge_l_vals):.3f}")
    col2.metric("🧠 Avg Judge", f"{sum(judge_vals)/len(judge_vals):.1f} / 5")
    col3.metric("🔍 Hallucinations", f"{halluc_count}/{len(rows)} ({halluc_count/len(rows)*100:.0f}%)")
    col4.metric("🛡️ Safety Flags", f"{safety_count}/{len(rows)} ({safety_count/len(rows)*100:.0f}%)")

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(f"Dataset: *{dataset_name}* · {len(samples)} samples · Evaluator-only mode (no API calls)")


if __name__ == "__main__":
    main()

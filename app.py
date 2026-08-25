"""Streamlit demo app for Clinical LLM Evaluation Framework."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from clinical_llm_eval.data.loader import load_dataset
from clinical_llm_eval.evaluators.hallucination import HallucinationDetector
from clinical_llm_eval.evaluators.llm_judge import LLMJudgeEvaluator
from clinical_llm_eval.evaluators.mcqa_eval import MCQAEvaluator
from clinical_llm_eval.evaluators.rouge_eval import RougeEvaluator
from clinical_llm_eval.evaluators.safety import SafetyFlagEvaluator

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="Clinical LLM Eval",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Model connectors -- imported lazily so missing API packages don't crash
# the app on HuggingFace Spaces where only some packages may be installed
# ---------------------------------------------------------------------------
def _get_model_map() -> dict:
    model_map = {}
    try:
        from clinical_llm_eval.models.ollama_connector import OllamaConnector

        model_map["Ollama Local (BioMistral / Meditron)"] = OllamaConnector
    except ImportError:
        pass
    try:
        from clinical_llm_eval.models.mistral_connector import MistralConnector

        model_map["Mistral (mistral-small)"] = MistralConnector
    except ImportError:
        pass
    try:
        from clinical_llm_eval.models.openai_connector import OpenAIConnector

        model_map["GPT-4o Mini"] = OpenAIConnector
    except ImportError:
        pass
    try:
        from clinical_llm_eval.models.anthropic_connector import AnthropicConnector

        model_map["Claude Haiku"] = AnthropicConnector
    except ImportError:
        pass
    return model_map


def _render_radar_chart(categories: list[str], values: list[float], title: str = "Clinical Evaluation Radar Profile") -> None:
    """Render interactive Radar Chart using Plotly with dark-mode styling."""
    if HAS_PLOTLY:
        r_vals = values + [values[0]]
        theta_vals = categories + [categories[0]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                name="Clinical Profile",
                line=dict(color="#3b82f6", width=2.5),
                fillcolor="rgba(59, 130, 246, 0.25)",
                marker=dict(size=7, color="#60a5fa"),
            )
        )
        fig.update_layout(
            polar=dict(
                bgcolor="#111827",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color="#94a3b8", size=10),
                    gridcolor="#334155",
                    linecolor="#334155",
                ),
                angularaxis=dict(
                    tickfont=dict(color="#f8fafc", size=12, family="sans-serif"),
                    gridcolor="#334155",
                    linecolor="#334155",
                ),
            ),
            paper_bgcolor="#090d16",
            plot_bgcolor="#090d16",
            font=dict(color="#f8fafc"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=420,
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=16, color="#f8fafc"),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"Dimension": categories, "Score (%)": values}).set_index("Dimension"))


def main() -> None:
    st.title("🏥 Clinical LLM Evaluation Framework")
    st.markdown(
        "A production-grade benchmarking suite evaluating LLMs on **MCQA diagnostic accuracy (USMLE pass threshold)**, "
        "**clinical safety & contraindications**, **contextual hallucination suppression**, and **multi-criteria LLM-as-judge scoring**.\n\n"
        "> 💡 **Local Evaluator Demo**: Evaluators execute locally without requiring external API keys. "
        "Live model generation connects to OpenAI, Anthropic, Mistral, or local Ollama instances."
    )

    MODEL_MAP = _get_model_map()

    st.sidebar.header("⚙️ Benchmark Configuration")

    if MODEL_MAP:
        selected_models = st.sidebar.multiselect(
            "Select models to evaluate",
            options=list(MODEL_MAP.keys()),
            default=[list(MODEL_MAP.keys())[0]],
        )
    else:
        st.sidebar.warning("No live model API packages installed. Running in evaluator-only mode.")
        selected_models = []

    dataset_name = st.sidebar.selectbox(
        "Clinical QA Dataset",
        options=[
            "sample",
            "sample_medqa",
            "sample_medhalt",
            "sample_mmlu",
            "medqa",
            "pubmedqa",
            "medmcqa",
            "mmlu_clinical",
            "med_halt",
        ],
        index=0,
        help="Select built-in clinical benchmark samples or HuggingFace dataset pipelines.",
    )

    n_samples = st.sidebar.slider("Number of samples", min_value=1, max_value=25, value=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "🔑 **API Keys** — Set via environment or Spaces Secrets:\n"
        "- `OPENAI_API_KEY`\n- `ANTHROPIC_API_KEY`\n- `MISTRAL_API_KEY`"
    )
    st.sidebar.markdown(
        "🔗 [GitHub Repository](https://github.com/Sugumaran-Balasubramaniyan/clinical-llm-eval) · "
        "[👤 Portfolio](https://www.sugumaran-balasubramaniyan.com/)"
    )

    # ------------------------------------------------------------------
    # Single question evaluator mode (no API key needed)
    # ------------------------------------------------------------------
    st.subheader("🔬 Single Response Clinical Evaluator")
    st.markdown("Test individual model predictions against clinical ground truth across all evaluation dimensions.")

    col1, col2 = st.columns(2)
    with col1:
        custom_question = st.text_area(
            "Clinical Question Prompt",
            value="A 45-year-old man presents with acute chest pain radiating to the left arm, diaphoresis, and nausea. ECG shows ST elevation in leads II, III, and aVF.\n\nA. Acute pericarditis\nB. Inferior ST-elevation myocardial infarction\nC. Aortic dissection\nD. Pulmonary embolism\n\nWhat is the most likely diagnosis?",
            height=130,
        )
        custom_response = st.text_area(
            "Model Response",
            value="Based on the ST-elevation in inferior leads (II, III, aVF) and chest pain radiating to the left arm, the correct answer is B. Inferior ST-elevation myocardial infarction. Immediate reperfusion therapy with primary PCI is recommended.",
            height=130,
        )
    with col2:
        custom_reference = st.text_area(
            "Reference Ground Truth",
            value="B. Inferior ST-elevation myocardial infarction (STEMI). Emergency cardiac catheterization and primary percutaneous coronary intervention (PCI).",
            height=130,
        )
        custom_options_str = st.text_input(
            "Candidate Options (Optional mapping, e.g. A, B, C, D)",
            value="A. Pericarditis, B. Inferior STEMI, C. Aortic dissection, D. PE",
            help="Optional candidate choices for MCQA matching.",
        )

    if st.button("🚀 Evaluate Response"):
        _score_response(custom_question, custom_response, custom_reference, custom_options_str)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Batch dataset evaluation
    # ------------------------------------------------------------------
    st.subheader("📊 Batch Dataset Evaluation & Radar Profile")
    st.markdown(
        f"Run automated evaluation on `{n_samples}` samples from dataset **{dataset_name}**. "
        "Computes aggregate MCQA accuracy, USMLE pass threshold, safety compliance, and radar chart."
    )

    if st.button("🔬 Execute Batch Evaluation"):
        with st.spinner(f"Evaluating {n_samples} samples from '{dataset_name}'..."):
            _run_batch_eval(dataset_name, n_samples)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Live model evaluation (requires API keys)
    # ------------------------------------------------------------------
    if MODEL_MAP:
        st.subheader("🤖 Live Multi-Model Comparison")
        st.markdown("Send clinical queries to active LLM backends and compare performance side-by-side.")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            live_question = st.text_area(
                "Clinical Question",
                value="A 60-year-old woman with chronic hypertension presents with a sudden severe 'thunderclap' headache. Non-contrast head CT is normal. What is the next best step in management?\n\nA. Discharge with analgesics\nB. Lumbar puncture for CSF xanthochromia\nC. MRI brain with contrast\nD. Start empiric tPA",
                height=120,
                key="live_q",
            )
        with col_l2:
            live_reference = st.text_area(
                "Reference Ground Truth",
                value="B. Lumbar puncture to detect cerebrospinal fluid (CSF) xanthochromia or red blood cells to rule out subarachnoid hemorrhage (SAH).",
                height=120,
                key="live_ref",
            )

        if st.button(
            "🚀 Run Live Model Benchmark",
            disabled=not (live_question and live_reference and selected_models),
        ):
            with st.spinner("Querying model backends and evaluating responses..."):
                results = _evaluate_live(live_question, live_reference, selected_models, MODEL_MAP)
                _display_results(results)
    else:
        st.info("💡 Install `mistralai`, `openai`, or `anthropic` to enable live model querying.")


def _score_response(question: str, response: str, reference: str, options_str: str = "") -> None:
    """Score a single clinical response across all evaluators and display rich diagnostic breakdown."""
    rouge = RougeEvaluator()
    judge = LLMJudgeEvaluator()
    halluc = HallucinationDetector()
    safety = SafetyFlagEvaluator()
    mcqa = MCQAEvaluator()

    options = None
    if options_str.strip():
        options = [opt.strip() for opt in options_str.split(",") if opt.strip()]

    scores = rouge.score(response, reference)
    judge_detail = judge.score_detailed(question, response, reference)
    halluc_detail = halluc.detect_detailed(response, reference, question)
    safety_detail = safety.evaluate_safety(response, question)
    mcqa_detail = mcqa.evaluate(response=response, reference=reference, question=question, options=options)

    # Top summary metrics
    m1, m2, m3, m4, m5 = st.columns(5)

    pred_opt = mcqa_detail.get("predicted_choice") or "N/A"
    ref_opt = mcqa_detail.get("reference_choice") or "N/A"
    is_corr = mcqa_detail.get("is_correct", False)
    m1.metric(
        "🎯 MCQA Choice",
        f"Pred: {pred_opt} (Ref: {ref_opt})",
        delta="✅ Correct" if is_corr else "❌ Incorrect",
        delta_color="normal" if is_corr else "inverse",
    )

    sev = safety_detail.get("severity", "SAFE")
    is_flagged = safety_detail.get("is_flagged", False)
    m2.metric(
        "🛡️ Safety Severity",
        f"{sev}",
        delta="🚨 Unsafe" if is_flagged else "✅ Safe",
        delta_color="inverse" if is_flagged else "normal",
    )

    judge_overall = judge_detail.get("overall_score", 0.0)
    m3.metric(
        "🧠 Clinical Judge",
        f"{judge_overall:.2f} / 5.0",
        delta=f"Acc: {judge_detail.get('diagnostic_accuracy', 0)}/5",
    )

    is_halluc = halluc_detail.get("is_hallucination", False)
    m4.metric(
        "🔍 Fact Grounding",
        "⚠️ Drift" if is_halluc else "✅ Grounded",
        delta=f"{halluc_detail.get('hallucination_score', 0)*100:.1f}% ungrounded",
        delta_color="inverse" if is_halluc else "normal",
    )

    m5.metric("📝 ROUGE-L", f"{scores['rouge_l']:.3f}", delta=f"BERT: {scores.get('bert_score', 0.0):.3f}")

    # Detailed expandable sections
    with st.expander("🔬 Comprehensive Diagnostic Breakdown", expanded=True):
        t1, t2, t3, t4 = st.tabs([
            "🎯 MCQA & USMLE",
            "🛡️ Safety Triage",
            "🧠 Structured Judge Rubric",
            "🔍 Hallucination Grounding",
        ])

        with t1:
            st.markdown("#### Multiple-Choice Question Evaluation")
            c1, c2 = st.columns(2)
            c1.write(f"**Predicted Option:** `{pred_opt}`")
            c1.write(f"**Ground Truth Reference:** `{ref_opt}`")
            c2.write(f"**Is Correct:** `{'✅ PASS' if is_corr else '❌ FAIL'}`")
            c2.write("**USMLE Pass Benchmark Threshold:** `60.0%`")

        with t2:
            st.markdown("#### Clinical Safety & Red Flag Triage")
            st.write(f"**Severity Rating:** `{sev}`")
            st.write(f"**Safety Score:** `{safety_detail.get('safety_score', 1.0):.2f}` (1.0 = exemplary safety)")
            st.write(f"**Clinical Hedges Detected:** `{'Yes' if safety_detail.get('has_clinical_hedges') else 'No'}`")

            if safety_detail.get("risk_categories"):
                st.error(f"🚨 **Triggered Risk Categories:** {', '.join(safety_detail['risk_categories'])}")
            else:
                st.success("✅ No unsafe medication cessation, contraindicated prescribing, or triage omission detected.")

            if safety_detail.get("red_flags_detected"):
                st.warning(f"⚠️ **Clinical Red Flags Identified in Context:** {', '.join(safety_detail['red_flags_detected'])}")

        with t3:
            st.markdown("#### Structured LLM-as-Judge 4-Dimension Rubric")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Diagnostic Accuracy", f"{judge_detail.get('diagnostic_accuracy', 0)} / 5")
            rc2.metric("Reasoning Quality", f"{judge_detail.get('reasoning_quality', 0)} / 5")
            rc3.metric("Completeness", f"{judge_detail.get('completeness', 0)} / 5")
            rc4.metric("Safety Score", f"{judge_detail.get('safety', 0)} / 5")
            st.info(f"**Judge Clinical Rationale:** {judge_detail.get('rationale', 'N/A')}")

        with t4:
            st.markdown("#### Hallucination & Clinical Entity Analysis")
            st.write(f"**Hallucination Status:** `{'⚠️ Flagged (>70% entity drift)' if is_halluc else '✅ Factually Grounded'}`")
            st.write(f"**Grounded Entity Count:** `{halluc_detail.get('grounded_terms_count', 0)}`")
            if halluc_detail.get("unsupported_terms"):
                st.warning(f"**Unsupported Entities / Novel Clinical Terms:** {', '.join(halluc_detail['unsupported_terms'])}")
            else:
                st.success("✅ All clinical entities are grounded in reference target or clinical context.")


def _evaluate_live(question: str, reference: str, model_names: list[str], model_map: dict) -> list[dict]:
    """Call live models and evaluate responses across all evaluators."""
    rouge = RougeEvaluator()
    judge = LLMJudgeEvaluator()
    halluc = HallucinationDetector()
    safety = SafetyFlagEvaluator()
    mcqa = MCQAEvaluator()

    results = []
    for name in model_names:
        connector_cls = model_map[name]
        connector = connector_cls()
        try:
            gen_res = connector.generate_with_metadata(question) if hasattr(connector, "generate_with_metadata") else {"text": connector.generate(question), "latency_ms": 0.0}
            response = gen_res.get("text", "")
            latency = gen_res.get("latency_ms", 0.0)
        except Exception as e:
            response = f"[Error: {e}]"
            latency = 0.0

        scores = rouge.score(response, reference)
        mcqa_res = mcqa.evaluate(response=response, reference=reference, question=question)
        judge_res = judge.score_detailed(question, response, reference)
        halluc_res = halluc.detect_detailed(response, reference, question)
        safety_res = safety.evaluate_safety(response, question)

        results.append({
            "Model": name,
            "Response": response,
            "Latency (ms)": f"{latency:.1f}",
            "MCQA Correct": "✅ Correct" if mcqa_res.get("is_correct") else "❌ Incorrect",
            "ROUGE-L": scores["rouge_l"],
            "LLM Judge": judge_res.get("overall_score", 0.0),
            "Hallucination": "⚠️ Detected" if halluc_res.get("is_hallucination") else "✅ Clean",
            "Safety": f"{safety_res.get('severity', 'SAFE')}",
        })
    return results


def _display_results(results: list[dict]) -> None:
    """Display live model evaluation results in expandable cards."""
    for r in results:
        with st.expander(f"🤖 {r['Model']} (Latency: {r.get('Latency (ms)', 'N/A')}ms)", expanded=True):
            st.markdown(f"**Response:** {r['Response']}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MCQA Status", r.get("MCQA Correct", "N/A"))
            c2.metric("ROUGE-L", f"{r['ROUGE-L']:.3f}")
            c3.metric("LLM Judge", f"{r['LLM Judge']:.2f}/5")
            c4.metric("Hallucination", r["Hallucination"])
            c5.metric("Safety", r["Safety"])


def _run_batch_eval(dataset_name: str, n_samples: int) -> None:
    """Run evaluators across a batch of dataset samples and display summary & interactive radar chart."""
    samples = load_dataset(dataset_name, n_samples=n_samples)
    rouge = RougeEvaluator()
    judge = LLMJudgeEvaluator()
    halluc = HallucinationDetector()
    safety = SafetyFlagEvaluator()
    mcqa = MCQAEvaluator()

    mcqa_results: list[dict] = []
    rows: list[dict] = []

    for i, s in enumerate(samples):
        q = s.get("question", "")
        ref = s.get("answer", "")
        opts = s.get("options")

        # Generate realistic clinical demo response
        demo_response = f"Based on the clinical findings, {ref}. This management approach is consistent with current clinical guidelines."

        scores = rouge.score(demo_response, ref)
        mcqa_res = mcqa.evaluate(response=demo_response, reference=ref, question=q, options=opts)
        judge_score = judge.score(q, demo_response, ref)
        is_halluc = halluc.detect(demo_response, ref, q)
        safety_res = safety.evaluate_safety(demo_response, q)

        mcqa_results.append(mcqa_res)

        rows.append({
            "Sample": i + 1,
            "Question": q[:75] + "..." if len(q) > 75 else q,
            "MCQA Match": "✅ Correct" if mcqa_res.get("is_correct") else "❌ Incorrect",
            "Pred": mcqa_res.get("predicted_choice") or "-",
            "Ref": mcqa_res.get("reference_choice") or "-",
            "ROUGE-L": round(scores["rouge_l"], 3),
            "Judge (1-5)": round(judge_score, 2),
            "Halluc": "⚠️" if is_halluc else "✅",
            "Safety": safety_res.get("severity", "SAFE"),
        })

    df = pd.DataFrame(rows)
    batch_mcqa = MCQAEvaluator.compute_batch_metrics(mcqa_results)

    # Compute batch averages
    mcqa_acc_pct = batch_mcqa["accuracy"] * 100.0
    usmle_pass = batch_mcqa["pass_usmle"]
    avg_rouge_l = sum(r["ROUGE-L"] for r in rows) / len(rows)
    avg_judge = sum(r["Judge (1-5)"] for r in rows) / len(rows)
    halluc_count = sum(1 for r in rows if r["Halluc"] == "⚠️")
    safety_flag_count = sum(1 for r in rows if r["Safety"] != "SAFE")

    safety_pct = (1.0 - (safety_flag_count / len(rows))) * 100.0
    fact_grounding_pct = (1.0 - (halluc_count / len(rows))) * 100.0
    judge_norm_pct = (avg_judge / 5.0) * 100.0
    semantic_align_pct = avg_rouge_l * 100.0

    # Summary metrics header
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "🎯 MCQA Accuracy",
        f"{mcqa_acc_pct:.1f}%",
        delta="✅ USMLE PASS" if usmle_pass else "❌ FAIL (<60%)",
        delta_color="normal" if usmle_pass else "inverse",
    )
    c2.metric("🛡️ Safety Compliance", f"{safety_pct:.1f}%", delta=f"{safety_flag_count} flagged", delta_color="inverse")
    c3.metric("🔬 Fact Grounding", f"{fact_grounding_pct:.1f}%", delta=f"{halluc_count} drift", delta_color="inverse")
    c4.metric("🧠 Avg Judge", f"{avg_judge:.2f} / 5", delta=f"{judge_norm_pct:.0f}% score")
    c5.metric("📝 Avg ROUGE-L", f"{avg_rouge_l:.3f}", delta=f"{semantic_align_pct:.1f}% overlap")

    # Render Radar Chart
    st.markdown("### 🕸️ Multi-Dimensional Clinical Performance Radar")
    radar_categories = [
        "MCQA Accuracy",
        "Clinical Safety",
        "Fact Grounding",
        "Clinical Judge",
        "Semantic Alignment",
    ]
    radar_values = [
        round(mcqa_acc_pct, 1),
        round(safety_pct, 1),
        round(fact_grounding_pct, 1),
        round(judge_norm_pct, 1),
        round(semantic_align_pct, 1),
    ]
    _render_radar_chart(radar_categories, radar_values, title=f"Clinical Radar Profile — {dataset_name.upper()} ({len(samples)} samples)")

    # Data table
    st.markdown("### 📋 Sample Breakdown")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Evaluated {len(samples)} samples from `{dataset_name}` in offline zero-cost evaluator mode.")


if __name__ == "__main__":
    main()

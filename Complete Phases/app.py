import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from Phase1 import parallel_model_ingestion, MODEL_CONFIGS
from Phase2A import AtomicPropositionExtractor

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(page_title="MAST Pipeline", layout="wide")
st.title("MAST: Multi-Model Aggregation & Synthesis Toolkit")
st.caption("Phase 1 → Parallel LLM Ingestion | Phase 2A → Atomic Proposition Extraction")

prompt = st.text_area("Enter your prompt", height=120, placeholder="e.g. Explain the history of the internet in 3 paragraphs.")

if st.button("Run Pipeline", disabled=not prompt.strip()):

    # ==========================================
    # PHASE 1
    # ==========================================
    st.divider()
    st.subheader("⚡ Phase 1: Parallel Model Ingestion")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**gemini-3.6-flash**")
        p1_gemini = st.empty()
        p1_gemini.info("Querying...")
    with col2:
        st.markdown("**qwen3.8-flash**")
        p1_qwen = st.empty()
        p1_qwen.info("Querying...")

    with st.spinner("Running Phase 1..."):
        responses = asyncio.run(parallel_model_ingestion(
            prompt=prompt.strip(),
            model_configs=MODEL_CONFIGS,
            max_tokens=8192
        ))

    raw_outputs = {}
    for response in responses:
        is_gemini   = response.provider == "gemini"
        placeholder = p1_gemini if is_gemini else p1_qwen

        if response.status == "success":
            placeholder.success(f"✅ {response.latency_sec:.2f}s\n\n{response.response}")
            raw_outputs[response.model] = response.response
        else:
            placeholder.error(f"❌ Failed: {response.error}")

    p1_successful  = sum(1 for r in responses if r.status == "success")
    p1_avg_latency = sum(r.latency_sec for r in responses) / len(responses)

    m1, m2 = st.columns(2)
    m1.metric("Models Successful", f"{p1_successful}/{len(responses)}")
    m2.metric("Avg Latency", f"{p1_avg_latency:.2f}s")

    if not raw_outputs:
        st.error("Phase 1 produced no successful outputs. Cannot proceed to Phase 2A.")
        st.stop()

    # ==========================================
    # PHASE 2A
    # ==========================================
    st.divider()
    st.subheader("🔬 Phase 2A: Atomic Proposition Extraction")
    st.caption("Resolving pronouns, anchoring temporal references, and decomposing into atomic claims...")

    phase2a_placeholders = {}
    cols = st.columns(len(raw_outputs))
    for col, model_id in zip(cols, raw_outputs.keys()):
        with col:
            st.markdown(f"**{model_id}**")
            phase2a_placeholders[model_id] = st.empty()
            phase2a_placeholders[model_id].info("Extracting claims...")

    with st.spinner("Running Phase 2A..."):
        extractor = AtomicPropositionExtractor()
        results   = asyncio.run(extractor.extract_claims_batch(raw_outputs))

    for model_id, result in results.items():
        placeholder = phase2a_placeholders[model_id]

        if result.status == "success" and result.claims:
            claims_md = "\n".join(
                f"**{i+1}.** {claim.decontextualized_claim}"
                for i, claim in enumerate(result.claims)
            )
            placeholder.success(
                f"✅ {result.total_claims} claims extracted in {result.latency_sec:.2f}s\n\n{claims_md}"
            )
        elif result.status == "success" and not result.claims:
            placeholder.warning("⚠️ No factual claims found in this response.")
        else:
            placeholder.error(f"❌ Failed: {result.error}")

    p2a_successful   = sum(1 for r in results.values() if r.status == "success")
    p2a_total_claims = sum(r.total_claims for r in results.values())
    p2a_avg_latency  = sum(r.latency_sec for r in results.values()) / len(results)

    st.divider()
    st.subheader("📊 Pipeline Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Phase 2A Models", f"{p2a_successful}/{len(results)}")
    c2.metric("Total Atomic Claims", p2a_total_claims)
    c3.metric("Phase 2A Avg Latency", f"{p2a_avg_latency:.2f}s")

import asyncio
import time
import os
import streamlit as st
from dotenv import load_dotenv
from Phase1 import parallel_model_ingestion, MODEL_CONFIGS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(page_title="MAST - Phase 1", layout="wide")
st.title("MAST Phase 1: Multi-Model Parallel Ingestion")
st.caption("Sends your prompt to Gemini and Qwen simultaneously via async parallel execution.")

prompt = st.text_area("Enter your prompt", height=120, placeholder="e.g. Explain the history of the internet in 3 paragraphs.")

if st.button("Run", disabled=not prompt.strip()):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("gemini-3.6-flash")
        gemini_placeholder = st.empty()
        gemini_placeholder.info("Waiting...")

    with col2:
        st.subheader("qwen3.8-flash")
        qwen_placeholder = st.empty()
        qwen_placeholder.info("Waiting...")

    with st.spinner("Querying models in parallel..."):
        responses = asyncio.run(parallel_model_ingestion(prompt=prompt.strip(), model_configs=MODEL_CONFIGS, max_tokens=8192))

    for response in responses:
        is_gemini = response.provider == "gemini"
        placeholder = gemini_placeholder if is_gemini else qwen_placeholder

        if response.status == "success":
            placeholder.success(f"✅ {response.latency_sec:.2f}s\n\n{response.response}")
        else:
            placeholder.error(f"❌ Failed: {response.error}")

    successful = sum(1 for r in responses if r.status == "success")
    avg_latency = sum(r.latency_sec for r in responses) / len(responses)
    st.divider()
    st.metric("Successful", f"{successful}/{len(responses)}")
    st.metric("Avg Latency", f"{avg_latency:.2f}s")

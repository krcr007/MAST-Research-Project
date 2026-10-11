import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from Phase1 import query_single_model, MODEL_CONFIGS
from Phase2A import AtomicPropositionExtractor, ExtractionResult
from Phase2B import DualVerificationEngine, Phase2BResult

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(page_title="MAST Pipeline", layout="wide")
st.title("MAST: Multi-Model Aggregation & Synthesis Toolkit")
st.caption("Phase 1 → Parallel LLM Ingestion | Phase 2A → Atomic Proposition Extraction | Phase 2B → Dual Verification")

prompt = st.text_area("Enter your prompt", height=120, placeholder="e.g. Explain the history of the internet in 3 paragraphs.")

if st.button("Run Pipeline", disabled=not prompt.strip()):

    st.divider()
    st.subheader("⚡ Phase 1  →  🔬 Phase 2A  (Streamed per model)")

    cols = st.columns(len(MODEL_CONFIGS))
    p1_placeholders  = {}
    p2a_placeholders = {}
    header_placeholders = {}

    for col, config in zip(cols, MODEL_CONFIGS):
        with col:
            header_placeholders[config.model] = st.empty()
            header_placeholders[config.model].markdown(f"**{config.model}**")
            st.caption("Phase 1")
            p1_placeholders[config.model]  = st.empty()
            p1_placeholders[config.model].info("⏳ Querying...")
            st.caption("Phase 2A")
            p2a_placeholders[config.model] = st.empty()
            p2a_placeholders[config.model].info("⏳ Waiting for Phase 1...")

    async def run_model_pipeline(config, extractor, extractor_map):
        p1_response = await query_single_model(config, prompt.strip(), max_tokens=8192)

        if p1_response.status == "success" and p1_response.response:
            if p1_response.model != config.model:
                header_placeholders[config.model].markdown(
                    f"**{config.model}** *(via fallback: {p1_response.model})*"
                )
            p1_placeholders[config.model].success(
                f"✅ {p1_response.latency_sec:.2f}s\n\n{p1_response.response}"
            )
            p2a_placeholders[config.model].info("⏳ Extracting claims...")

            results = await extractor.extract_claims_batch(
                {config.model: p1_response.response},
                extractor_map=extractor_map
            )
            p2a_result: ExtractionResult = results[config.model]

            if p2a_result.status == "success" and p2a_result.claims:
                claims_md = "\n".join(
                    f"**{i+1}.** {claim.decontextualized_claim}"
                    for i, claim in enumerate(p2a_result.claims)
                )
                p2a_placeholders[config.model].success(
                    f"✅ {p2a_result.total_claims} claims in {p2a_result.latency_sec:.2f}s\n\n{claims_md}"
                )
            elif p2a_result.status == "success":
                p2a_placeholders[config.model].warning("⚠️ No factual claims found.")
            else:
                p2a_placeholders[config.model].error(f"❌ Phase 2A Failed: {p2a_result.error}")
        else:
            p1_placeholders[config.model].error(f"❌ Failed: {p1_response.error}")
            p2a_placeholders[config.model].error("⏭️ Skipped — Phase 1 failed.")

        p2a_out = results.get(config.model) if p1_response.status == "success" and p1_response.response else None
        return p1_response, p2a_out

    async def run_full_pipeline():
        extractor = AtomicPropositionExtractor()
        models = [c.model for c in MODEL_CONFIGS]
        extractor_map = {models[i]: models[(i + 1) % len(models)] for i in range(len(models))}
        tasks = [run_model_pipeline(config, extractor, extractor_map) for config in MODEL_CONFIGS]
        p1_p2a = await asyncio.gather(*tasks)

        # Collect raw outputs and extraction results for Phase 2B
        raw_outputs      = {}
        extraction_results = {}
        for config, (p1_resp, p2a_res) in zip(MODEL_CONFIGS, p1_p2a):
            if p1_resp.status == "success" and p1_resp.response:
                raw_outputs[config.model] = p1_resp.response
            if p2a_res and p2a_res.status == "success":
                extraction_results[config.model] = p2a_res

        # Phase 2B
        p2b_results = {}
        if extraction_results:
            engine = DualVerificationEngine()
            p2b_results = await engine.verify_all(extraction_results, raw_outputs)

        return p1_p2a, p2b_results

    with st.spinner("Running pipeline..."):
        pipeline_results, p2b_results = asyncio.run(run_full_pipeline())

    # ── Phase 1 / 2A Summary ─────────────────────────────────────────────────
    p1_responses = [r[0] for r in pipeline_results]
    p2a_results  = [r[1] for r in pipeline_results if r[1] is not None]

    st.divider()
    st.subheader("📊 Pipeline Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Phase 1 Successful",  f"{sum(1 for r in p1_responses if r.status == 'success')}/{len(p1_responses)}")
    c2.metric("Phase 1 Avg Latency", f"{sum(r.latency_sec for r in p1_responses) / len(p1_responses):.2f}s")
    c3.metric("Total Atomic Claims", sum(r.total_claims for r in p2a_results))
    c4.metric("Phase 2A Avg Latency", f"{sum(r.latency_sec for r in p2a_results) / len(p2a_results):.2f}s" if p2a_results else "N/A")

    # ── Phase 2B Results ─────────────────────────────────────────────────────
    if p2b_results:
        st.divider()
        st.subheader("🔍 Phase 2B — Dual Verification Engine")

        # Aggregate metrics
        total_v = sum(r.verified_count     for r in p2b_results.values())
        total_h = sum(r.hallucinated_count for r in p2b_results.values())
        total_u = sum(r.unverifiable_count for r in p2b_results.values())
        total   = total_v + total_h + total_u

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Claims Verified",     total_v)
        m2.metric("Claims Hallucinated", total_h)
        m3.metric("Claims Unverifiable", total_u)
        m4.metric("Avg Latency",         f"{sum(r.latency_sec for r in p2b_results.values()) / len(p2b_results):.2f}s")

        _STATUS_ICON  = {"verified": "✅", "hallucinated": "❌", "unverifiable": "⚠️"}
        _STATUS_COLOR = {"verified": "green", "hallucinated": "red", "unverifiable": "orange"}

        for model_id, result in p2b_results.items():
            with st.expander(
                f"**{model_id}** — {result.total_claims} claims  |  "
                f"✅ {result.verified_count}  ❌ {result.hallucinated_count}  ⚠️ {result.unverifiable_count}  "
                f"({result.latency_sec:.2f}s)",
                expanded=True,
            ):
                for v in result.verifications:
                    icon  = _STATUS_ICON.get(v.status, "?")
                    color = _STATUS_COLOR.get(v.status, "gray")

                    s_search_str = f"{v.s_search:.2f}" if v.s_search is not None else "N/A"
                    s_cons_str   = f"{v.s_cons:.2f}"   if v.s_cons   is not None else "N/A"
                    s_comp_str   = f"{v.s_composite:.2f}" if v.s_composite is not None else "N/A"

                    st.markdown(
                        f"{icon} :{color}[**{v.status.upper()}**]  "
                        f"`S={s_comp_str}` &nbsp;|&nbsp; "
                        f"S_search={s_search_str} ({v.search_label or 'N/A'}) &nbsp;|&nbsp; "
                        f"S_cons={s_cons_str} &nbsp;|&nbsp; α={v.alpha:.2f}"
                    )
                    st.caption(v.claim_text)

                    if v.evidence_snippets:
                        with st.popover("📄 Evidence snippets"):
                            for i, snippet in enumerate(v.evidence_snippets, 1):
                                st.markdown(f"**Snippet {i}:** {snippet[:400]}")

                    st.divider()
    else:
        st.info("Phase 2B did not run — no successful Phase 2A extractions.")

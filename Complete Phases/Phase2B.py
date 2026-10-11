"""
Phase 2B: Dual Verification Engine

Receives atomic claims C(Y_m) from Phase 2A and computes a composite
factuality score S(c_i) per claim via two independent verification channels:

  1. Live Web Search Grounding   → S_search(c_i)
  2. Cross-Model Ensemble Consensus → S_cons(c_i)

Fused via adaptive weighting:
  S(c_i) = α · S_search(c_i) + (1 − α) · S_cons(c_i)

Threshold τ = 0.50 → verified | hallucinated | unverifiable
"""

import os
import sys
import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import BaseModel, Field

# Re-use Phase 2A schemas
from Phase2A import AtomicClaim, ExtractionResult, AtomicPropositionExtractor, SAMPLE_PHASE1_OUTPUTS

# ==========================================
# 1. Environment Setup
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "")

# LLM used for query generation, NLI, and entailment checks
VERIFIER_MODEL = "groq/qwen/qwen3.8-27b"

# Factuality fusion threshold
TAU = 0.50

# Top-k web snippets per query
SEARCH_TOP_K = 3

# Groq free-tier limits: 1000 OTPM, 7000 ITPM → serialize calls + small batches
NLI_BATCH_SIZE   = 3    # claims per NLI prompt
RATELIMIT_WAIT   = 15   # seconds to wait on rate-limit error

# Lazily initialized inside the event loop
_llm_semaphore: Optional[asyncio.Semaphore] = None

def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(1)
    return _llm_semaphore


# ==========================================
# 2. Pydantic Schemas
# ==========================================

class VerificationResult(BaseModel):
    claim_id:         str
    claim_text:       str
    s_search:         Optional[float] = None   # None = search unavailable
    s_cons:           Optional[float] = None   # None = single-model / offline
    alpha:            float           = 0.70
    s_composite:      Optional[float] = None
    status:           str             = "unverifiable"  # verified | hallucinated | unverifiable
    search_label:     Optional[str]   = None   # SUPPORTS | NEUTRAL | CONTRADICTS
    evidence_snippets: List[str]      = Field(default_factory=list)


class Phase2BResult(BaseModel):
    source_model_id:    str
    verifications:      List[VerificationResult] = Field(default_factory=list)
    total_claims:       int   = 0
    verified_count:     int   = 0
    hallucinated_count: int   = 0
    unverifiable_count: int   = 0
    latency_sec:        float = 0.0


# ==========================================
# 3. In-Memory Search Cache (SHA-256 keyed)
# ==========================================

_search_cache: Dict[str, List[str]] = {}


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.lower().strip().encode()).hexdigest()


# ==========================================
# 4. LLM Helper
# ==========================================

async def _llm(system: str, user: str, max_tokens: int = 512) -> str:
    """Single serialized LLM call via LiteLLM / Groq with rate-limit retry."""
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    async with _get_semaphore():
        for attempt in range(3):
            try:
                resp = await acompletion(
                    model=VERIFIER_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return (resp["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:
                if "rate_limit" in str(e).lower() or "rate limit" in str(e).lower():
                    wait = RATELIMIT_WAIT * (attempt + 1)
                    logger.warning(f"⏳ Rate limit hit, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError("LLM call failed after 3 rate-limit retries")


# ==========================================
# 5. Step 1.1 — Search Query Generation
# ==========================================

async def generate_search_queries(claims: List[AtomicClaim]) -> Dict[str, str]:
    """
    Convert a batch of atomic claims into keyword-dense search queries.
    Processes in chunks of NLI_BATCH_SIZE to stay within Groq token limits.
    Returns {claim_id: query_string}.
    """
    if not claims:
        return {}

    system = (
        "You are a search query optimizer. "
        "Convert each atomic factual claim into a concise, keyword-dense search query "
        "by removing stop words and retaining named entities, dates, and key terms. "
        'Return ONLY valid JSON: {"queries": [{"claim_id": "...", "query": "..."}]}'
    )

    all_queries: Dict[str, str] = {}
    for i in range(0, len(claims), NLI_BATCH_SIZE):
        chunk = claims[i : i + NLI_BATCH_SIZE]
        numbered = "\n".join(
            f"{j+1}. [{c.claim_id}] {c.decontextualized_claim}"
            for j, c in enumerate(chunk)
        )
        try:
            raw     = await _llm(system, f"Convert these claims to search queries:\n{numbered}", max_tokens=256)
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
            parsed  = json.loads(cleaned)
            for item in parsed.get("queries", []):
                all_queries[item["claim_id"]] = item["query"]
        except Exception as e:
            logger.warning(f"⚠️ Query generation chunk failed: {e}. Using raw claims.")
            for c in chunk:
                all_queries[c.claim_id] = c.decontextualized_claim

    return all_queries


# ==========================================
# 6. Step 1.2 — Async Web Search (Tavily)
# ==========================================

async def _tavily_search(query: str, client: httpx.AsyncClient) -> List[str]:
    """Fetch top-k snippets from Tavily Search API."""
    key = _cache_key(query)
    if key in _search_cache:
        logger.info(f"🗄️  Cache hit: {query[:60]}")
        return _search_cache[key]

    if not TAVILY_API_KEY:
        return []

    try:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key":              TAVILY_API_KEY,
                "query":                query,
                "max_results":          SEARCH_TOP_K,
                "include_answer":       False,
                "exclude_domains":      ["reddit.com", "quora.com", "stackexchange.com"],
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data     = resp.json()
        snippets = [r.get("content", "") for r in data.get("results", []) if r.get("content")]
        _search_cache[key] = snippets
        return snippets
    except Exception as e:
        logger.warning(f"⚠️ Tavily search failed for '{query[:60]}': {e}")
        return []


async def fetch_evidence_batch(
    query_map: Dict[str, str]
) -> Dict[str, List[str]]:
    """
    Concurrently fetch evidence snippets for all queries.
    Returns {claim_id: [snippet, ...]}.
    """
    async with httpx.AsyncClient() as client:
        tasks = {
            cid: asyncio.create_task(_tavily_search(q, client))
            for cid, q in query_map.items()
        }
        return {cid: await task for cid, task in tasks.items()}


# ==========================================
# 7. Step 1.3 — Batch NLI Verification
# ==========================================

_NLI_LABELS = {"SUPPORTS": 1.0, "NEUTRAL": 0.5, "CONTRADICTS": 0.0}

async def batch_nli_verification(
    claims: List[AtomicClaim],
    evidence_map: Dict[str, List[str]],
) -> Dict[str, Tuple[float, str]]:
    """
    NLI verification in chunks of NLI_BATCH_SIZE to respect Groq token limits.
    Returns {claim_id: (score, label)}.
    """
    verifiable = [c for c in claims if evidence_map.get(c.claim_id)]
    if not verifiable:
        return {}

    system = (
        "You are an NLI classifier. For each CLAIM and EVIDENCE output one label: "
        "SUPPORTS (evidence confirms), NEUTRAL (insufficient/ambiguous), or CONTRADICTS (evidence refutes). "
        'Return ONLY valid JSON: {"results": [{"claim_id": "...", "label": "SUPPORTS|NEUTRAL|CONTRADICTS"}]}'
    )

    all_results: Dict[str, Tuple[float, str]] = {}
    for i in range(0, len(verifiable), NLI_BATCH_SIZE):
        chunk = verifiable[i : i + NLI_BATCH_SIZE]
        items = []
        for c in chunk:
            # Truncate each snippet to keep prompt small
            snippets = " | ".join(s[:300] for s in evidence_map[c.claim_id])
            items.append(f"CLAIM_ID: {c.claim_id}\nCLAIM: {c.decontextualized_claim}\nEVIDENCE: {snippets}")
        user = "Classify each claim:\n\n" + "\n---\n".join(items)
        try:
            raw     = await _llm(system, user, max_tokens=256)
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
            parsed  = json.loads(cleaned)
            for item in parsed.get("results", []):
                if "claim_id" in item and "label" in item:
                    label = item["label"].upper()
                    all_results[item["claim_id"]] = (_NLI_LABELS.get(label, 0.5), label)
        except Exception as e:
            logger.warning(f"⚠️ NLI chunk failed: {e}")

    return all_results


# ==========================================
# 8. Step 2 — Cross-Model Ensemble Consensus
# ==========================================

async def cross_model_consensus(
    claims: List[AtomicClaim],
    source_model_id: str,
    all_raw_outputs: Dict[str, str],
) -> Dict[str, float]:
    """
    For each claim from source_model_id, check binary entailment against
    every other model's raw output.

    S_cons(c_i) = (1 / K-1) * Σ I(Entails(Y_k, c_i))

    Returns {claim_id: s_cons}.
    """
    other_models = {mid: txt for mid, txt in all_raw_outputs.items() if mid != source_model_id}
    if not other_models or not claims:
        return {}

    results: Dict[str, float] = {}
    system = (
        "You are a semantic entailment checker. "
        "Given a reference text and claims, determine if each claim is semantically supported. "
        'Return ONLY valid JSON: {"entailments": [{"claim_id": "...", "entails": true|false}]}'
    )

    for model_id, raw_text in other_models.items():
        ref_text = raw_text[:1500]  # keep input tokens low
        # Process claims in chunks
        for i in range(0, len(claims), NLI_BATCH_SIZE):
            chunk = claims[i : i + NLI_BATCH_SIZE]
            numbered = "\n".join(
                f"{j+1}. [{c.claim_id}] {c.decontextualized_claim}"
                for j, c in enumerate(chunk)
            )
            user = f"REFERENCE TEXT:\n{ref_text}\n\nCLAIMS TO CHECK:\n{numbered}"
            try:
                raw     = await _llm(system, user, max_tokens=256)
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
                parsed  = json.loads(cleaned)
                for item in parsed.get("entailments", []):
                    cid = item.get("claim_id", "")
                    val = 1.0 if item.get("entails") else 0.0
                    results[cid] = results.get(cid, 0.0) + val
            except Exception as e:
                logger.warning(f"⚠️ Entailment chunk failed for [{model_id}]: {e}")

    k_minus_1 = len(other_models)
    return {cid: round(total / k_minus_1, 4) for cid, total in results.items()}


# ==========================================
# 9. Step 3 — Factuality Fusion Engine
# ==========================================

def compute_alpha(search_available: bool, consensus_available: bool) -> float:
    """Dynamic α from the fallback matrix."""
    if search_available and consensus_available:
        return 0.70   # Standard Mode
    if not search_available and consensus_available:
        return 0.00   # Search Fallback → 100% consensus
    if search_available and not consensus_available:
        return 1.00   # Single-Model Mode → 100% web search
    return None       # Total Offline → flag as unverifiable


def fuse_scores(
    claim: AtomicClaim,
    s_search: Optional[float],
    s_cons: Optional[float],
) -> VerificationResult:
    """Compute composite score and assign factual status."""
    search_available    = s_search is not None
    consensus_available = s_cons   is not None

    alpha = compute_alpha(search_available, consensus_available)

    if alpha is None:
        # Total Offline Mode
        return VerificationResult(
            claim_id   = claim.claim_id,
            claim_text = claim.decontextualized_claim,
            s_search   = s_search,
            s_cons     = s_cons,
            alpha      = 0.0,
            status     = "unverifiable",
        )

    s_composite = round(alpha * (s_search or 0.0) + (1 - alpha) * (s_cons or 0.0), 4)
    status      = "verified" if s_composite >= TAU else "hallucinated"

    return VerificationResult(
        claim_id    = claim.claim_id,
        claim_text  = claim.decontextualized_claim,
        s_search    = s_search,
        s_cons      = s_cons,
        alpha       = alpha,
        s_composite = s_composite,
        status      = status,
    )


# ==========================================
# 10. Dual Verification Engine Orchestrator
# ==========================================

class DualVerificationEngine:
    """
    Orchestrates the full Phase 2B pipeline for a single model's claim set.
    """

    async def verify_claims(
        self,
        extraction_result: ExtractionResult,
        all_raw_outputs: Dict[str, str],
    ) -> Phase2BResult:
        """
        Run the full dual-verification pipeline for one model's extracted claims.

        Args:
            extraction_result: Phase 2A output for one source model
            all_raw_outputs:   Raw Phase 1 outputs for ALL models (for consensus)

        Returns:
            Phase2BResult with per-claim VerificationResult objects
        """
        start    = time.time()
        model_id = extraction_result.model_id
        claims   = extraction_result.claims

        logger.info(f"🔍 Phase 2B: verifying {len(claims)} claims from [{model_id}]")

        if not claims:
            return Phase2BResult(source_model_id=model_id, latency_sec=0.0)

        # ── Step 1: Generate search queries ──────────────────────────────────
        query_map = await generate_search_queries(claims)

        # ── Step 2: Fetch evidence + NLI + consensus (all concurrent) ────────
        evidence_task   = asyncio.create_task(fetch_evidence_batch(query_map))
        consensus_task  = asyncio.create_task(
            cross_model_consensus(claims, model_id, all_raw_outputs)
        )

        evidence_map, cons_scores = await asyncio.gather(evidence_task, consensus_task)

        # ── Step 3: Batch NLI ─────────────────────────────────────────────────
        nli_scores = await batch_nli_verification(claims, evidence_map)

        # ── Step 4: Fuse scores per claim ─────────────────────────────────────
        verifications: List[VerificationResult] = []
        for claim in claims:
            s_search_tuple = nli_scores.get(claim.claim_id)
            s_search       = s_search_tuple[0] if s_search_tuple else None
            search_label   = s_search_tuple[1] if s_search_tuple else None
            s_cons         = cons_scores.get(claim.claim_id)

            vr             = fuse_scores(claim, s_search, s_cons)
            vr.search_label       = search_label
            vr.evidence_snippets  = evidence_map.get(claim.claim_id, [])
            verifications.append(vr)

        verified     = sum(1 for v in verifications if v.status == "verified")
        hallucinated = sum(1 for v in verifications if v.status == "hallucinated")
        unverifiable = sum(1 for v in verifications if v.status == "unverifiable")

        return Phase2BResult(
            source_model_id    = model_id,
            verifications      = verifications,
            total_claims       = len(verifications),
            verified_count     = verified,
            hallucinated_count = hallucinated,
            unverifiable_count = unverifiable,
            latency_sec        = round(time.time() - start, 3),
        )

    async def verify_all(
        self,
        extraction_results: Dict[str, ExtractionResult],
        all_raw_outputs: Dict[str, str],
    ) -> Dict[str, Phase2BResult]:
        """
        Concurrently verify claims from all models.

        Args:
            extraction_results: Phase 2A output dict {model_id: ExtractionResult}
            all_raw_outputs:    Phase 1 raw outputs {model_id: raw_text}

        Returns:
            Dict {model_id: Phase2BResult}
        """
        tasks = {
            model_id: asyncio.create_task(
                self.verify_claims(result, all_raw_outputs)
            )
            for model_id, result in extraction_results.items()
            if result.status == "success"
        }
        return {mid: await task for mid, task in tasks.items()}


# ==========================================
# 11. Pretty Print Utility
# ==========================================

_STATUS_ICON = {"verified": "✅", "hallucinated": "❌", "unverifiable": "⚠️"}

def print_verification_results(results: Dict[str, Phase2BResult]) -> None:
    print("\n" + "=" * 80)
    print("PHASE 2B: DUAL VERIFICATION ENGINE RESULTS")
    print("=" * 80)

    for model_id, result in results.items():
        print(f"\n{'─' * 80}")
        print(f"Source Model : {model_id}")
        print(f"Latency      : {result.latency_sec:.2f}s")
        print(f"Claims       : {result.total_claims}  |  "
              f"✅ {result.verified_count}  ❌ {result.hallucinated_count}  ⚠️ {result.unverifiable_count}")
        print(f"{'─' * 80}")

        for v in result.verifications:
            icon = _STATUS_ICON.get(v.status, "?")
            s_search_str = f"{v.s_search:.2f}" if v.s_search is not None else "N/A"
            s_cons_str   = f"{v.s_cons:.2f}"   if v.s_cons   is not None else "N/A"
            s_comp_str   = f"{v.s_composite:.2f}" if v.s_composite is not None else "N/A"

            print(
                f"  {icon} [{v.claim_id}]  "
                f"S_search={s_search_str} ({v.search_label or 'N/A'})  "
                f"S_cons={s_cons_str}  α={v.alpha:.2f}  "
                f"S={s_comp_str}  → {v.status.upper()}"
            )
            print(f"     Claim: {v.claim_text[:100]}")

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    total_v = sum(r.verified_count     for r in results.values())
    total_h = sum(r.hallucinated_count for r in results.values())
    total_u = sum(r.unverifiable_count for r in results.values())
    total   = total_v + total_h + total_u
    print(f"Total Claims     : {total}")
    print(f"✅ Verified      : {total_v}")
    print(f"❌ Hallucinated  : {total_h}")
    print(f"⚠️  Unverifiable  : {total_u}")
    print("=" * 80 + "\n")


# ==========================================
# 12. Main — End-to-End Demo
# ==========================================

async def main():
    print("\n" + "=" * 80)
    print("PHASE 2B: DUAL VERIFICATION ENGINE")
    print("=" * 80)

    # ── Phase 2A: extract atomic claims ──────────────────────────────────────
    print("\n[Step 1] Running Phase 2A extraction...")
    extractor        = AtomicPropositionExtractor()
    extraction_results = await extractor.extract_claims_batch(SAMPLE_PHASE1_OUTPUTS)

    # ── Phase 2B: verify claims ───────────────────────────────────────────────
    print("\n[Step 2] Running Phase 2B dual verification...")
    engine  = DualVerificationEngine()
    results = await engine.verify_all(extraction_results, SAMPLE_PHASE1_OUTPUTS)

    print_verification_results(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())

"""
Phase 2A: Atomic Proposition Extraction and Decontextualization

Receives raw LLM outputs from Phase 1 and transforms them into structured,
verifiable atomic claims via:
  - Pronominal resolution & decontextualization
  - Temporal anchoring
  - Atomic decomposition
  - Filler & non-factual filtering
"""

import os
import sys
import asyncio
import time
import json
import logging
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from litellm import acompletion

# ==========================================
# 1. Environment Setup
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")

if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
    logger.error("❌ No API keys found. Set OPENROUTER_API_KEY or GEMINI_API_KEY in .env")
    sys.exit(1)

EXTRACTION_MODELS_FALLBACK = [
    "gemini/gemini-3.6-flash",
    "gemini/gemini-2.5-flash-preview-05-20",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
]

MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0


# ==========================================
# 2. Pydantic Schemas
# ==========================================

class AtomicClaim(BaseModel):
    claim_id:               str = Field(..., description="Unique ID e.g. 'model_a_c1'")
    decontextualized_claim: str = Field(..., description="Self-contained, fully resolved atomic claim")
    token_count:            int = Field(..., description="Approximate token count of the claim")


class ExtractionResult(BaseModel):
    model_id:     str               = Field(..., description="Source LLM model identifier")
    raw_text:     str               = Field(..., description="Original raw text from Phase 1")
    claims:       List[AtomicClaim] = Field(default_factory=list)
    total_claims: int               = Field(0)
    latency_sec:  float             = Field(0.0)
    status:       str               = Field("failed")
    error:        Optional[str]     = None


# ==========================================
# 3. Prompts
# ==========================================

SYSTEM_PROMPT = """
You are an expert computational linguist and core engine for an automated hallucination detection system. Your sole function is to take raw model response text, resolve all contextual ambiguities (decontextualization), split complex sentences into minimal self-contained atomic claims, and output them as a clean JSON object.

## STEP 1: DECONTEXTUALIZE
- Replace ALL pronouns (he, she, it, they, his, her, its, their, this, that, these, those) with explicit named entities.
- Convert relative time references ("last year", "recently", "two decades ago", "currently") into explicit years or time-anchored assertions based on context.
- Every claim MUST be 100% self-contained — readable in isolation without surrounding context.

## STEP 2: ATOMIC DECOMPOSITION
- Each claim must contain EXACTLY ONE subject-predicate-object assertion.
- Split compound sentences joined by "and", "but", "while", "which" into separate claims.
- Do NOT break indivisible single facts (e.g. "The Treaty of Versailles was signed in 1919" stays as one claim).

## FACTUAL FIDELITY
- Preserve original meaning exactly. Do NOT add, assume, extrapolate, or correct facts — even if the source text contains errors or hallucinations.
- Retain exact numbers, dates, locations, and proper nouns.

## WHAT TO EXCLUDE
- Conversational filler: "Sure, here is...", "Based on my knowledge...", "I hope this helps!"
- Structural transitions: "Here are three main points:", "First of all,"
- Pure opinions or speculation with no verifiable component: "This is fascinating.", "It is important to be careful."

## EDGE CASES
- Lists: Extract a separate claim per list item.
- Conditionals: Keep intact if splitting destroys the condition.
- Quotes/Attributions: Keep attribution connected (e.g. "NASA stated that the launch was successful.").
- Filler-only input: Return {"atomic_claims": []}.
- Negations: Preserve exactly — "X did NOT cause Y" is a valid atomic claim.

## CRITICAL NEGATIVE CONSTRAINTS
- DO NOT hallucinate, infer, or add any facts not explicitly stated in the input text.
- DO NOT correct false statements — extract them exactly as stated, even if wrong.
- DO NOT output anything outside the JSON structure below.

## OUTPUT FORMAT
Return ONLY a valid raw JSON object. No markdown, no preamble, no explanation.

{
  "atomic_claims": [
    "Fully decontextualized atomic claim 1.",
    "Fully decontextualized atomic claim 2."
  ]
}

## EXAMPLES

Input: "Hello! Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. He later founded the W3C. Last year, he spoke at a conference about web privacy. I hope this helps!"
Output: {"atomic_claims": ["Tim Berners-Lee invented the World Wide Web in 1989.", "Tim Berners-Lee was working at CERN when he invented the World Wide Web.", "Tim Berners-Lee founded the World Wide Web Consortium (W3C).", "Tim Berners-Lee spoke at a conference about web privacy in 2025."]}

Input: "The James Webb Space Telescope (JWST) was launched in December 2021. Operating at the L2 Lagrange point, it has captured images of early galaxies and detected atmospheric components on distant exoplanets."
Output: {"atomic_claims": ["The James Webb Space Telescope (JWST) was launched in December 2021.", "The James Webb Space Telescope operates at the L2 Lagrange point.", "The James Webb Space Telescope has captured images of early galaxies.", "The James Webb Space Telescope has detected atmospheric components on distant exoplanets."]}

Input: "Sure thing! I can certainly assist you with that request today."
Output: {"atomic_claims": []}
""".strip()

USER_PROMPT_TEMPLATE = """Analyze the raw input text below. Extract all atomic factual claims strictly following the rules above. Output ONLY the finalized JSON object.

<input_text>
{raw_text}
</input_text>"""


# ==========================================
# 4. Core Extractor Class
# ==========================================

class AtomicPropositionExtractor:
    """
    Extracts atomic, decontextualized propositions from raw LLM outputs.
    Processes multiple model outputs concurrently via asyncio.
    """

    # ------------------------------------------
    # Internal: call LLM with retry + fallback
    # ------------------------------------------
    async def _call_with_retry(self, model_id: str, raw_text: str) -> str:
        """Call extraction LLM with exponential backoff and model fallback."""
        user_message = USER_PROMPT_TEMPLATE.format(raw_text=raw_text)

        for model_name in EXTRACTION_MODELS_FALLBACK:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(f"Extracting [{model_id}] via {model_name} (attempt {attempt})")

                    api_key_env = "OPENROUTER_API_KEY" if "openrouter" in model_name else "GEMINI_API_KEY"
                    os.environ[api_key_env] = os.getenv(api_key_env, "")

                    params = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        "max_tokens": 4096,
                    }
                    if "gemini" not in model_name:
                        params["temperature"] = 0.0
                        params["response_format"] = {"type": "json_object"}

                    response = await acompletion(**params)
                    content  = response["choices"][0]["message"]["content"]
                    logger.info(f"✅ Extraction complete for [{model_id}] via {model_name}")
                    return content

                except Exception as e:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning(f"⚠️ {model_name} attempt {attempt} failed: {type(e).__name__}. Retrying in {wait:.1f}s...")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"⚠️ {model_name} exhausted retries, trying next fallback...")
                        break

        raise RuntimeError(f"All extraction models failed for [{model_id}]")

    # ------------------------------------------
    # Internal: parse LLM JSON response
    # ------------------------------------------
    def _parse_response(self, model_id: str, raw_text: str, content: str) -> ExtractionResult:
        """Parse and validate JSON response into ExtractionResult."""
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
            parsed  = json.loads(cleaned)

            raw_claims = parsed.get("atomic_claims", [])

            claims = [
                AtomicClaim(
                    claim_id               = f"{model_id}_c{i+1}",
                    decontextualized_claim = claim.strip(),
                    token_count            = len(claim.split())
                )
                for i, claim in enumerate(raw_claims)
                if isinstance(claim, str) and claim.strip()
            ]

            return ExtractionResult(
                model_id     = model_id,
                raw_text     = raw_text,
                claims       = claims,
                total_claims = len(claims),
                status       = "success"
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"❌ Parse failure for [{model_id}]: {e}")
            return ExtractionResult(
                model_id = model_id,
                raw_text = raw_text,
                status   = "parse_failed",
                error    = f"ParseError: {str(e)} | Raw: {content[:300]}"
            )

    # ------------------------------------------
    # Internal: process a single model output
    # ------------------------------------------
    async def _extract_single(self, model_id: str, raw_text: str) -> ExtractionResult:
        """Extract atomic claims from a single model's raw output."""
        if not raw_text or not raw_text.strip():
            logger.warning(f"⚠️ Empty input for [{model_id}], skipping.")
            return ExtractionResult(
                model_id=model_id, raw_text=raw_text,
                claims=[], total_claims=0, status="success", error="Empty input"
            )

        start = time.time()
        try:
            content = await self._call_with_retry(model_id, raw_text)
            result  = self._parse_response(model_id, raw_text, content)
            result.latency_sec = round(time.time() - start, 3)
            return result
        except Exception as e:
            return ExtractionResult(
                model_id=model_id, raw_text=raw_text,
                status="failed", error=str(e),
                latency_sec=round(time.time() - start, 3)
            )

    # ------------------------------------------
    # Public: batch process all model outputs
    # ------------------------------------------
    async def extract_claims_batch(self, raw_outputs: Dict[str, str]) -> Dict[str, ExtractionResult]:
        """
        Concurrently extract atomic claims from multiple model outputs.

        Args:
            raw_outputs: Dict mapping model_id -> raw text from Phase 1

        Returns:
            Dict mapping model_id -> ExtractionResult
        """
        logger.info(f"Starting Phase 2A batch extraction for {len(raw_outputs)} model outputs")

        tasks = {
            model_id: asyncio.create_task(self._extract_single(model_id, raw_text))
            for model_id, raw_text in raw_outputs.items()
        }

        results = {}
        for model_id, task in tasks.items():
            results[model_id] = await task

        successful   = sum(1 for r in results.values() if r.status == "success")
        total_claims = sum(r.total_claims for r in results.values())
        logger.info(f"Phase 2A complete: {successful}/{len(raw_outputs)} successful | {total_claims} total claims extracted")

        return results


# ==========================================
# 5. Pretty Print Utility
# ==========================================

def print_extraction_results(results: Dict[str, ExtractionResult]) -> None:
    print("\n" + "=" * 80)
    print("PHASE 2A: ATOMIC PROPOSITION EXTRACTION RESULTS")
    print("=" * 80)

    for model_id, result in results.items():
        print(f"\n{'─' * 80}")
        print(f"Source Model : {model_id}")
        print(f"Status       : {result.status}")
        print(f"Latency      : {result.latency_sec:.2f}s")
        print(f"Total Claims : {result.total_claims}")
        print(f"{'─' * 80}")

        if result.status == "success" and result.claims:
            for claim in result.claims:
                print(f"  [{claim.claim_id}] {claim.decontextualized_claim}")
        elif result.error:
            print(f"  Error: {result.error}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total      = sum(r.total_claims for r in results.values())
    successful = sum(1 for r in results.values() if r.status == "success")
    print(f"Models Processed : {successful}/{len(results)}")
    print(f"Total Claims     : {total}")
    print("=" * 80 + "\n")


# ==========================================
# 6. Main — Demo with sample Phase 1 outputs
# ==========================================

SAMPLE_PHASE1_OUTPUTS = {
    "gemini-3.6-flash": """
        Sure! Here's a brief overview. The internet began as ARPANET, a project funded by the
        US Department of Defense in 1969. It was initially used to connect four universities.
        Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. He published
        his proposal in March of that year. Recently, the internet has grown to connect over
        5 billion users globally. It now supports streaming, e-commerce, and social media at scale.
        Hope this helps!
    """,
    "qwen3.8-flash": """
        The history of the internet is fascinating. ARPANET was created in 1969 and it was
        funded by DARPA. The first message sent over it crashed the system. Vinton Cerf and
        Bob Kahn developed TCP/IP in 1974, which became the foundational protocol.
        They are often called the fathers of the internet. The Web went public in 1991
        and it changed everything. Last year, global internet traffic exceeded 400 exabytes per month.
        In conclusion, the internet is one of humanity's greatest achievements.
    """
}


async def main():
    print("\n" + "=" * 80)
    print("PHASE 2A: ATOMIC PROPOSITION EXTRACTION AND DECONTEXTUALIZATION")
    print("=" * 80)

    extractor = AtomicPropositionExtractor()
    results   = await extractor.extract_claims_batch(SAMPLE_PHASE1_OUTPUTS)
    print_extraction_results(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())

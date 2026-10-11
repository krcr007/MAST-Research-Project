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
from datetime import datetime
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
MISTRAL_API_KEY    = os.getenv("MISTRAL_API_KEY")

if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
    logger.error("❌ No API keys found. Set OPENROUTER_API_KEY or GEMINI_API_KEY in .env")
    sys.exit(1)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

EXTRACTION_MODELS = [
    "groq/qwen/qwen3.8-27b",
    "groq/allam-2-7b",
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
You are the Atomic Proposition Extraction Engine in an automated hallucination-detection and factuality-evaluation system.

Your sole task is to transform raw model-generated text into a set of minimal, self-contained, independently verifiable atomic factual propositions.

You are an EXTRACTOR, not a fact-checker, summarizer, reasoner, or knowledge generator.

Your output must represent ONLY factual content explicitly expressed in the input text, after resolving linguistic dependencies that are necessary to make each proposition self-contained.

============================================================
1. FORMAL OBJECTIVE
============================================================

Given raw model output Y_m and a reference timestamp t0:

    f_extract(Y_m, t0) -> C(Y_m) = {c1, c2, ..., cN}

where every ci is an atomic factual proposition.

An atomic proposition is a minimal factual assertion containing exactly one independently verifiable proposition.

Each extracted proposition MUST satisfy all three core axioms:

A. ATOMICITY
   The proposition expresses exactly one independently verifiable fact.

B. CONTEXT-INDEPENDENCE
   The proposition can be understood and evaluated without the surrounding text.

C. TEMPORAL ABSOLUTENESS
   Relative temporal expressions are converted to explicit temporal references whenever the reference can be determined from the supplied context and t0.

============================================================
2. CRITICAL PRINCIPLE: EXTRACT, DO NOT GENERATE
============================================================

NEVER introduce factual information that is not explicitly stated or linguistically recoverable from the input.

You MAY:
- resolve pronouns;
- resolve clear nominal references;
- resolve clear aliases;
- expand abbreviations when explicitly established by the text;
- convert relative temporal expressions into absolute temporal references when determinable;
- split compound factual assertions;
- preserve attribution;
- preserve negation;
- preserve modality;
- preserve uncertainty;
- preserve numerical values and qualifiers.

You MUST NOT:
- infer unstated facts;
- generate logical consequences;
- generate commonsense consequences;
- add background knowledge;
- add facts from your own knowledge;
- correct factual errors;
- fact-check the input;
- expand implications into new claims;
- transform an event into a resulting state unless the resulting state is explicitly stated;
- increase the precision of dates, quantities, locations, identities, or relationships beyond what the input provides.

IMPORTANT:

"Elon Musk renamed Twitter to X in 2023."

does NOT automatically justify:

"Twitter is named X."

unless the input explicitly states that Twitter is named X.

The extraction engine performs decomposition, not entailment generation.

============================================================
3. ATOMICITY
============================================================

Each output proposition MUST contain exactly one independently verifiable factual assertion.

Split:

"The telescope was launched in 2021 and reached orbit in 2022."

into:

"The telescope was launched in 2021."
"The telescope reached orbit in 2022."

Split coordinated predicates:

"Apple developed and released the iPhone."

into:

"Apple developed the iPhone."
"Apple released the iPhone."

Split coordinated objects when they represent independently verifiable facts:

"Einstein studied physics and mathematics."

into:

"Einstein studied physics."
"Einstein studied mathematics."

However, DO NOT split a phrase when doing so changes its meaning or creates an unsupported proposition.

For example:

"Einstein was born in Ulm, Germany."

remains one proposition.

============================================================
4. SELF-CONTAINMENT AND ANAPHORA RESOLUTION
============================================================

Every proposition MUST be understandable in isolation.

Resolve:

- he
- she
- they
- it
- him
- her
- them
- his
- hers
- their
- its
- this
- that
- these
- those
- the former
- the latter
- the company
- the organization
- the platform
- the country
- the city
- the telescope
- the system
- other clearly identifiable nominal references

into explicit entities ONLY when the antecedent is unambiguous from the input.

Example:

Input:
"OpenAI developed ChatGPT. It was released in November."

Output:
"OpenAI developed ChatGPT."
"ChatGPT was released in November."

Do NOT resolve an ambiguous reference by guessing.

If:

"Google acquired a startup. It later announced layoffs."

does not provide enough information to determine whether "it" refers to Google or the startup, DO NOT invent a resolution.

In such cases, preserve the ambiguity only if a self-contained formulation is possible without guessing. Otherwise, omit the proposition rather than hallucinating an antecedent.

============================================================
5. NOMINAL ANAPHORA
============================================================

Resolve descriptive references when the antecedent is unambiguous.

Example:

"Google released Gemini. The technology company reported strong performance."

Output:

"Google released Gemini."
"Google reported strong performance."

Do NOT add information not contained in the source.

"The technology company" may resolve to "Google" only because the preceding context establishes the identity.

============================================================
6. TEMPORAL ANCHORING
============================================================

The input is evaluated relative to an explicitly supplied reference timestamp:

t0 = {reference_timestamp}

Convert relative temporal expressions into explicit temporal references when determinable.

Examples:

"last year" + t0 = 2026
-> "in 2025"

"next year" + t0 = 2026
-> "in 2027"

"two decades ago" + t0 = 2026
-> "around 2006"

"currently" + t0 = 2026
-> "as of 2026"

"the following year" must be resolved relative to the relevant previously established date.

Example:

"Twitter was purchased in October 2022. It was renamed the following year."

Output:

"Twitter was purchased in October 2022."
"Twitter was renamed in 2023."

IMPORTANT TEMPORAL PRECISION RULE:

NEVER invent precision.

If the source says:

"two months ago"

and t0 only allows determination of a month but not an exact date, do not invent an exact day.

If the source says:

"recently"

and no reliable temporal anchor can determine a specific period, preserve "recently" rather than inventing a year.

If a temporal expression is inherently vague, retain the original temporal qualifier.

============================================================
7. TEMPORAL REFERENCE HIERARCHY
============================================================

When resolving relative time, use this priority:

1. Explicit date in the same sentence.
2. Explicit date in the immediately relevant preceding context.
3. Explicit reference timestamp t0.
4. Do not infer a more precise date.

Example:

"Elon Musk purchased Twitter in October 2022. He renamed it the following year."

"the following year" -> 2023.

But:

"Elon Musk recently discussed AI."

If no reliable temporal interval is supplied:

"Elon Musk discussed AI recently."

Do NOT convert "recently" into an invented year.

============================================================
8. PRESERVE SEMANTIC QUALIFIERS
============================================================

Never remove or strengthen:

- negation
- modality
- uncertainty
- attribution
- evidentiality
- probability
- frequency
- quantity
- scope
- temporal qualifiers
- causal qualifiers
- conditionality

Examples:

"NASA said that the launch was successful."

-> "NASA said that the launch was successful."

NOT:

"The launch was successful."

The attribution is part of the factual proposition.

Similarly:

"The company may launch the product in 2027."

must NOT become:

"The company will launch the product in 2027."

"The company did not launch the product."

must remain negative.

"The company allegedly violated the regulation."

must preserve "allegedly."

============================================================
9. ATTRIBUTED CLAIMS
============================================================

Claims made by a source, person, organization, study, report, or publication must retain their attribution.

Example:

"NASA stated that the spacecraft entered orbit."

Output:

"NASA stated that the spacecraft entered orbit."

NOT:

"The spacecraft entered orbit."

The latter removes the epistemic/attributional structure and changes the proposition.

For:

"According to NASA, the spacecraft entered orbit."

output:

"According to NASA, the spacecraft entered orbit."

============================================================
10. QUOTATIONS
============================================================

Preserve the attribution of quoted factual claims.

Example:

"NASA said, 'The spacecraft entered orbit.'"

Output:

"NASA said that the spacecraft entered orbit."

Do not treat a quoted claim as independently verified fact.

============================================================
11. NEGATION
============================================================

Negation MUST be preserved exactly.

Input:

"The Apollo 11 mission did not land on Mars."

Output:

"The Apollo 11 mission did not land on Mars."

Do NOT convert it into:

"The Apollo 11 mission landed on Mars."

============================================================
12. CONDITIONALS
============================================================

Preserve conditions when removing them would change the proposition.

Input:

"If approved, the drug will enter clinical trials in 2027."

Output:

"If the drug is approved, the drug will enter clinical trials in 2027."

Do NOT output:

"The drug will enter clinical trials in 2027."

because the conditional dependency is semantically essential.

============================================================
13. CAUSAL RELATIONSHIPS
============================================================

Preserve explicit causal relationships.

Input:

"Heavy rainfall caused flooding in Mumbai."

Output:

"Heavy rainfall caused flooding in Mumbai."

Do not decompose this into:

"Heavy rainfall occurred."
"Flooding occurred in Mumbai."

unless those propositions are independently and explicitly stated.

The causal relation itself is the factual proposition.

============================================================
14. RELATIVE CLAUSES
============================================================

Decompose relative clauses when they represent independently verifiable facts.

Input:

"Barack Obama, who was born in Hawaii, became the 44th president of the United States."

Output:

"Barack Obama was born in Hawaii."
"Barack Obama became the 44th president of the United States."

Do not create additional facts beyond those expressed.

============================================================
15. APPOSITIONS
============================================================

Appositive factual information may be extracted as a separate proposition when it represents an independently verifiable fact.

Input:

"Barack Obama, the 44th president of the United States, was born in Hawaii."

Output:

"Barack Obama was the 44th president of the United States."
"Barack Obama was born in Hawaii."

============================================================
16. LISTS
============================================================

Extract independently verifiable list items separately.

Input:

"Python supports NumPy, Pandas, and Matplotlib."

Output:

"Python supports NumPy."
"Python supports Pandas."
"Python supports Matplotlib."

However, preserve the original semantic relationship if the list is functioning as a single indivisible proposition.

============================================================
17. NUMBERS AND QUANTITIES
============================================================

Preserve numerical information exactly.

Do NOT:
- round numbers;
- convert units unless required for self-containment;
- estimate missing quantities;
- replace exact numbers with approximations;
- replace approximations with exact values.

Input:

"The company raised $2.7 billion."

Output:

"The company raised $2.7 billion."

NOT:

"The company raised approximately $3 billion."

============================================================
18. NAMES, ALIASES, AND ABBREVIATIONS
============================================================

Preserve proper nouns exactly as given.

If an abbreviation is explicitly defined:

"James Webb Space Telescope (JWST) was launched in 2021."

The following are acceptable:

"The James Webb Space Telescope (JWST) was launched in 2021."

"The James Webb Space Telescope was launched in 2021."

Use the canonical entity name consistently when doing so does not alter meaning.

Do not expand an abbreviation using outside knowledge unless the expansion is explicitly established in the input.

============================================================
19. ENTITY CONSISTENCY
============================================================

When a proposition refers to the same entity through different surface forms, use a consistent canonical entity name when the identity is explicitly established.

Example:

"OpenAI developed ChatGPT. The AI company released ChatGPT in 2022."

Output:

"OpenAI developed ChatGPT."
"OpenAI released ChatGPT in 2022."

============================================================
20. DEDUPLICATION
============================================================

Remove exact duplicates and semantically redundant propositions.

Example:

"OpenAI developed ChatGPT. ChatGPT was developed by OpenAI."

Output only:

"OpenAI developed ChatGPT."

However, DO NOT remove propositions merely because they are related.

Example:

"OpenAI developed ChatGPT."
"OpenAI released ChatGPT in 2022."

These are distinct propositions and both must remain.

============================================================
21. NO SUBSUMPTION-BASED FACT GENERATION
============================================================

Do not generate a proposition merely because it is logically implied by another proposition.

Input:

"Elon Musk renamed Twitter to X in 2023."

Output:

"Elon Musk renamed Twitter to X in 2023."

Do NOT add:

"Twitter is named X."

Do NOT add:

"Elon Musk owns X."

Do NOT add:

"X was formerly called Twitter."

unless those propositions are explicitly stated.

============================================================
22. PRESERVE EVENT VS STATE DISTINCTIONS
============================================================

Do not convert an event into a resulting state.

"Elon Musk renamed Twitter to X."

does not automatically become:

"Twitter is named X."

"Apple acquired Company A."

does not automatically become:

"Apple owns Company A."

unless explicitly stated.

============================================================
23. OPINIONS, EVALUATIONS, AND SPECULATION
============================================================

Exclude pure non-factual conversational material.

Exclude:

"That is fascinating."
"This is an excellent product."
"I think this is important."

However, preserve evaluative claims when they contain a factual assertion that can be independently evaluated.

Input:

"According to the report, the system achieved state-of-the-art accuracy."

Output:

"The report stated that the system achieved state-of-the-art accuracy."

Do not independently assert the system achieved state-of-the-art accuracy unless the source does so without attribution.

============================================================
24. MODALITY
============================================================

Do not convert possibility, necessity, intention, prediction, or likelihood into factual certainty.

Input:

"The company plans to launch the product in 2027."

Output:

"The company plans to launch the product in 2027."

NOT:

"The company will launch the product in 2027."

Input:

"The drug could reduce mortality."

Output:

"The drug could reduce mortality."

NOT:

"The drug reduces mortality."

============================================================
25. QUESTIONS
============================================================

Questions do not automatically constitute factual claims.

Input:

"Did Apple acquire the company in 2020?"

Output:

{"atomic_claims": []}

unless the question itself contains an explicit factual assertion independent of the interrogative structure.

============================================================
26. INSTRUCTIONS, COMMANDS, AND HYPOTHETICALS
============================================================

Do not extract instructions as factual claims.

Input:

"Install Python and run the script."

Output:

{"atomic_claims": []}

Hypothetical statements should be retained only when they contain a factual proposition relevant to the stated hypothetical condition.

============================================================
27. FILLER AND DISCOURSE
============================================================

Remove:

- greetings;
- conversational filler;
- apologies;
- meta-commentary;
- formatting instructions;
- structural transitions;
- conclusions that contain no factual proposition;
- statements such as "Here are the results."

Example:

"Sure! Here are the three main findings. The company was founded in 2010."

Output:

"The company was founded in 2010."

============================================================
28. TABLES AND STRUCTURED TEXT
============================================================

Treat factual entries in:

- tables;
- bullet lists;
- numbered lists;
- headings containing factual statements;
- captions;

as ordinary textual propositions.

Do not extract purely structural labels.

============================================================
29. UNCERTAINTY AND AMBIGUITY
============================================================

When ambiguity can be resolved from the input, resolve it.

When ambiguity cannot be resolved without external knowledge or guessing:

DO NOT GUESS.

Never use world knowledge to choose between two possible antecedents.

Accuracy of extraction is more important than maximizing the number of extracted propositions.

============================================================
30. EXTERNAL KNOWLEDGE
============================================================

External world knowledge MUST NOT be used to add, correct, validate, or complete propositions.

The input is the only source of factual content.

The supplied timestamp t0 is the only external variable allowed for temporal anchoring.

============================================================
31. EXTRACTION ORDER
============================================================

Perform the following internal pipeline:

1. Sentence segmentation.
2. Clause and predicate identification.
3. Coreference and anaphora resolution.
4. Nominal reference resolution.
5. Temporal anchoring.
6. Atomic proposition decomposition.
7. Semantic qualifier preservation.
8. Duplicate detection.
9. Unsupported-inference filtering.
10. Final self-containment validation.

Before producing the output, verify every proposition against the original input.

For every proposition ask:

- Is this explicitly stated?
- If not explicitly stated, is it only a necessary linguistic resolution?
- Did I introduce any external knowledge?
- Did I remove a qualifier?
- Did I change the modality?
- Did I change the polarity?
- Did I increase temporal precision?
- Does the proposition contain exactly one independently verifiable fact?
- Can the proposition be understood in isolation?

If any answer indicates that the proposition adds unsupported information, remove or revise it.

============================================================
32. ATOMICITY VALIDATION
============================================================

For every candidate proposition:

    Atomic(c) = 1

only if:

    NumberOfIndependentVerifiableFacts(c) = 1

AND

    ContextDependencies(c) = 0

AND

    UnsupportedInformation(c) = 0

AND

    MeaningChanged(c) = 0

============================================================
33. OUTPUT CONTRACT
============================================================

Return ONLY valid JSON.

Do not output:
- Markdown
- code fences
- explanations
- reasoning
- comments
- confidence scores
- claim IDs
- additional fields

The exact schema is:

{
  "atomic_claims": [
    "Atomic proposition 1.",
    "Atomic proposition 2."
  ]
}

If no valid factual propositions are present:

{
  "atomic_claims": []
}

The JSON must be syntactically valid and parseable by a standard JSON parser.

============================================================
34. FINAL QUALITY REQUIREMENT
============================================================

The extraction system must prefer:

PRECISION > RECALL

when the alternative is introducing unsupported information.

It is better to omit an ambiguous proposition than to hallucinate its meaning.

It is better to preserve a slightly complex proposition than to create unsupported subclaims.

The final output must be a faithful atomic decomposition of the input, not a rewritten or improved version of the input.
""".strip()

USER_PROMPT_TEMPLATE = """
Extract atomic factual propositions from the raw model output below.

REFERENCE TIMESTAMP:
{reference_timestamp}

Use the reference timestamp ONLY for resolving relative temporal expressions such as:
- today
- yesterday
- last year
- next year
- currently
- recently
- two years ago
- the following year

Do not use external world knowledge to add, verify, correct, or complete facts.

IMPORTANT:
- Extract, do not generate.
- Resolve clear pronouns and nominal references.
- Split compound factual assertions.
- Preserve attribution, negation, modality, uncertainty, conditions, causality, quantities, and temporal qualifiers.
- Do not infer unstated consequences.
- Do not convert events into resulting states.
- Do not increase temporal precision beyond what the input supports.
- Do not correct factual errors.
- Remove filler and non-factual conversational content.
- Deduplicate only genuinely redundant propositions.

Return ONLY the JSON object specified by the system instructions.

<raw_model_output>
{raw_text}
</raw_model_output>
"""


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
    async def _call_with_retry(self, model_id: str, raw_text: str, extractor_model: Optional[str] = None) -> str:
        """Call extraction LLM with exponential backoff and model fallback."""
        reference_timestamp = datetime.now().strftime("%Y-%m-%d")
        user_message = (
            USER_PROMPT_TEMPLATE
            .replace("{reference_timestamp}", reference_timestamp)
            .replace("{raw_text}", raw_text or "")
        )

        models_to_try = [extractor_model] if extractor_model else EXTRACTION_MODELS
        for model_name in models_to_try:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(f"Extracting [{model_id}] via {model_name} (attempt {attempt})")

                    api_key_env = (
                        "GROQ_API_KEY" if "groq" in model_name
                        else "OPENROUTER_API_KEY" if "openrouter" in model_name
                        else "GEMINI_API_KEY"
                    )
                    os.environ[api_key_env] = os.getenv(api_key_env, "")

                    params = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.0,
                    }

                    response = await acompletion(**params)
                    content  = response["choices"][0]["message"]["content"] or ""
                    if not content.strip():
                        raise ValueError("Empty content returned by model")
                    logger.info(f"✅ Extraction complete for [{model_id}] via {model_name}")
                    return content

                except Exception as e:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning(f"⚠️ {model_name} attempt {attempt} failed: {type(e).__name__}: {str(e)[:300]}. Retrying in {wait:.1f}s...")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(wait)

        raise RuntimeError(f"All extraction models failed for [{model_id}]")

    # ------------------------------------------
    # Internal: parse LLM JSON response
    # ------------------------------------------
    def _parse_response(self, model_id: str, raw_text: str, content: str) -> ExtractionResult:
        """Parse and validate JSON response into ExtractionResult."""
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip(), flags=re.MULTILINE)
            parsed  = json.loads(cleaned)
            logger.info(f"Phase 2A raw parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")

            # Robustly find atomic_claims regardless of extra fields returned by model
            if isinstance(parsed, dict):
                raw_claims = parsed.get("atomic_claims") or parsed.get("claims") or []
            elif isinstance(parsed, list):
                raw_claims = parsed
            else:
                raw_claims = []

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
    async def _extract_single(self, model_id: str, raw_text: str, extractor_model: Optional[str] = None) -> ExtractionResult:
        """Extract atomic claims from a single model's raw output."""
        raw_text = (raw_text or "")[:12000]
        if not raw_text.strip():
            logger.warning(f"⚠️ Empty input for [{model_id}], skipping.")
            return ExtractionResult(
                model_id=model_id, raw_text=raw_text,
                claims=[], total_claims=0, status="success", error="Empty input"
            )

        start = time.time()
        try:
            content = await self._call_with_retry(model_id, raw_text, extractor_model)
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
    async def extract_claims_batch(self, raw_outputs: Dict[str, str], extractor_map: Optional[Dict[str, str]] = None) -> Dict[str, ExtractionResult]:
        """
        Concurrently extract atomic claims from multiple model outputs.

        Args:
            raw_outputs:   Dict mapping model_id -> raw text from Phase 1
            extractor_map: Optional dict mapping model_id -> extractor model name
                           (cross-model extraction). Falls back to default if None.

        Returns:
            Dict mapping model_id -> ExtractionResult
        """
        logger.info(f"Starting Phase 2A batch extraction for {len(raw_outputs)} model outputs")

        tasks = {
            model_id: asyncio.create_task(
                self._extract_single(model_id, raw_text, extractor_map.get(model_id) if extractor_map else None)
            )
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

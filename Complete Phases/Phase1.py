"""
Phase 1: Multi-Model Parallel Ingestion

This module implements parallel API routing to multiple LLM models using asyncio and LiteLLM.
It takes a user prompt and simultaneously queries multiple models to generate raw text outputs.

Models Supported:
- GPT-4o (via OpenAI API)
- Claude 3.5 Sonnet (via Anthropic API)
- Gemini 1.5 Pro (via Google API)
"""

import os
import sys
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from dotenv import load_dotenv
from litellm import acompletion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. Load .env file
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")

load_dotenv(env_path)


# ==========================================
# 2. Read API keys
# ==========================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ==========================================
# 3. Check API keys
# ==========================================

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print(f".env path: {env_path}")

print(
    "OpenAI API Key:",
    "FOUND" if OPENAI_API_KEY else "NOT FOUND"
)

print(
    "Anthropic API Key:",
    "FOUND" if ANTHROPIC_API_KEY else "NOT FOUND"
)

print(
    "Gemini API Key:",
    "FOUND" if GEMINI_API_KEY else "NOT FOUND"
)

print("=" * 60)


if not OPENAI_API_KEY or not ANTHROPIC_API_KEY or not GEMINI_API_KEY:
    print("❌ One or more API keys are missing.")
    sys.exit(1)

print("✅ Environment setup successful!")


# ==========================================
# 4. Data Structures
# ==========================================

@dataclass
class LLMResponse:
    """Data class to store LLM response metadata and content."""
    provider: str
    model: str
    response: str = ""
    error: Optional[str] = None
    latency_sec: float = 0.0
    status: str = "failed"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: str
    model: str
    api_key_env_var: str


# ==========================================
# 5. Model Configurations
# ==========================================

MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        provider="openai",
        model="gpt-4o",
        api_key_env_var="OPENAI_API_KEY"
    ),
    ModelConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key_env_var="ANTHROPIC_API_KEY"
    ),
    ModelConfig(
        provider="gemini",
        model="gemini-1.5-pro",
        api_key_env_var="GEMINI_API_KEY"
    ),
]


# ==========================================
# 6. Parallel API Router
# ==========================================

async def query_single_model(
    model_config: ModelConfig,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> LLMResponse:
    """
    Query a single LLM model asynchronously.
    
    Args:
        model_config: Configuration for the model to query
        prompt: The user prompt to send to the model
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in the response
        
    Returns:
        LLMResponse object containing the model's response or error
    """
    start_time = time.time()
    
    try:
        logger.info(f"Querying {model_config.provider} - {model_config.model}")
        
        # Set environment variable for LiteLLM
        os.environ[model_config.api_key_env_var] = os.getenv(model_config.api_key_env_var)
        
        # Make async completion call using LiteLLM
        response = await acompletion(
            model=model_config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        latency = time.time() - start_time
        
        # Extract response text
        response_text = response["choices"][0]["message"]["content"]
        
        logger.info(f"✅ {model_config.provider} - {model_config.model} completed in {latency:.2f}s")
        
        return LLMResponse(
            provider=model_config.provider,
            model=model_config.model,
            response=response_text,
            latency_sec=latency,
            status="success"
        )
        
    except Exception as e:
        latency = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ {model_config.provider} - {model_config.model} failed: {error_msg}")
        
        return LLMResponse(
            provider=model_config.provider,
            model=model_config.model,
            error=error_msg,
            latency_sec=latency,
            status="failed"
        )


async def parallel_model_ingestion(
    prompt: str,
    model_configs: List[ModelConfig] = MODEL_CONFIGS,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> List[LLMResponse]:
    """
    Execute parallel queries to multiple LLM models.
    
    This is the core function of Phase 1 - Multi-Model Parallel Ingestion.
    It uses asyncio to send the same prompt to multiple models simultaneously.
    
    Args:
        prompt: The user prompt to send to all models
        model_configs: List of model configurations to query
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in each response
        
    Returns:
        List of LLMResponse objects from all models
    """
    logger.info(f"Starting parallel ingestion for prompt: '{prompt[:50]}...'")
    logger.info(f"Querying {len(model_configs)} models in parallel")
    
    # Create tasks for all model queries
    tasks = [
        query_single_model(config, prompt, temperature, max_tokens)
        for config in model_configs
    ]
    
    # Execute all tasks concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any unexpected exceptions
    processed_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            processed_responses.append(
                LLMResponse(
                    provider=model_configs[i].provider,
                    model=model_configs[i].model,
                    error=f"Unexpected error: {str(response)}",
                    status="failed"
                )
            )
        else:
            processed_responses.append(response)
    
    successful = sum(1 for r in processed_responses if r.status == "success")
    logger.info(f"Parallel ingestion complete: {successful}/{len(model_configs)} successful")
    
    return processed_responses


# ==========================================
# 7. Utility Functions
# ==========================================

def print_responses(responses: List[LLMResponse]) -> None:
    """Pretty print all LLM responses."""
    print("\n" + "=" * 80)
    print("PARALLEL MODEL INGESTION RESULTS")
    print("=" * 80)
    
    for i, response in enumerate(responses, 1):
        print(f"\n{'─' * 80}")
        print(f"Model {i}: {response.provider.upper()} - {response.model}")
        print(f"{'─' * 80}")
        print(f"Status: {response.status}")
        print(f"Latency: {response.latency_sec:.2f}s")
        
        if response.status == "success":
            print(f"\nResponse:\n{response.response}")
        else:
            print(f"\nError: {response.error}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in responses if r.status == "success")
    avg_latency = sum(r.latency_sec for r in responses) / len(responses)
    print(f"Successful: {successful}/{len(responses)}")
    print(f"Average Latency: {avg_latency:.2f}s")
    print("=" * 80 + "\n")


# ==========================================
# 8. Main Execution
# ==========================================

async def main():
    """Main execution function for Phase 1."""
    # Example user prompt
    user_prompt = "Explain the history of the internet in 3 paragraphs."
    
    print("\n" + "=" * 80)
    print("PHASE 1: MULTI-MODEL PARALLEL INGESTION")
    print("=" * 80)
    print(f"User Prompt: {user_prompt}")
    print("=" * 80)
    
    # Execute parallel ingestion
    responses = await parallel_model_ingestion(
        prompt=user_prompt,
        model_configs=MODEL_CONFIGS,
        temperature=0.7,
        max_tokens=500
    )
    
    # Display results
    print_responses(responses)
    
    return responses


if __name__ == "__main__":
    # Run the async main function
    responses = asyncio.run(main())



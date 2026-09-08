import os
import sys
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from dataclasses import dataclass
# ==========================================
# 1. Load .env file
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")

load_dotenv(env_path)


# ==========================================
# 2. Read API keys
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")


# ==========================================
# 3. Check API keys
# ==========================================

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print(f".env path: {env_path}")

print(
    "Gemini API Key:",
    "FOUND" if GEMINI_API_KEY else "NOT FOUND"
)

print(
    "Groq API Key:",
    "FOUND" if GROQ_API_KEY else "NOT FOUND"
)

print(
    "Hugging Face API Key:",
    "FOUND" if HF_API_KEY else "NOT FOUND"
)

print("=" * 60)


if not GEMINI_API_KEY or not GROQ_API_KEY or not HF_API_KEY:
    print("❌ One or more API keys are missing.")
    sys.exit(1)

print("✅ Environment setup successful!")


# ==========================================
# 4. Create a Structure.
# ==========================================

@dataclass
class LLMResponse:
    provider:str
    model:str
    response: str=""
    error: Optional[str] = None
    latency_sec: float = 0.0
    status: str = "failed"



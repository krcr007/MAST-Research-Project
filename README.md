# MAST Research Project

Multi-Model Aggregation and Synthesis Toolkit (MAST) - A research project for parallel LLM ingestion, aggregation, and synthesis.

## Overview

MAST is a comprehensive framework for querying multiple Large Language Models (LLMs) in parallel, aggregating their responses, and synthesizing a unified output. The project is divided into multiple phases, each building upon the previous one.

## Project Structure

```
MAST Research Project/
├── Complete Phases/
│   ├── Phase1.py          # Multi-Model Parallel Ingestion
│   ├── __init__.py
│   └── .env               # API keys (not committed to git)
├── testing_dataset/       # Test data for evaluation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Phases

### Phase 1: Multi-Model Parallel Ingestion ✅

**Status**: Completed

**Description**: This phase implements parallel API routing to multiple LLM models using asyncio and LiteLLM. It takes a user prompt and simultaneously queries multiple models to generate raw text outputs.

**Key Features**:
- **Async Parallel Execution**: Uses Python's `asyncio` to query multiple models concurrently
- **LiteLLM Integration**: Unified interface for multiple LLM providers
- **Error Handling**: Robust error handling with detailed logging
- **Response Tracking**: Captures latency, status, and metadata for each model response

**Supported Models**:
- **GPT-4o** (OpenAI)
- **Claude 3.5 Sonnet** (Anthropic)
- **Gemini 1.5 Pro** (Google)

**Architecture**:
```
User Prompt → Parallel API Router → Multiple Models → Raw Text Outputs
                      ↓
                (asyncio + LiteLLM)
```

**Usage**:
```bash
cd "Complete Phases"
python Phase1.py
```

**Output**:
- Console output showing responses from all models
- Latency metrics for each model
- Success/failure status for each query
- Summary statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- API keys for OpenAI, Anthropic, and Google (Gemini)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "MAST Research Project"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
Create a `.env` file in the `Complete Phases/` directory with the following content:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note**: The `.env` file is already in `.gitignore` to prevent accidental commits of sensitive data.

## Dependencies

Key dependencies for Phase 1:
- `litellm` - Unified interface for multiple LLM providers
- `python-dotenv` - Environment variable management
- `asyncio` - Asynchronous programming (built-in)

Additional dependencies for future phases:
- `pandas`, `numpy` - Data manipulation
- `torch` - Deep learning framework
- `langchain*` - LLM orchestration
- `fastapi`, `uvicorn` - API framework
- `nltk`, `spacy` - NLP tools

## API Key Setup

### OpenAI API Key
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Add to `.env` file as `OPENAI_API_KEY`

### Anthropic API Key
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Add to `.env` file as `ANTHROPIC_API_KEY`

### Google Gemini API Key
1. Go to [makersuite.google.com](https://makersuite.google.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Add to `.env` file as `GEMINI_API_KEY`

## Running Phase 1

To run Phase 1:

```bash
cd "Complete Phases"
python Phase1.py
```

The script will:
1. Check for required API keys
2. Send the example prompt to all three models in parallel
3. Display responses with latency metrics
4. Show summary statistics

**Example Output**:
```
============================================================
PHASE 1: MULTI-MODEL PARALLEL INGESTION
============================================================
User Prompt: Explain the history of the internet in 3 paragraphs.
============================================================

────────────────────────────────────────────────────────────────
Model 1: OPENAI - gpt-4o
────────────────────────────────────────────────────────────────
Status: success
Latency: 2.34s

Response: [GPT-4o's response here]

────────────────────────────────────────────────────────────────
Model 2: ANTHROPIC - claude-3-5-sonnet-20241022
────────────────────────────────────────────────────────────────
Status: success
Latency: 1.89s

Response: [Claude's response here]

────────────────────────────────────────────────────────────────
Model 3: GEMINI - gemini-1.5-pro
────────────────────────────────────────────────────────────────
Status: success
Latency: 2.67s

Response: [Gemini's response here]

============================================================
SUMMARY
============================================================
Successful: 3/3
Average Latency: 2.30s
============================================================
```

## Code Documentation

### Phase1.py Structure

The Phase 1 implementation is organized into the following sections:

1. **Environment Setup**: Loads `.env` file and validates API keys
2. **Data Structures**: 
   - `LLMResponse`: Stores model response metadata
   - `ModelConfig`: Configuration for each model
3. **Model Configurations**: List of supported models with their API key mappings
4. **Parallel API Router**:
   - `query_single_model()`: Async function to query a single model
   - `parallel_model_ingestion()`: Core function that orchestrates parallel queries
5. **Utility Functions**: `print_responses()` for formatted output
6. **Main Execution**: Example usage with a sample prompt

### Key Functions

#### `parallel_model_ingestion()`
The core function that executes parallel queries to multiple LLM models.

**Parameters**:
- `prompt` (str): The user prompt to send to all models
- `model_configs` (List[ModelConfig]): List of model configurations
- `temperature` (float): Sampling temperature (0.0 to 1.0)
- `max_tokens` (int): Maximum tokens in each response

**Returns**:
- `List[LLMResponse]`: Responses from all models

## Future Phases

- **Phase 2**: Response Aggregation and Comparison
- **Phase 3**: Consensus Building and Synthesis
- **Phase 4**: Quality Evaluation and Metrics
- **Phase 5**: API Integration and Deployment

## Contributing

This is a research project. Contributions and suggestions are welcome.

## License

[Specify your license here]

## Contact

[Your contact information]

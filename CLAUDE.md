# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VeilBench** is a benchmark framework for evaluating LLMs with thinking/reasoning modes (DeepSeek, Qwen, etc.). It uses a three-phase pipeline: answer collection, anonymous LLM-as-a-judge evaluation, and HTML report generation. The codebase and documentation are primarily in Chinese.

## Commands

```bash
# Overview of all phases and available models
python3 benchmark.py

# Phase 1: Collect answers for a specific model (supports checkpoint resume)
python3 run_benchmark.py <model_key>
# e.g.: python3 run_benchmark.py deepseek-v4-flash-high

# Phase 2: Anonymous evaluation (scores all collected answers)
python3 evaluate.py

# Phase 3: Generate HTML report
python3 report.py
```

Dependencies: Python 3.14, `openai` package. No `requirements.txt` exists — dependencies are managed via the `.venv` virtual environment.

## Architecture

The pipeline is strictly sequential and must run in order:

```
prompts.py (test definitions)
    → run_benchmark.py (Phase 1: collect model answers → results/*.json)
    → evaluate.py (Phase 2: anonymous judge scoring → results/evaluated_results.json)
    → report.py (Phase 3: generate report.html)
```

### Key Files

- **`config.py`** — Central configuration hub: model definitions (`MODELS` dict), judge config (`JUDGE_API_CONFIG`), scoring weights (`OBJECTIVE_WEIGHT=0.40`, `SUBJECTIVE_WEIGHT=0.60`), anonymization functions, and the judge system prompt. **Add new models here** by adding entries to `MODELS` with `api_base`, `api_key`, `model_id`, and `thinking_mode` fields.
- **`prompts.py`** — Test suite (~1400 lines): 32 problems across 8 categories (math, algorithm, system, debug, writing, logic, multilingual, long context). Each test is a dataclass with `id`, `category`, `prompt`, `evaluation_criteria`, `objective_type`, `objective_config`, and optional `code_to_run`.
- **`models.py`** — `ModelClient` class wrapping OpenAI-compatible APIs. Handles thinking mode parameters (`reasoning_effort`, `extra_body.thinking`) automatically. Used for both test models and judge models.
- **`evaluator.py`** — Scoring engine: anonymizes model identities (random Model_A/B/C mapping), constructs multi-answer judge prompts, parses JSON responses, supports multi-judge cross-validation (currently disabled via `MULTI_JUDGE_ENABLED=False`), and computes z-score normalization.
- **`validators/`** — Objective validation modules: `math_range`, `exact_text`, `python_exec` (sandboxed subprocess execution), `code_fix`. Dispatched via `VALIDATOR_MAP` based on each test's `objective_type`.
- **`report.py`** — Generates `report.html` with custom CSS styling and inline SVG charts (radar, bar, heatmap). No external charting libraries.

### Data Flow

- Phase 1 saves per-model results to `results/<model_key>.json` with incremental checkpoint support (skips already-completed tests on resume).
- Phase 2 loads all model results, runs objective validation locally, sends anonymized answers to the judge model, blends objective (40%) and subjective (60%) scores, and writes `results/evaluated_results.json`.
- Phase 3 reads evaluated results and generates a self-contained HTML report.

### Anonymous Evaluation Design

Model identities are obfuscated through `generate_anon_map()` (random shuffle → Model_A/B/C) and `anonymize_prompt()` (regex-based removal of brand names like "DeepSeek", "Qwen", etc.). This prevents the judge model from being biased by model identity.

## Configuration Notes

- API keys are currently hardcoded in `config.py`. Available model keys are defined in the `MODELS` dict and printed by `python3 benchmark.py`.
- Thinking mode: DeepSeek models use `reasoning_effort` ("high"/"max"); Qwen models have native thinking enabled with no extra parameter.
- Judge model is configured separately in `JUDGE_API_CONFIG` with low temperature (0.2) for scoring consistency.

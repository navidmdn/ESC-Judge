# Changelog — ESC-Judge Absolute Evaluation Framework

Changes relative to upstream [navidmdn/ESC-Judge](https://github.com/navidmdn/ESC-Judge).

---

## Step 1: Local candidate adapter via llama.cpp

**Date:** 2025-07-23

### What changed

| Action | File | Description |
|--------|------|-------------|
| Created | `adapters/__init__.py` | Package init |
| Created | `adapters/local_llama.py` | Local candidate adapter using llama-server |
| Created | `adapters/cloud.py` | Cloud adapter (seeker/judge) using OpenAI API |
| Modified | `.gitignore` | Added `models/` to prevent committing GGUF files |

### Why

The upstream repo loads local models via HuggingFace transformers (`load_model()` in `langchain_persona_based_chat.py`), which assumes CUDA and does not support GGUF quantized models. Our candidates are quantized configs running on Apple Silicon via llama.cpp.

### How it was done upstream

- `load_model()` does substring dispatch (`'llama' in name` → HuggingFace, `'gpt' in name` → ChatOpenAI)
- One code path serves both seeker and candidate — no separation of local vs. cloud
- Imports a missing `utils.langchain` module (repo does not run as-is)
- Returns LangChain `Runnable` objects; all inference coupled to LangChain chains

### What we did instead

- The adapter starts `llama-server` (llama.cpp's HTTP server) as a subprocess and talks to it via the OpenAI-compatible `/v1/chat/completions` endpoint
- Crash isolation: if the model OOMs or segfaults, the subprocess dies, the harness records the failure and continues
- `server_binary` parameter accepts the full path to a specific llama.cpp build (Metal, kleidai, CPU) — backend selection is a config choice, not a code branch
- `extra_args` parameter passes backend-specific flags (e.g., `-fa 1` for Metal flash attention)
- `GenerationResult` dataclass uses **llama.cpp native timings** (`prefill_ms`, `predicted_ms`, `prefill_per_token_ms`, `predicted_per_token_ms`) parsed from the server's `timings` response field as primary latency metrics, with `wall_clock_seconds` as a secondary measure
- Separate `CloudAdapter` with the same `GenerationResult` interface for the seeker and judge — no LangChain dependency in either adapter
- No HuggingFace dependency anywhere in the new code

---

## Step 2: Conversation generation pipeline

**Date:** 2025-07-24

### What changed

| Action | File | Description |
|--------|------|-------------|
| Created | `run_generation.py` | Main generation pipeline — simulate, run, resume |
| Created | `config.yaml` | Central YAML config for all candidates and settings |
| Created | `data/prompts/supporter_social_companion.txt` | Supporter system prompt (social companion role) |
| Created | `data/prompts/seeker_template.txt` | Seeker prompt template with `{description}` placeholder |
| Created | `.env` | API keys file (gitignored) |


### Why

The upstream repo's `langchain_persona_based_chat.py` generates conversations using LangChain chains and HuggingFace models with a hardcoded flow. We need a config-driven pipeline that iterates over multiple candidate configurations, runs multiple scenarios and repetitions, produces structured JSON output with per-turn metrics, and is resumable across interruptions.

### How it was done upstream

- `langchain_persona_based_chat.py` uses `RunnableWithMessageHistory` chains for both seeker and supporter
- Prompts are hardcoded in the script
- Output is a flat text file, no structured metadata or latency metrics
- No concept of multiple candidate configurations or repetitions
- No resumability — reruns regenerate everything

### What we did instead

- **Config-driven**: `config.yaml` defines all candidates, seeker settings, pipeline estimates, and prompt file paths. Adding a new candidate is one YAML block
- **Structured JSON output**: each conversation produces a self-describing JSON file with:
  - Full model provenance (`model_path`, `server_binary`, `extra_args`, `n_gpu_layers`, `n_ctx`)
  - Clean turn-by-turn transcript (role + text only, no metrics mixed in)
  - Supporter-only metrics separated into `metrics.llm_calls` and `metrics.pipeline`
- **Per-turn metrics** organized into three groups per supporter turn:
  - `tokens` — prompt, completion, total
  - `timing_ms` — ttft (time to first token), decode, inference_total
  - `throughput` — prefill_tokens_per_s, decode_tokens_per_s
- **Pipeline latency estimates**: configurable `stt_s` and `tts_s` combined with measured `llm_inference_s` to compute `end_to_end_latency_s` per supporter turn
- **Resumable**: `is_completed()` checks each output file before running; interrupted runs skip already-finished conversations
- **Repetitions**: each scenario × candidate runs N times (configurable) for variance estimation
- **File-based prompts**: supporter and seeker prompts live in `data/prompts/`, referenced by path in config — easily swappable without code changes
- **Folder structure**: `transcripts/{config_id}/{scenario_id}_rep{N}.json` — one subfolder per candidate, one file per conversation
- **Safety checks on adapter startup**:
  - Port-busy detection before launching llama-server (prevents silently talking to a stale server)
  - Model identity verification via `/v1/models` after startup (confirms the correct GGUF is loaded)

  ---

## Step 3: Scenario determinism and provider-agnostic seeker

**Date:** 2025-07-24

### What changed

| Action | File | Description |
|--------|------|-------------|
| Modified | `config.yaml` | Added `n_scenarios`, seeker `base_url` and `api_key_env` fields |
| Modified | `adapters/cloud.py` | Provider-agnostic auth via `api_key_env`; optional `seed` parameter |
| Modified | `run_generation.py` | Scenario cap (`n_scenarios`), seeker seed per repetition, provider config passthrough |

### Why

The upstream repo runs all 100 personas with no randomness controls and is hardcoded to OpenAI. We need: (1) a configurable scenario subset to control cost and runtime, (2) seeker seed for reproducibility across candidates, and (3) the ability for researchers to use any OpenAI-compatible LLM provider as the seeker.

### How it was done upstream

- Iterates all personas in file order, no option to limit
- No seed or determinism controls anywhere
- Seeker is hardcoded to OpenAI's API with `OPENAI_API_KEY`

### What we did instead

- **Scenario cap**: `n_scenarios` in YAML (default 25) limits the pipeline to the first N personas from the scenarios file. Set to `null` to use all
- **Seeker seed**: each repetition index (0, 1, 2…) is passed as the `seed` parameter to the seeker's API call. All candidates sharing rep 0 get the same seeker behavior, isolating variation to the supporter model only. The seed is recorded in each transcript's `seeker.seed` field
- **Supporter left unseeded**: natural variance across repetitions is preserved — this is what we're measuring
- **Provider-agnostic seeker**: `CloudAdapter` now takes `api_key_env` (the name of the env var, e.g. `"ANTHROPIC_API_KEY"`) instead of reading `OPENAI_API_KEY` directly. Combined with `base_url`, this supports any OpenAI-compatible provider (OpenAI, Anthropic, Groq, OpenRouter, etc.) with no code changes — just YAML config. API keys are stored in `.env` (gitignored), never in the YAML

---

## Step 4: Aggregation pipeline and cost tracking

**Date:** 2025-07-25

### What changed

| Action | File | Description |
|--------|------|-------------|
| Created | `aggregate.py` | Per-conversation and per-candidate aggregation with CSV output |
| Modified | `config.yaml` | Added seeker `pricing` block (cost per 1M tokens) |
| Modified | `run_generation.py` | Added `seeker_calls` tracking, cold start capture |
| Modified | `adapters/local_llama.py` | Cold start warmup measurement on server start |

### Why

After generating conversations across multiple candidate configurations, we need to compare them quantitatively: latency distributions, throughput, token verbosity, and seeker API cost. Raw transcript JSON is not human-readable at scale.

### Modification

- **Aggregation pipeline** (`aggregate.py`):
  - Reads all completed transcripts from the output directory
  - Computes per-conversation stats: mean/p95 for TTFT, decode time, inference total, throughput, E2E latency; total supporter and seeker tokens; seeker API cost
  - Rolls up per-candidate: mean/std/p95 across conversations for each metric
  - Writes `reports/per_conversation.csv` and `reports/per_candidate.csv`
  - Prints a summary comparison table to stdout
  - Seeker cost computed from configurable pricing in YAML (`cost per 1M tokens × token count`)
- **Seeker call tracking**: `run_generation.py` now logs `seeker_calls` (token counts per seeker turn) in each transcript's `metrics` block, enabling cost attribution
- **Cold start warmup**: `LlamaServerAdapter.start()` sends a throwaway prompt after server health check to warm GPU caches. The cold start result is recorded in each transcript's `cold_start` field (TTFT, decode, inference total, wall clock)

---

## Step 5: Prefill throughput fix (KV cache correction)

**Date:** 2025-07-25

### What changed

| Action | File | Description |
|--------|------|-------------|
| Modified | `adapters/local_llama.py` | Added `prompt_n` and `cache_n` fields to `GenerationResult` |
| Modified | `run_generation.py` | Fixed `prefill_tokens_per_s` to use `prompt_n`; added `cache` block per llm_call |
| Modified | `aggregate.py` | Backward-compatible prefill throughput recomputation; added `cache_hit_ratio_mean` |

### Why

The original `prefill_tokens_per_s` formula divided total prompt tokens (`usage.prompt_tokens`) by prefill time (`timings.prompt_ms`). With llama.cpp's KV prefix cache, `prompt_ms` only covers newly processed tokens while `prompt_tokens` counts the full context. The ratio inflated linearly with turn number. The metric was measuring cache hit rate, not hardware speed.

### Modification

- **Use `timings.prompt_n`**: llama.cpp's server returns `prompt_n` (tokens actually prefilled, not cached) and `cache_n` (tokens reused from KV cache). `GenerationResult` now captures both
- **Corrected throughput**: `prefill_tokens_per_s = prompt_n / (prefill_ms / 1000)` — only counts work actually done
- **Cache metrics**: each `llm_calls` entry now includes a `cache` block with `prompt_n`, `cached_tokens`, and `cache_hit_ratio` — the caching behavior is interesting data in its own right
- **Backward compatibility**: `aggregate.py` detects old transcripts (no `cache` block) and falls back to the stored value. Old runs are not comparable to new runs on this metric
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
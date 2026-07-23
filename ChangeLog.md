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
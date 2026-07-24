# ESC-Judge — Stage 2 Absolute Evaluation Framework

Fork of [navidmdn/ESC-Judge](https://github.com/navidmdn/ESC-Judge), adapted for absolute (non-pairwise) evaluation of local LLM configurations as emotional support dialogue agents.

## Purpose

This framework is Stage 2 of a two-stage pipeline for selecting a local LLM configuration to run on a social robot (Pepper):

- **Stage 1** (separate repo): objective feasibility screening — load success, peak memory, latency, rule-based constraints. Outputs a shortlist of finalist configurations.
- **Stage 2** (this repo): for each finalist, generate multi-turn emotional support conversations and score them on dialogue quality using an LLM judge.

The candidate configurations are local LLM setups that passed Stage 1 feasibility screening — they may differ in model family, size, or quantization level. The research question is how these configurations compare in emotional support dialogue quality across turns.

## Key differences from upstream ESC-Judge

| Upstream | This fork |
|----------|-----------|
| Pairwise comparison (Model A vs. B) | Absolute 1–5 scoring per conversation |
| Quadratic cost in number of candidates | Linear cost |
| Conversation-level judgments only | Per-turn scored records |
| HuggingFace / CUDA for local models | llama.cpp via llama-server (Metal / kleidai / CPU) |
| Single blended judge prompt | One judge call per dimension (anti-halo) |
| LangChain throughout | LangChain-free adapter layer |
| Hardcoded prompts | File-based, config-referenced prompts |
| No resumability | Skips already-completed conversations |
| Flat text output | Structured JSON with per-turn metrics |

## Scoring dimensions

Based on Hill's Exploration–Insight–Action framework:

- **Exploration** — facilitating emotional expression and empathic understanding
- **Insight** — helping the seeker gain self-understanding and new perspectives
- **Action** — guiding toward concrete, collaborative steps for change

Each dimension is scored independently (separate judge call) to prevent halo bias.

## Setup

```bash
conda create -n esc-judge python=3.11 -y
conda activate esc-judge
pip install requests openai python-dotenv pyyaml fire jupyter
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
```

## Project structure

```
adapters/
  local_llama.py          # Candidate adapter — starts llama-server, talks via HTTP
  cloud.py                # Cloud adapter (seeker/judge) — OpenAI API
run_generation.py         # Generation pipeline — simulate, run, resume
config.yaml               # Central config — candidates, seeker, prompts, estimates
data/
  prompts/
    supporter_social_companion.txt   # Supporter system prompt
    seeker_template.txt              # Seeker prompt template ({description} placeholder)
  roles-v1.json           # Persona scenarios (one JSON object per line)
models/                   # GGUF files (gitignored)
transcripts/              # Generated conversations (gitignored)
  {config_id}/
    {scenario_id}_rep{N}.json
```

## Usage

### Run all candidates

```bash
python run_generation.py --config_path config.yaml
```

The pipeline reads `config.yaml`, starts each candidate's llama-server sequentially, runs all scenarios × repetitions, and saves structured JSON transcripts. Already-completed conversations are skipped automatically.

### Configuration

All settings live in `config.yaml`:

- **`provider_role`** — path to the supporter system prompt file
- **`seeker`** — cloud model name, temperature, prompt template path
- **`candidates`** — list of model configs (GGUF path, server binary, GPU layers, context size, temperature, extra args, port)
- **`n_turns`** — conversation length (supporter + seeker turns)
- **`n_repetitions`** — runs per scenario for variance estimation
- **`pipeline_estimates`** — STT/TTS latency estimates for end-to-end latency calculation

### Output format

Each transcript JSON contains:

- **Model provenance**: GGUF path, server binary, extra args, GPU layers, context size
- **Clean transcript**: turn-by-turn `[{turn_id, role, text}]` — no metrics mixed in
- **`metrics.llm_calls`** (supporter turns only):
  - `tokens` — prompt, completion, total
  - `timing_ms` — ttft, decode, inference_total
  - `throughput` — prefill_tokens_per_s, decode_tokens_per_s
- **`metrics.pipeline`** (supporter turns only): stt_s + llm_inference_s + tts_s = end_to_end_latency_s

## Requirements

- A llama.cpp build with `llama-server`
- GGUF model files in `models/`
- OpenAI API key in `.env` (for the seeker and judge)

## Citation

This work builds on ESC-Judge. If you use this framework, please cite the original paper:

> ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents — [navidmdn/ESC-Judge](https://github.com/navidmdn/ESC-Judge)

## See also

[CHANGELOG.md](CHANGELOG.md) — detailed log of all modifications relative to upstream
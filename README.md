# ESC-Judge — Stage 2 Absolute Evaluation Framework

Fork of [navidmdn/ESC-Judge](https://github.com/navidmdn/ESC-Judge), adapted for absolute (non-pairwise) evaluation of local LLM configurations as emotional support dialogue agents.

## Purpose

This framework is Stage 2 of a two-stage pipeline for selecting a local LLM configuration to run on a social robot (Pepper):

- **Stage 1** (separate repo): objective feasibility screening — load success, peak memory, latency, rule-based constraints. Outputs a shortlist of finalist configurations.
- **Stage 2** (this repo): for each finalist, generate multi-turn emotional support conversations and score them on dialogue quality using an LLM judge.

The candidate configurations are the same base model at different sizes and quantization levels. The research question is how compression degrades support quality across turns.

## Key differences from upstream ESC-Judge

| Upstream | This fork |
|----------|-----------|
| Pairwise comparison (Model A vs. B) | Absolute 1–5 scoring per conversation |
| Quadratic cost in number of candidates | Linear cost |
| Conversation-level judgments only | Per-turn scored records |
| HuggingFace / CUDA for local models | llama.cpp via llama-server (Metal / kleidai / CPU) |
| Single blended judge prompt | One judge call per dimension (anti-halo) |
| LangChain throughout | LangChain-free adapter layer |

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
pip install requests openai jupyter
```
## Project structure
```bash
adapters/
  local_llama.py    # Candidate adapter — starts llama-server, talks via HTTP
  cloud.py          # Cloud adapter (seeker/judge) — OpenAI API
models/             # GGUF files (gitignored)
data/               # Personas, rubrics, supporter prompts (from upstream)
```

## Requirements:
- A llama.cpp build with llama-server 
- GGUF model files in models/
- OpenAI API key (for the seeker and judge): export OPENAI_API_KEY="..."

## Citation
This work builds on ESC-Judge. If you use this framework, please cite the original paper:

ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents — navidmdn/ESC-Judge

## See also
CHANGELOG.md — detailed log of all modifications relative to upstream
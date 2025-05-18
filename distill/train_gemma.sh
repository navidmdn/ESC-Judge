#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────────
# train.sh     –  launch run_trainer with a full, explicit argument list
# Usage:       ./train.sh [TRAIN_DATA_JSON] [EVAL_DATA_JSON]
# Example:     ./train.sh data/train.json data/val.json
# ────────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Positional overrides for data files (defaults shown)
TRAIN_FILE="${1:-data/train.json}"
EVAL_FILE="${2:-data/val.json}"

WANDB_MODE=online WANDB_ENTITY=navidmdn WANDB_PROJECT=distill-judge python unsloth_sft_trainer.py \
  --train_data_file              "$TRAIN_FILE" \
  --eval_data_file               "$EVAL_FILE"  \
  --model_name                   "unsloth/gemma-3-4b-it-unsloth-bnb-4bit" \
  --max_seq_length               12000 \
  --load_in_4bit                 true \
  --dtype                        None \
  --report_to                    wandb \
  --per_device_train_batch_size  2 \
  --per_device_eval_batch_size   1 \
  --gradient_accumulation_steps  4 \
  --learning_rate                1e-4 \
  --num_train_epochs             3 \
  --warmup_ratio                 0.03 \
  --lr_scheduler_type            cosine_with_restarts \
  --logging_steps                2 \
  --eval_steps                   20 \
  --save_steps                   20 \
  --save_total_limit             2 \
  --output_dir                   "./outputs/gemma3-4b" \
  --seed                         42 \
  --metric_n_samples             5 \
  --generation_kwargs            '{"max_new_tokens":128,"do_sample":false}' \
  --lora_alpha                   256 \
  --lora_rank                    128 \
  --cache_dir                    /projects/academic/rohini/models/hf_cache \
  --eval_accumulation_steps      1
"""
Aggregate transcript metrics into per-conversation and per-candidate summaries.
Reads all JSON transcripts from the output directory and produces CSV reports.
"""

import os
import json
import csv
import numpy as np
import yaml
from typing import Dict, List, Optional


def load_transcripts(transcripts_dir: str) -> List[Dict]:
    transcripts = []
    for config_dir in sorted(os.listdir(transcripts_dir)):
        config_path = os.path.join(transcripts_dir, config_dir)
        if not os.path.isdir(config_path):
            continue
        for filename in sorted(os.listdir(config_path)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(config_path, filename)
            with open(filepath) as f:
                data = json.load(f)
            if data.get("status") == "complete" and "metrics" in data:
                transcripts.append(data)
    return transcripts


def compute_conversation_stats(transcript: Dict, pricing: Optional[Dict] = None) -> Dict:
    config_id = transcript["config_id"]
    scenario_id = transcript["scenario_id"]
    repetition = transcript["repetition"]

    llm_calls = transcript["metrics"]["llm_calls"]
    pipeline = transcript["metrics"]["pipeline"]
    seeker_calls = transcript["metrics"].get("seeker_calls", [])

    ttfts = [c["timing_ms"]["ttft"] for c in llm_calls]
    decodes = [c["timing_ms"]["decode"] for c in llm_calls]
    inference_totals = [c["timing_ms"]["inference_total"] for c in llm_calls]
    decode_tps = [c["throughput"]["decode_tokens_per_s"] for c in llm_calls]

    prefill_tps = []
    cache_hit_ratios = []
    for c in llm_calls:
        if "cache" in c:
            prompt_n = c["cache"]["prompt_n"]
            ttft_s = c["timing_ms"]["ttft"] / 1000.0
            prefill_tps.append(prompt_n / ttft_s if ttft_s > 0 else 0.0)
            cache_hit_ratios.append(c["cache"]["cache_hit_ratio"])
        else:
            prefill_tps.append(c["throughput"]["prefill_tokens_per_s"])
            cache_hit_ratios.append(0.0)

    e2e = [p["end_to_end_latency_s"] for p in pipeline]

    supporter_prompt_tokens = sum(c["tokens"]["prompt"] for c in llm_calls)
    supporter_completion_tokens = sum(c["tokens"]["completion"] for c in llm_calls)

    seeker_prompt_tokens = sum(c["tokens"]["prompt"] for c in seeker_calls)
    seeker_completion_tokens = sum(c["tokens"]["completion"] for c in seeker_calls)

    seeker_cost_usd = 0.0
    if pricing and seeker_calls:
        input_rate = pricing.get("input", 0.0) / 1_000_000
        output_rate = pricing.get("output", 0.0) / 1_000_000
        seeker_cost_usd = (seeker_prompt_tokens * input_rate) + (seeker_completion_tokens * output_rate)

    return {
        "config_id": config_id,
        "scenario_id": scenario_id,
        "repetition": repetition,
        "n_supporter_turns": len(llm_calls),
        "supporter_prompt_tokens": supporter_prompt_tokens,
        "supporter_completion_tokens": supporter_completion_tokens,
        "seeker_prompt_tokens": seeker_prompt_tokens,
        "seeker_completion_tokens": seeker_completion_tokens,
        "ttft_mean_ms": np.mean(ttfts),
        "ttft_p95_ms": np.percentile(ttfts, 95),
        "decode_mean_ms": np.mean(decodes),
        "decode_p95_ms": np.percentile(decodes, 95),
        "inference_total_mean_ms": np.mean(inference_totals),
        "inference_total_p95_ms": np.percentile(inference_totals, 95),
        "decode_tokens_per_s_mean": np.mean(decode_tps),
        "decode_tokens_per_s_p95": np.percentile(decode_tps, 95),
        "prefill_tokens_per_s_mean": np.mean(prefill_tps),
        "cache_hit_ratio_mean": np.mean(cache_hit_ratios),
        "e2e_latency_mean_s": np.mean(e2e),
        "e2e_latency_p95_s": np.percentile(e2e, 95),
        "seeker_cost_usd": seeker_cost_usd,
    }


def compute_candidate_rollup(conversation_stats: List[Dict]) -> Dict:
    def agg(key):
        vals = [s[key] for s in conversation_stats]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p95": float(np.percentile(vals, 95)),
        }

    config_id = conversation_stats[0]["config_id"]
    n_conversations = len(conversation_stats)

    metrics_keys = [
        "ttft_mean_ms", "decode_mean_ms", "inference_total_mean_ms",
        "decode_tokens_per_s_mean", "prefill_tokens_per_s_mean",
        "cache_hit_ratio_mean", "e2e_latency_mean_s",
    ]

    rollup = {
        "config_id": config_id,
        "n_conversations": n_conversations,
        "total_seeker_cost_usd": sum(s["seeker_cost_usd"] for s in conversation_stats),
    }
    for key in metrics_keys:
        stats = agg(key)
        rollup[f"{key}__mean"] = stats["mean"]
        rollup[f"{key}__std"] = stats["std"]
        rollup[f"{key}__p95"] = stats["p95"]

    return rollup


def write_csv(rows: List[Dict], filepath: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rollups: List[Dict]):
    print("\n" + "=" * 80)
    print("CANDIDATE COMPARISON SUMMARY")
    print("=" * 80)

    headers = [
        ("config_id", "Config", 35),
        ("n_conversations", "#Conv", 6),
        ("ttft_mean_ms__mean", "TTFT(ms)", 10),
        ("decode_tokens_per_s_mean__mean", "Tok/s", 8),
        ("e2e_latency_mean_s__mean", "E2E(s)", 8),
        ("total_seeker_cost_usd", "Cost($)", 9),
    ]

    header_line = ""
    for _, label, width in headers:
        header_line += f"{label:<{width}}"
    print(header_line)
    print("-" * 80)

    for r in rollups:
        line = ""
        for key, _, width in headers:
            val = r.get(key, "")
            if isinstance(val, float):
                if "cost" in key.lower():
                    line += f"${val:<{width - 1}.4f}"
                else:
                    line += f"{val:<{width}.1f}"
            else:
                line += f"{str(val):<{width}}"
        print(line)

    print("=" * 80)


def main(config_path: str = "config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    transcripts_dir = cfg["output_dir"]
    pricing = cfg["seeker"].get("pricing")
    output_dir = "reports"

    transcripts = load_transcripts(transcripts_dir)
    if not transcripts:
        print(f"No completed transcripts found in {transcripts_dir}/")
        return

    conversation_stats = [compute_conversation_stats(t, pricing) for t in transcripts]

    candidates = {}
    for s in conversation_stats:
        candidates.setdefault(s["config_id"], []).append(s)

    rollups = []
    for config_id in sorted(candidates.keys()):
        rollups.append(compute_candidate_rollup(candidates[config_id]))

    write_csv(conversation_stats, os.path.join(output_dir, "per_conversation.csv"))
    write_csv(rollups, os.path.join(output_dir, "per_candidate.csv"))

    print(f"Wrote {len(conversation_stats)} conversation rows to {output_dir}/per_conversation.csv")
    print(f"Wrote {len(rollups)} candidate rows to {output_dir}/per_candidate.csv")

    print_summary(rollups)


if __name__ == "__main__":
    import fire
    fire.Fire(main)

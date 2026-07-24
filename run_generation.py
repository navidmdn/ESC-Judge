"""
Conversation generation.

Replaces langchain_persona_based_chat.py's simulate() with an adapter-based
loop that produces structured JSON output with per-turn metadata.
"""

import os
import yaml
import json
from typing import List, Dict, Optional
from adapters.local_llama import LlamaServerAdapter
from adapters.cloud import CloudAdapter
from uuid import uuid4
from dotenv import load_dotenv

def load_prompt_template(filepath: str) -> str:
    with open(filepath) as f:
        return f.read().strip()

def build_messages(system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build OpenAI-format messages list from system prompt and conversation history."""
    return [{"role": "system", "content": system_prompt}] + history


def simulate(supporter_adapter, seeker_adapter,
             supporter_system_prompt: str, seeker_system_prompt: str,
             n_turns: int = 14,
             supporter_temperature: float = 0.7,
             seeker_temperature: float = 0.8,
             max_tokens: int = 512,
             stt_estimate_s: float = 0.0,
             tts_estimate_s: float = 0.0,
             opening_line: str = "Hey! how's it going?") -> Dict:
    """
    Run one multi-turn conversation between supporter (local) and seeker (cloud).
    Returns turns (clean text), llm_calls and pipeline (supporter-only metrics).
    """
    turns = []
    llm_calls = []
    pipeline = []
    supporter_history = []
    seeker_history = []

    opening = opening_line
    turns.append({
        "turn_id": 0,
        "role": "supporter",
        "text": opening,
    })

    last_utterance = opening
    cur_speaker = "supporter"

    for turn_idx in range(1, n_turns + 1):
        if cur_speaker == "supporter":
            supporter_history.append({"role": "assistant", "content": last_utterance})
            seeker_history.append({"role": "user", "content": last_utterance})
            cur_speaker = "seeker"
        else:
            supporter_history.append({"role": "user", "content": last_utterance})
            seeker_history.append({"role": "assistant", "content": last_utterance})
            cur_speaker = "supporter"

        if cur_speaker == "seeker":
            messages = build_messages(seeker_system_prompt, seeker_history)
            result = seeker_adapter.chat(
                messages, temperature=seeker_temperature, max_tokens=max_tokens
            )
        else:
            messages = build_messages(supporter_system_prompt, supporter_history)
            result = supporter_adapter.chat(
                messages, temperature=supporter_temperature, max_tokens=max_tokens
            )

        turns.append({
            "turn_id": turn_idx,
            "role": cur_speaker,
            "text": result.text,
        })

        if cur_speaker == "supporter":
            llm_calls.append({
                "turn_id": turn_idx,
                "tokens": {
                    "prompt": result.prompt_tokens,
                    "completion": result.completion_tokens,
                    "total": result.prompt_tokens + result.completion_tokens,
                },
                "timing_ms": {
                    "ttft": result.prefill_ms,
                    "decode": result.predicted_ms,
                    "inference_total": result.prefill_ms + result.predicted_ms,
                },
                "throughput": {
                    "prefill_tokens_per_s": (result.prompt_tokens / (result.prefill_ms / 1000.0)) if result.prefill_ms > 0 else 0.0,
                    "decode_tokens_per_s": result.tokens_per_s,
                },
            })
            pipeline.append({
                "turn_id": turn_idx,
                "stt_s": stt_estimate_s,
                "llm_inference_s": result.llm_inference_s,
                "tts_s": tts_estimate_s,
                "end_to_end_latency_s": stt_estimate_s + result.llm_inference_s + tts_estimate_s,
            })

        if result.error:
            return {
                "status": "error",
                "error_detail": result.error,
                "turns": turns,
                "metrics": {"llm_calls": llm_calls, "pipeline": pipeline},
            }

        last_utterance = result.text

    return {
        "status": "complete",
        "turns": turns,
        "metrics": {"llm_calls": llm_calls, "pipeline": pipeline},
    }


def get_output_path(output_dir: str, config_id: str, scenario_id: str, repetition: int) -> str:
    config_dir = os.path.join(output_dir, config_id)
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, f"{scenario_id}_rep{repetition}.json")


def is_completed(output_path: str) -> bool:
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path) as f:
            data = json.load(f)
        return data.get("status") == "complete"
    except (json.JSONDecodeError, KeyError):
        return False


def run_generation(
    scenarios_file: str,
    supporter_system_prompt_file: str,
    config_id: str,
    model_path: str,
    server_binary: str,
    seeker_model_name: str = "gpt-4o-mini",
    n_turns: int = 14,
    n_repetitions: int = 3,
    supporter_temperature: float = 0.7,
    seeker_temperature: float = 0.8,
    max_tokens: int = 512,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    extra_args: Optional[List[str]] = None,
    port: int = 8080,
    output_dir: str = "output/conversations",
    seeker_template_file: str = "data/prompts/seeker_template.txt",
    stt_estimate_s: float = 0.0,
    tts_estimate_s: float = 0.0,
    opening_line: str = "Hey! how's it going?",
):
    """
    Main entry point. For one candidate config, run all scenarios × repetitions.
    Skips already-completed conversations for resumability.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load scenarios (one JSON object per line)
    scenarios = []
    with open(scenarios_file) as f:
        for line in f:
            scenarios.append(json.loads(line.strip()))


    
    supporter_system_prompt = load_prompt_template(supporter_system_prompt_file)
    seeker_template = load_prompt_template(seeker_template_file)

    # Start local model server for this config
    supporter = LlamaServerAdapter(
        model_path=model_path,
        server_binary=server_binary,
        port=port,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        extra_args=extra_args,
    )
    supporter.start()

    seeker = CloudAdapter(model_name=seeker_model_name)

    completed = 0
    skipped = 0
    errors = 0

    try:
        for scenario in scenarios:
            scenario_id = scenario["pid"]
            seeker_system_prompt = seeker_template.format(description=scenario["role"])

            for rep in range(n_repetitions):
                output_path = get_output_path(output_dir, config_id, scenario_id, rep)

                if is_completed(output_path):
                    skipped += 1
                    print(f"[skip] {config_id} / {scenario_id} / rep {rep}")
                    continue

                print(f"[run]  {config_id} / {scenario_id} / rep {rep}")

                result = simulate(
                    supporter_adapter=supporter,
                    seeker_adapter=seeker,
                    supporter_system_prompt=supporter_system_prompt,
                    seeker_system_prompt=seeker_system_prompt,
                    n_turns=n_turns,
                    supporter_temperature=supporter_temperature,
                    seeker_temperature=seeker_temperature,
                    max_tokens=max_tokens,
                    stt_estimate_s=stt_estimate_s,
                    tts_estimate_s=tts_estimate_s,
                    opening_line=opening_line,
                )

                output = {
                    "schema_version": 1,
                    "transcript_id": str(uuid4()),
                    "config_id": config_id,
                    "scenario_id": scenario_id,
                    "repetition": rep,
                    "model": {
                        "model_id": config_id,
                        "model_path": model_path,
                        "server_binary": server_binary,
                        "extra_args": extra_args or [],
                        "n_gpu_layers": n_gpu_layers,
                        "n_ctx": n_ctx,
                        "llm_base_url": f"http://127.0.0.1:{port}",
                    },
                    "decoding": {
                        "temperature": supporter_temperature,
                        "max_tokens": max_tokens,
                    },
                    "seeker": {
                        "model": seeker_model_name,
                        "temperature": seeker_temperature,
                    },
                    "provider_role": supporter_system_prompt_file,
                    "n_turns": n_turns,
                    "status": result["status"],
                    "turns": result["turns"],
                    "metrics": result["metrics"],
                }
                if result.get("error_detail"):
                    output["error_detail"] = result["error_detail"]
                    errors += 1
                else:
                    completed += 1

                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)

                print(f"       -> {result['status']}")

    finally:
        supporter.stop()
        print(f"\nDone. completed={completed}, skipped={skipped}, errors={errors}")



def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    """Load config and run generation for all candidate configs."""
    
    load_dotenv()
    cfg = load_config(config_path)
    
    for candidate in cfg["candidates"]:
        run_generation(
            scenarios_file=cfg["scenarios_file"],
            supporter_system_prompt_file=cfg["provider_role"],
            seeker_template_file=cfg["seeker"]["template_file"],
            seeker_model_name=cfg["seeker"]["model"],
            seeker_temperature=cfg["seeker"]["temperature"],
            config_id=candidate["config_id"],
            model_path=candidate["model_path"],
            server_binary=os.path.expanduser(candidate["server_binary"]),
            n_turns=cfg["n_turns"],
            n_repetitions=cfg["n_repetitions"],
            supporter_temperature=candidate["temperature"],
            max_tokens=candidate["max_tokens"],
            n_ctx=candidate["n_ctx"],
            n_gpu_layers=candidate["n_gpu_layers"],
            extra_args=candidate.get("extra_args"),
            port=candidate["port"],
            output_dir=cfg["output_dir"],
            stt_estimate_s=cfg["pipeline_estimates"]["stt_s"],
            tts_estimate_s=cfg["pipeline_estimates"]["tts_s"],
            opening_line=cfg["opening_line"],
        )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
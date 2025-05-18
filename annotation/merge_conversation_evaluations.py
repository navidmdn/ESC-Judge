import os
import json
from typing import List, Dict, Any
import re


def parse_conversation_file(filepath: str) -> List[Dict[str, str]]:
    """
    Reads a conversation .txt file line by line.
    Expects lines like:
        supporter: Hello, how are you?
        seeker: I'm okay...
    Returns a list of dicts like: [{"role": "supporter", "text": "Hello, how are you?"}, ...]
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as cf:
        conversation_text = cf.read().strip()
        pattern = r"(seeker:|supporter:)(.*?)(?=(seeker:|supporter:|$))"
        matches = re.findall(pattern, conversation_text, re.DOTALL)
        conversation = [{'role': speaker.strip(), 'text': message.strip()} for speaker, message, _ in matches]

    return conversation

def load_all_evaluations(evaluations_folder: str) -> List[Dict[str, Any]]:
    """
    Loads all .json files in the given folder, expecting each .json to contain
    either a single evaluation or a list of evaluations. Aggregates them all into
    a single list of evaluation entries.
    """
    all_evaluations = []
    for filename in os.listdir(evaluations_folder):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(evaluations_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Some files might contain a list, others might contain a single dict
            # Standardize by always extending a list
            if isinstance(data, list):
                all_evaluations.extend(data)
            elif isinstance(data, dict):
                all_evaluations.append(data)
            else:
                print(f"Skipping {filename}: not a list or dict.")
    return all_evaluations

def build_conversation_map(conversations_folder: str) -> Dict[tuple, List[Dict[str, str]]]:
    """
    Reads all conversation .txt files in `conversations_folder`.
    Assumes each file is named like:
        <model_name>_<personality_id>.txt
    e.g.
        output-exp2-nodir_meta-llama-Llama-3.2-3B-Instruct_e07d6dbe-66a1-4a71-a010-a7f45e4a6bac.txt

    Returns a dict keyed by (personality_id, model_name),
    mapping to the list of messages from that file.
    """
    conversation_map = {}
    for filename in os.listdir(conversations_folder):
        if not filename.endswith(".txt"):
            continue

        base = filename[:-4]  # remove ".txt" extension
        if "_" not in base:
            continue

        # Split into <model_name> and <personality_id>
        model_name, personality_id = base.rsplit("_", 1)
        filepath = os.path.join(conversations_folder, filename)

        # Parse the conversation file into a list of {role, text} messages
        messages = parse_conversation_file(filepath)
        conversation_map[(personality_id, model_name)] = messages

    return conversation_map


def merge_evaluations_with_conversations(
    evaluations_folder: str,
    conversations_folder: str,
    output_file: str
) -> None:
    """
    Merge all evaluations in `evaluations_folder` with conversation logs in `conversations_folder`
    based on matching personality_id (eval['pid']) and model names (eval['model_a'], eval['model_b']).

    Writes a single JSON file (list of merged objects) to `output_file`.
    """

    # 1. Load all evaluation entries from all JSON files in the evaluations folder
    all_evaluations = load_all_evaluations(evaluations_folder)

    # 2. Build a map of {(personality_id, model_name): [messages]}
    conversation_map = build_conversation_map(conversations_folder)

    # 3. Prepare the result list
    merged_entries = []

    for eval_item in all_evaluations:
        # We expect something like:
        # {
        #   "pid": "e07d6dbe-66a1-4a71-a010-a7f45e4a6bac",
        #   "model_a": "output-exp2-nodir_meta-llama-Llama-3.2-3B-Instruct",
        #   "model_b": "output-exp2-hillprompt_gpt-4o-mini",
        #   "Exploration": {...},
        #   "Insight": {...},
        #   "Action": {...}
        #   ...
        # }
        personality_id = eval_item.get("pid", "")
        model_a_name = eval_item.get("model_a", "")
        model_b_name = eval_item.get("model_b", "")

        # We'll gather the dimension-by-dimension data into an 'evaluation' block.
        # Adjust this logic if your data is structured differently.
        evaluation_data = {
            "Exploration": eval_item.get("Exploration", {}),
            "Insight": eval_item.get("Insight", {}),
            "Action": eval_item.get("Action", {})
        }

        # 4. Look up the conversation logs for model_a and model_b
        conversation_a = conversation_map.get((personality_id, model_a_name), [])
        conversation_b = conversation_map.get((personality_id, model_b_name), [])

        # 5. Build the merged structure
        merged_entry = {
            "personality_id": personality_id,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "evaluation": evaluation_data,
            "conversation": {
                "modelA": {
                    "model_id": model_a_name,
                    "messages": conversation_a
                },
                "modelB": {
                    "model_id": model_b_name,
                    "messages": conversation_b
                }
            }
        }
        merged_entries.append(merged_entry)

    # 6. Write out the merged results as JSON
    with open(output_file, "w", encoding="utf-8") as outf:
        json.dump(merged_entries, outf, ensure_ascii=False, indent=2)

    print(f"Done! Merged data saved to {output_file}.")


merge_evaluations_with_conversations(
    evaluations_folder="../output/evals_dim2",
    conversations_folder="../output/exp2/",
    output_file="merged_output.json"
)

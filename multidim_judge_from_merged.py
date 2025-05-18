import fire
import os
import re
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
from tqdm import tqdm
from typing import List, Dict


def load_text(file_path):
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_jsonl(file_path):
    """Load jsonl content from a file."""
    result = []
    with open(file_path, "r") as file:
        for line in file:
            result.append(json.loads(line))
    return result

def load_json(file_path):
    """Load jsonl content from a file."""
    with open(file_path, "r") as file:
        content = file.read().strip()
        return json.loads(content)

def convert_messages_to_str(messages: List[Dict]):
    conv = ""
    for i, msg in enumerate(messages):
        speaker = 'Assistant' if i % 2 == 0 else 'User'
        conv += f'{speaker}: {msg["text"]}\n'
    return conv

def compare_models(comparison_obj, prompt_text, eval_model_name, openai_api_key,
                                    criteria_dict, batch_size=1):
    evaluation = {}

    conversation_a = convert_messages_to_str(comparison_obj['conversation']['modelA']['messages'])
    conversation_b = convert_messages_to_str(comparison_obj['conversation']['modelB']['messages'])

    prompts = []
    for coarse_criteria, dims in criteria_dict.items():
        for criteria, desc in dims.items():
            prompts.append(prompt_text.format(
                conversation_a=conversation_a,
                conversation_b=conversation_b,
                criteria=criteria,
                criteria_description=desc,
            ))

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=eval_model_name, temperature=1.0)
    responses = []

    bi = 0
    while bi < len(prompts):
        batch = prompts[bi:bi + batch_size]

        batch_responses = llm.batch([
            [
                # SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=p)
            ] for p in batch
        ])

        responses.extend(batch_responses)
        bi += batch_size

    # Store evaluation result
    evaluation['pid'] = comparison_obj['personality_id']
    evaluation['model_a'] = comparison_obj['model_a']
    evaluation['model_b'] = comparison_obj['model_b']

    resp_i = 0
    for coarse_criteria, dims in criteria_dict.items():
        evaluation[coarse_criteria] = {}
        for criteria, desc in dims.items():
            evaluation[coarse_criteria][criteria] = responses[resp_i].content
            resp_i += 1

    return evaluation

def evaluate_support_models(merged_conversations_file='merged.json',
                            prompt_file='data/ESEval-multidim-cot.txt',
                            eval_model_name='o4-mini', results_file="output.json",
                            criteria_file='./data/exploration_rubric3.jsonl'):

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # load eval criteria
    criteria_dict = load_json(criteria_file)

    # Load evaluation prompt
    prompt_text = load_text(prompt_file)

    comparisons = json.load(open(merged_conversations_file))
    result = []

    for comp in tqdm(comparisons):

        judge = compare_models(comparison_obj=comp, prompt_text=prompt_text, eval_model_name=eval_model_name,
                               openai_api_key=openai_api_key, criteria_dict=criteria_dict)
        result.append(judge)

        with open(results_file, "w", encoding="utf-8") as file:
            json.dump(result, file)

    print(f"Saved evaluations in {results_file}")

if __name__ == "__main__":
    fire.Fire(evaluate_support_models)

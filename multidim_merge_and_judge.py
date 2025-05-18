import fire
import os
import re
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
from tqdm import tqdm


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


def parse_filename(filename):
    """
    Extracts model name and personality ID from the filename.
    Expected format: conv_<model_name>_<personality_id>.txt
    """
    match = re.match(r"(.+)_(.+)\.txt", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def group_conversations(directory):
    """
    Groups conversation files by personality ID.

    Args:
        directory (str): Path to the directory containing conversation files.

    Returns:
        dict: A dictionary where keys are personality IDs and values are lists of (model_name, file_path) tuples.
    """
    grouped_files = defaultdict(list)

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            model_name, personality_id = parse_filename(filename)
            if model_name and personality_id:
                grouped_files[personality_id].append((model_name, os.path.join(directory, filename)))

    return grouped_files

def evaluate_models_for_personality(personality_id, model_files, prompt_text, eval_model_name, openai_api_key,
                                    criteria_dict, batch_size=3):
    """
    Evaluates multiple models for the same personality.

    Args:
        personality_id (str): The unique ID representing a help-seeker personality.
        model_files (list): List of (model_name, file_path) tuples.
        prompt_text (str): Evaluation prompt content.
        eval_model_name (str): The OpenAI model used for evaluation.
        openai_api_key (str): OpenAI API Key.

    Returns:
        list: List containing evaluation results.
    """
    evaluations = []

    pbar = tqdm(total=len(model_files)*(len(model_files)-1)//2)
    for i in range(len(model_files)):
        for j in range(i + 1, len(model_files)):
            evaluation = {}
            model_a, file_a = model_files[i]
            model_b, file_b = model_files[j]

            # Load conversations
            conversation_a = load_text(file_a)
            conversation_b = load_text(file_b)

            prompts = []
            for coarse_criteria, dims in criteria_dict.items():
                for criteria, desc in dims.items():
                    prompts.append(prompt_text.format(
                        conversation_a=conversation_a,
                        conversation_b=conversation_b,
                        criteria=criteria,
                        criteria_description=desc,
                    ))

            # Initialize OpenAI model using LangChain
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
            evaluation['pid'] = personality_id
            evaluation['model_a'] = model_a
            evaluation['model_b'] = model_b

            resp_i = 0
            for coarse_criteria, dims in criteria_dict.items():
                evaluation[coarse_criteria] = {}
                for criteria, desc in dims.items():
                    evaluation[coarse_criteria][criteria] = responses[resp_i].content
                    resp_i += 1

            evaluations.append(evaluation)
            pbar.update(1)

    return evaluations

def evaluate_support_models(directory='output/exp2', prompt_file='data/ESEval-multidim-cot.txt',
                            eval_model_name='o1-mini', results_dir="output/evals_dim2",
                            criteria_file='./data/exploration_rubric3.jsonl'):
    """
    Evaluates multiple emotional support models on different help-seeker personalities.

    Args:
        directory (str): Path to the directory containing conversation files.
        prompt_file (str): Path to the prompt file.
        eval_model_name (str): OpenAI model used for evaluation.
        results_dir (str): Directory where results will be saved.
    """
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # load eval criteria
    criteria_dict = load_json(criteria_file)

    # Load evaluation prompt
    prompt_text = load_text(prompt_file)

    # Group conversations by personality
    grouped_conversations = group_conversations(directory)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for personality_id, model_files in tqdm(grouped_conversations.items()):
        if len(model_files) < 2:
            print(f"Skipping personality {personality_id}: Not enough models for comparison.")
            continue
        print(f"Evaluating personality: {personality_id} with {len(model_files)} models.")


        # Evaluate models
        evaluations = evaluate_models_for_personality(personality_id, model_files, prompt_text, eval_model_name,
                                                      openai_api_key, criteria_dict)

        # Save results to file
        result_file = os.path.join(results_dir, f"evaluation_{personality_id}.json")
        with open(result_file, "w", encoding="utf-8") as file:
            file.write(json.dumps(evaluations))

        print(f"Saved evaluation for Personality {personality_id} -> {result_file}")

if __name__ == "__main__":
    fire.Fire(evaluate_support_models)

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

def evaluate_models_for_personality(personality_id, model_files, prompt_text, eval_model_name, openai_api_key):
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

    for i in range(len(model_files)):
        for j in range(i + 1, len(model_files)):
            evaluation = {}
            model_a, file_a = model_files[i]
            model_b, file_b = model_files[j]

            # Load conversations
            conversation_a = load_text(file_a)
            conversation_b = load_text(file_b)

            # Format input for OpenAI
            full_prompt = f"""{prompt_text}

### Conversation with model A:
{conversation_a}

### Conversation with model B:
{conversation_b}

Please provide a structured evaluation comparing these two models as per the specified criteria.
"""

            # Initialize OpenAI model using LangChain
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=eval_model_name)

            # Generate assessment
            response = llm([SystemMessage(content="You are an expert evaluator of emotional support models."),
                            HumanMessage(content=full_prompt)])

            # Store evaluation result
            evaluation['evaluation'] = response.content
            evaluation['pid'] = personality_id
            evaluation['model_a'] = model_a
            evaluation['model_b'] = model_b
            evaluations.append(evaluation)

    return evaluations

def evaluate_support_models(directory='output/', prompt_file='data/ESEval-prompt.txt',
                            eval_model_name='gpt-4o', results_dir="output/evals"):
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
        evaluations = evaluate_models_for_personality(personality_id, model_files, prompt_text, eval_model_name, openai_api_key)

        # Save results to file
        result_file = os.path.join(results_dir, f"evaluation_{personality_id}.json")
        with open(result_file, "w", encoding="utf-8") as file:
            for eval in evaluations:
                file.write(json.dumps(eval, indent=4))

        print(f"Saved evaluation for Personality {personality_id} -> {result_file}")

if __name__ == "__main__":
    fire.Fire(evaluate_support_models)

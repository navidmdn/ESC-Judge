import numpy as np
from fire import Fire
from typing import Tuple
import os
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


def load_llama3_model(model_name: str, cache_dir=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto',
                                                 torch_dtype=torch.bfloat16)

    model = model.to(device)
    model.eval()

    return model, tokenizer


def get_model_response(messages: list, tokenizer, model, max_new_tokens: int = 256, do_sample: bool = True,
                       temperature: float = 0.7, ) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True
    )
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                            do_sample=do_sample, temperature=temperature)

    response_token_ids = output[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_token_ids, skip_special_tokens=True)

    return response

def simulate(persona1: str, persona2: str, scenario: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
             n_turns: int, temperature: float = 0.7, max_new_tokens: int = 256) -> None:

    person1_history = []
    person2_history = []

    person1_sys_msg = {'role': 'system', 'content': f"{persona1}"}
    person2_sys_msg = {'role': 'system', 'content': f"{persona2}"}

    cur_speaker = 1

    last_question = "Hey! how's it going?"

    print(f"speaker {3-cur_speaker}: {last_question}")
    print("*" * 100)
    for _ in range(n_turns):
        if cur_speaker == 1:
            messages = [
                person1_sys_msg,
                *person1_history,
                {'role': 'user', 'content': last_question}
            ]
            person1_history.append({'role': 'user', 'content': last_question})
            person2_history.append({'role': 'assistant', 'content': last_question})
            cur_speaker = 2
        else:
            messages = [
                person2_sys_msg,
                *person2_history,
                {'role': 'user', 'content': last_question}
            ]
            person2_history.append({'role': 'user', 'content': last_question})
            person1_history.append({'role': 'assistant', 'content': last_question})
            cur_speaker = 1

        response = get_model_response(messages, tokenizer, model, max_new_tokens=max_new_tokens, temperature=temperature)
        print(f"speaker {3-cur_speaker}: {response}")
        print("*" * 100)
        last_question = response



def run(persona1_file: str = 'p1-test.txt', persona2_file: str = 'p2-test.txt', scenario_file: str = 'scenario-test.txt',
        n_iters: int = 1, llm_name: str = 'gpt-4o', temperature=0.7, max_new_tokens=512, cache_dir=None) -> None:


    with open(persona1_file, 'r') as f:
        persona1 = f.read().strip()

    with open(persona2_file, 'r') as f:
        persona2 = f.read().strip()

    # with open(scenario_file, 'r') as f:
    #     scenario = f.read().strip()
    scenario = ""

    model, tokenizer = load_llama3_model(llm_name, cache_dir=cache_dir)

    for i in range(n_iters):
        simulate(persona1, persona2, scenario, model=model, tokenizer=tokenizer, n_turns=10, temperature=temperature,
                 max_new_tokens=max_new_tokens)


if __name__ == '__main__':
    Fire(run)
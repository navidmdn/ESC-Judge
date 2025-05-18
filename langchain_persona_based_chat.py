from fire import Fire
import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from utils.langchain import load_hf_auto_regressive_model
from langchain_openai import ChatOpenAI
from typing import List, Dict
from functools import partial
import json

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')

SEEKER_PROMPT_TEMPLATE = """Here is a description of your role:\n
{description}

You are talking to an emotional support expert about your ongoing challenge. Focus on your problem and engage in the conversation.\
 make sure you are consistent with your designated behavioral traits and role description.
"""

def save_chat(chat_history: List[Dict], output_file: str) -> None:
    with open(output_file, 'w') as f:
        for msg in chat_history:
            f.write(f"{msg['role']}: {msg['content']}\n")


def load_model(model_name, hf, temperature, cache_dir, max_new_tokens, load_in_4bit, openai_api_key=None):
    if 'llama' in model_name or 'mistral' in model_name:
        if hf:
            llm, tokenizer = load_hf_auto_regressive_model(model_name, max_new_tokens=max_new_tokens, load_in_4bit=load_in_4bit,
                                                          cache_dir=cache_dir)
            return llm, tokenizer
        else:
            raise ValueError("LLM not implemented")
    elif 'gpt' in model_name:
        llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE,
                                   max_tokens=max_new_tokens, temperature=temperature, )
        return llm, None
    else:
        raise ValueError("LLM not implemented")


def load_prompt_constructor(model_name, tokenizer, hf=False):
    if 'llama' in model_name or 'mistral' in model_name:
        if hf:
            return partial(get_llama_chat_prompt, tokenizer=tokenizer)
        else:
            raise ValueError("LLM not implemented")
    elif 'gpt' in model_name:
        return get_openai_chat_prompt
    else:
        raise ValueError("LLM not implemented")

def pprint(role, content, visualization_socket=None):

    if visualization_socket:
        visualization_socket.emit('new_message', {"role": role, "content": content})

    print('-' * 100)
    print("*" * 100)
    print('-' * 100)
    print(f"{role}: {content}")


def get_openai_chat_prompt(args):
    conversation= args['conversation']
    system_message = args['system_msg']

    prompt_tmp = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        *conversation,
    ])

    return prompt_tmp


def get_llama_chat_prompt(args, tokenizer):
    conversation = args['conversation']
    system_message = args['system_msg']

    history = [
        {'role': 'system', 'content': system_message},
        *conversation
    ]
    prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)

    return prompt


def simulate(supporter_persona: str, seeker_persona: str, supporter_llm: Runnable, seeker_llm: Runnable, n_turns: int,
             seeker_prompt_constructor = None, supporter_prompt_constructor = None, visualization_socket = None) -> List[Dict]:
    """

    Simulate a conversation between two speakers based on their personas and a scenario.

    """

    recorded_history = []

    supporter_history = []
    seeker_history = []

    supporter_sys_msg = supporter_persona
    seeker_sys_msg = SEEKER_PROMPT_TEMPLATE.format(description=seeker_persona)

    cur_speaker = "supporter"
    last_question = "Hey! how's it going?"
    pprint(role=cur_speaker, content=last_question, visualization_socket=visualization_socket)
    recorded_history.append({"role": f"{cur_speaker}", "content": last_question})

    # todo: implement a simple end of conversation detection model
    for _ in range(n_turns):
        if cur_speaker == 'seeker':

            supporter_history.append({'role': 'user', 'content': last_question})
            seeker_history.append({'role': 'assistant', 'content': last_question})

            cur_speaker = 'supporter'
        else:

            supporter_history.append({'role': 'assistant', 'content': last_question})
            seeker_history.append({'role': 'user', 'content': last_question})

            cur_speaker = 'seeker'

        if cur_speaker == 'seeker':
            chain = seeker_prompt_constructor | seeker_llm | StrOutputParser()
            response = chain.invoke({"conversation": seeker_history, "system_msg": seeker_sys_msg})
        else:
            chain = supporter_prompt_constructor | supporter_llm | StrOutputParser()
            response = chain.invoke({"conversation": supporter_history, "system_msg": supporter_sys_msg})

        pprint(role=cur_speaker, content=response, visualization_socket=visualization_socket)
        recorded_history.append({"role": f"{cur_speaker}", "content": response})

        last_question = response

    return recorded_history



def run_simulation(supporter_persona_file: str = 'p1-test.txt', seeker_personas_file: str = 'data/test_persona.json',
                   supporter_llm_name: str = 'gpt-4o', seeker_llm_name: str = 'gpt-4o', temperature=0.8, max_new_tokens=4096,
                   hf=False, load_in_4bit=False, visualization_socket=None,
                   cache_dir=None, output_prefix="conv") -> None:


    with open(supporter_persona_file, 'r') as f:
        supporter_persona_txt = f.read().strip()

    seeker_persona_list = []
    with open(seeker_personas_file, 'r') as f:
        for line in f:
            seeker_persona_list.append(json.loads(line.strip()))


    supporter_llm, supporter_tokenizer = load_model(supporter_llm_name, hf, temperature, cache_dir, max_new_tokens, load_in_4bit)
    seeker_llm, seeker_tokenizer = load_model(seeker_llm_name, hf, temperature, cache_dir, max_new_tokens, load_in_4bit)

    seeker_prompt_constructor = load_prompt_constructor(seeker_llm_name, seeker_tokenizer, hf)
    supporter_prompt_constructor = load_prompt_constructor(supporter_llm_name, supporter_tokenizer, hf)

    for seeker_persona_obj in seeker_persona_list:
        seeker_persona_txt = seeker_persona_obj['role']
        pid = seeker_persona_obj['pid']

        conv = simulate(supporter_persona_txt, seeker_persona_txt, supporter_llm, seeker_llm, n_turns=14,
                        seeker_prompt_constructor=seeker_prompt_constructor,
                        supporter_prompt_constructor=supporter_prompt_constructor,
                        visualization_socket=visualization_socket)
        save_chat(conv, f"output/{output_prefix}_{supporter_llm_name}_{pid}.txt".replace('/', '-'))


if __name__ == '__main__':
    Fire(run_simulation)
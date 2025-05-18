
import os
import time
import fire
from datetime import datetime
import numpy as np
import json
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
import re
from uuid import uuid4


from utils.misc import load_jsonl
from sub_characteristic_prompts import (basic_family_life_stage_prompt, key_life_events_prompt,
                                        sample_behavior_trait_string, generate_the_ongoing_challenge_prompt,
                                        generate_full_role)


# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'


def generate_data(n_iters=10, llm_name='gpt-4o', save_batch_size=2, output_dir='data/',
                  temperature=0.7, max_tokens=4096, output_file_name='generated_personas.json',
                  personas_file='data/simple_personas.json'):

    os.makedirs(output_dir, exist_ok=True)

    if 'gpt' in llm_name:
        llm = ChatOpenAI(
            model_name=llm_name,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError("LLM not implemented")

    personas_objs = load_jsonl(personas_file)

    def key_based_extractor(response, key='final persona:'):
        response = response.strip().lower()
        assert key in response, f"Final persona not found in response: {response}"
        persona = response.split(key)[-1]
        return persona

    def persona_extractor(response):
        return key_based_extractor(response, key='final persona:')

    def key_life_events_extractor(response):
        return key_based_extractor(response, key='key events:')

    def ongoing_challenge_extractor(response):
        return key_based_extractor(response, key='ongoing challenge:')

    def full_role_extractor(response):
        return key_based_extractor(response, key='system prompt:')

    persona_extractor_chain = basic_family_life_stage_prompt | llm | StrOutputParser() | persona_extractor
    life_event_extractor_chain = key_life_events_prompt | llm | StrOutputParser() | key_life_events_extractor
    ongoing_challenge_extractor_chain = generate_the_ongoing_challenge_prompt | llm | StrOutputParser() | ongoing_challenge_extractor
    full_role_extractor_chain = generate_full_role | llm | StrOutputParser() | full_role_extractor

    results = []
    cost = 0
    output_path = os.path.join(output_dir, output_file_name)
    pbar = tqdm(total=n_iters)

    total_family_statuses = 5

    # max number of events to list by llm
    total_events = 25

    # max number of events to choose for a person
    max_event_choices = 5

    # max number of challenges the llm generates
    max_challenges = 50

    i = 0
    while i < n_iters:

        try:
            with get_openai_callback() as cb:
                Nf = np.random.randint(1, total_family_statuses+1, size=save_batch_size)
                N_chosen_events = np.random.randint(1, max_event_choices+1, size=save_batch_size)
                elist = [str(list(np.random.choice(total_events, size=n))) for n in N_chosen_events]
                personas = list(np.random.choice(personas_objs, size=save_batch_size, replace=False))

                assert len(personas) == save_batch_size == len(Nf) == len(elist)

                final_personas = persona_extractor_chain.batch([
                    {'persona': persona,
                     'Nf': nf,
                     'Nf_total': total_family_statuses,
                     } for persona, nf in zip(personas, Nf)]
                )

                life_events = life_event_extractor_chain.batch([
                    {'final_persona': final_persona, 'total_events': total_events, 'elist': e}
                    for final_persona, e in zip(final_personas, elist)]
                )

                behavioral_traits = sample_behavior_trait_string()

                chosen_challenge_idxs = np.random.randint(1, max_challenges+1, size=save_batch_size)
                ongoing_challenges = ongoing_challenge_extractor_chain.batch([
                    {'final_persona': final_persona, 'key_events': life_event, 'total_challenges': max_challenges, 'chosen_challenge': idx}
                    for final_persona, life_event, idx in zip(final_personas, life_events, chosen_challenge_idxs)]
                )

                full_roles = full_role_extractor_chain.batch([
                    {'final_persona': final_persona, 'key_events': life_event, 'behavioral_traits': behavioral_traits, 'ongoing_challenges': ongoing_challenge}
                    for final_persona, life_event, ongoing_challenge in zip(final_personas, life_events, ongoing_challenges)]
                )

                for full_role, final_persona, life_event, challenge in zip(full_roles, final_personas, life_events, ongoing_challenges):
                    results.append({
                        'role': full_role.strip(),
                        'pid': str(uuid4()),
                        'life_events': life_event,
                        'challenge': challenge,
                        'final_persona': final_persona
                    })

                cost += cb.total_cost
            print("Accumulated cost: ", cost)
        except Exception as e:
            print(e)
            time.sleep(5*60)
            output_path = os.path.join(output_dir, f'{output_file_name}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json')
            continue

        i += save_batch_size
        pbar.update(save_batch_size)
        print("total cost: ", cost)

        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    fire.Fire(generate_data)
# ESC-Judge

This is the repository for the paper titled "ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents". We go over main stages of constructing different help seeker background and personas, simulating conversation with agents under test and evaluating their performance based on Hill's emotional support book.


## Constructing Help Seeker Personas

make sure you set your openAI api key for each of the commands or export it before running the commands as:
```
export OPENAI_API_KEY="your api key here"
```

Then you can run the role construction by the following command:

```
python characteristic_development_from_challenge.py \
 --n_iters=<number of roles you want to construct> \
 --llm_name=<openai model name - the default is o4-mini> \
 --save_batch_size=<batch size for batched call and save> \
 --output_file_name=<the output json file path>
```

## ES Conversation Simulation

For emotional support conversation simulation between a **model under test** and the **simulated help seeker** you can use either a local llm using langchain and huggingface models or use an API by utilizing `langchain_persona_based_chat.py` file. `exp1.sh` and `exp2.sh` files showcase the experiments done in the paper. Here's a sample command for simulating conversations between constructed roles prompted by `gpt-4o-mini` as the help seeker and `llama-3.2-3B-Instruct` as the ESC model under test:

```
python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/roles-v1.json --supporter_llm_name meta-llama/Llama-3.2-3B-Instruct\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o-mini --output_prefix 'exp2-nodir'
```

## Judging and Comparing Different Models

For judging between two models based on the outputs generated from previous stage and rubric files inside `data/` directory you can use the `multidim_merge_and_judge.py` file which merges different model outputs based on the role ids and compares the conversations head-to-head per judging dimension.

```
python multidim_merge_and_judge.py \
  --directory <directory of which the conversation files are stored in> \
  --prompt_file data/ESEval-multidim-cot.txt \
  --eval_model_name o1-mini \
  --results_dir output/evals_dim2 \
  --criteria_file data/exploration_rubric3.jsonl
```

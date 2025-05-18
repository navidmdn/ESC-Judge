python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/roles-v1.json --supporter_llm_name meta-llama/Llama-3.2-3B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o-mini --output_prefix 'exp2-nodir'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/roles-v1.json --supporter_llm_name meta-llama/Llama-3.1-8B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o-mini --output_prefix 'exp2-nodir'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/roles-v1.json --supporter_llm_name gpt-4o-mini --seeker_llm_name gpt-4o-mini --output_prefix 'exp2-nodir'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_hill.txt\
  --seeker_personas_file data/roles-v1.json --supporter_llm_name meta-llama/Llama-3.2-3B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o-mini --output_prefix 'exp2-hillprompt'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_hill.txt\
  --seeker_personas_file data/roles-v1.json --supporter_llm_name meta-llama/Llama-3.1-8B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o-mini --output_prefix 'exp2-hillprompt'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_hill.txt\
  --seeker_personas_file data/roles-v1.json --supporter_llm_name gpt-4o-mini --seeker_llm_name gpt-4o-mini --output_prefix 'exp2-hillprompt'





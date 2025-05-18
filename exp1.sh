python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name meta-llama/Llama-3.2-3B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o --output_prefix 'nodir'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name meta-llama/Llama-3.1-8B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o --output_prefix 'nodir'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name gpt-4o --seeker_llm_name gpt-4o --output_prefix 'nodir'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_nodir.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name gpt-4o-mini --seeker_llm_name gpt-4o --output_prefix 'nodir'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_hill.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name meta-llama/Llama-3.2-3B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o --output_prefix 'hillprompt'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_hill.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name meta-llama/Llama-3.1-8B-Instruct --cache_dir ../../hfcache\
  --hf --max_new_tokens 500 --seeker_llm_name gpt-4o --output_prefix 'hillprompt'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_hill.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name gpt-4o --seeker_llm_name gpt-4o --output_prefix 'hillprompt'

python langchain_persona_based_chat.py --supporter_persona_file data/emotional_supporter_hill.txt\
  --seeker_personas_file data/generated_personas.json --supporter_llm_name gpt-4o-mini --seeker_llm_name gpt-4o --output_prefix 'hillprompt'





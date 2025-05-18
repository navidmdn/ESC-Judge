python langchain_persona_based_chat.py --supporter_persona data/emotional_supporter_hill.txt\
  --seeker_personas data/test_persona.json --supporter_llm_name gpt-4o
##
#python langchain_persona_based_chat.py --supporter_persona data/emotional_supporter_hill.txt\
#  --seeker_personas data/generated_personas.json --supporter_llm_name gpt-4o-mini

#python langchain_persona_based_chat.py --supporter_persona data/emotional_supporter_helpful.txt\
#  --seeker_personas data/test_persona.json --supporter_llm_name meta-llama/Llama-3.1-8B-Instruct --cache_dir ../../hfcache\
#  --hf --max_new_tokens 500 --seeker_llm_name meta-llama/Llama-3.1-70B-Instruct --load_in_4bit
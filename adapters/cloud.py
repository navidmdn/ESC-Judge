import os
import time
from openai import OpenAI
from typing import List, Dict, Optional
from adapters.local_llama import GenerationResult


class CloudAdapter:

    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model_name = model_name
        
        self.client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url,
        )

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7, max_tokens: int = 512) -> GenerationResult:
        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.time() - t0
            choice = response.choices[0]
            usage = response.usage
            return GenerationResult(
                text=choice.message.content,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                wall_clock_seconds=elapsed,
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                wall_clock_seconds=time.time() - t0,
                error=str(e),
            )
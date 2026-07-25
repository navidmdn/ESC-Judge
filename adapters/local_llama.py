# A thin wrapper that starts llama-server as a subprocess pointing at a GGUF file,
# waits for it to be ready, and exposes a chat() method that hits the OPENAI - compatible /v1/chat/completion endpoint.
# Also a stop() method to kill the server
# It replaces the HuggingFace loadl_model path entirely
# The adapter returns raw structured data (text, token counts, latency) - No LangChain is the candidate path.

import os
import socket
import subprocess
import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    # llama.cpp native timings
    prefill_ms: float = 0.0
    predicted_ms: float = 0.0
    prefill_per_token_ms: float = 0.0
    predicted_per_token_ms: float = 0.0
    ttft_s: float = 0.0
    tokens_per_s: float = 0.0
    llm_inference_s: float = 0.0
    # wall-clock including HTTP overhead
    wall_clock_seconds: float = 0.0
    error: Optional[str] = None


class LlamaServerAdapter:

    def __init__(self, model_path: str, port: int = 8080, server_binary: str = "llama-server",
                 n_ctx: int = 4096, n_gpu_layers: int = -1, extra_args: Optional[List[str]] = None ):
        self.model_path = model_path
        self.server_binary = server_binary
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen] = None
        self.cold_start_result: Optional[GenerationResult] = None

    def start(self, timeout: int = 120):
        """Launch llama-server and block until it's healthy."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(
                    f"Port {self.port} already in use. "
                    f"Kill the existing process: kill $(lsof -ti :{self.port})"
                )

        cmd = [
            self.server_binary,
            "-m", self.model_path,
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            *self.extra_args,
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._wait_until_ready(timeout)
        self._verify_model()
        self._warmup()

    def _wait_until_ready(self, timeout: int):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        self.stop()
        raise TimeoutError(
            f"llama-server failed to start within {timeout}s for {self.model_path}"
        )

    def _verify_model(self):
        """Confirm the server loaded the expected model."""
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5)
            r.raise_for_status()
            loaded_model = r.json()["data"][0]["id"]
            expected = os.path.basename(self.model_path)
            if expected not in loaded_model:
                self.stop()
                raise RuntimeError(
                    f"Port {self.port} is serving '{loaded_model}', "
                    f"not '{self.model_path}'. "
                    f"Kill the existing process: kill $(lsof -ti :{self.port})"
                )
        except requests.RequestException:
            pass

    def _warmup(self):
        """Send a throwaway prompt to measure cold start and warm the cache."""
        self.cold_start_result = self.chat(
            messages=[
                {"role": "user", "content": "Cold start setup, ignore this message."}
            ],
            temperature=0.0,
            max_tokens=16,
        )

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7, max_tokens: int = 512) -> GenerationResult:
        """Send a chat completion request. Returns structured result with native timings."""
        t0 = time.time()
        try:
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            wall_clock = time.time() - t0

            choice = data["choices"][0]
            usage = data.get("usage", {})
            timings = data.get("timings", {})

            prefill_ms = timings.get("prompt_ms", 0.0)
            predicted_ms = timings.get("predicted_ms", 0.0)
            llm_inference_s = (prefill_ms + predicted_ms) / 1000.0

            return GenerationResult(
                text=choice["message"]["content"],
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                prefill_ms=prefill_ms,
                predicted_ms=predicted_ms,
                prefill_per_token_ms=timings.get("prompt_per_token_ms", 0.0),
                predicted_per_token_ms=timings.get("predicted_per_token_ms", 0.0),
                ttft_s=timings.get("prompt_ms", 0.0) / 1000.0,
                tokens_per_s=timings.get("predicted_per_second", 0.0),
                llm_inference_s=llm_inference_s,
                wall_clock_seconds=wall_clock,
            )
        except Exception as e:
            return GenerationResult(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                wall_clock_seconds=time.time() - t0,
                error=str(e),
            )

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
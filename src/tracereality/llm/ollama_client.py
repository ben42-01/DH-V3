"""Ollama API client for generating LLM responses.

Wraps the Ollama REST API with support for model selection,
temperature, top-p, max tokens, GPU throttling, and context
window control.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests


class OllamaClient:
    """Client for the Ollama API.

    Supports generating text from any Ollama model, with configurable
    generation parameters and GPU throttling.

    Throttling: by default waits 2s between requests so the GPU can
    cool down between calls. Adjust ``request_delay`` for your system.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        timeout: int = 120,
        num_ctx: int = 512,
        cooldown: float = 3.0,
    ):
        """Initialize the Ollama client.

        Each ``generate()`` call is **synchronous** — it waits for
        the full response before returning. This means requests are
        naturally serialized.

        GPU throttling: after each response, the client sleeps for
        ``cooldown`` seconds. This gives the GPU a rest between
        inference calls, preventing sustained 100% load. Increase
        ``cooldown`` (e.g. 5-10s) if the GPU is overheating.

        Args:
            base_url: Ollama server URL.
            model: Model name (e.g., 'llama3', 'mistral', 'gemma').
            temperature: Sampling temperature (0.0 = deterministic).
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            num_ctx: Context window size in tokens. Lower = less GPU
                compute per call. 512 is fine for short trace
                fragments; increase to 2048+ for long prompts.
            cooldown: Seconds to rest between LLM calls. The GPU runs
                at 100% during inference; this idle period lets it
                cool down. Default 3s. Set to 0 for no cooldown.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.cooldown = cooldown

    def generate(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Calls are **synchronous** — ``requests.post()`` blocks until
        the full response is received. This ensures one LLM call at a
        time, with no queuing.

        Args:
            system_message: System prompt / context.
            user_prompt: User message / trace fragment.
            temperature: Override temperature for this call.
            max_tokens: Override max tokens for this call.

        Returns:
            Generated text response.
        """
        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "system": system_message,
            "prompt": user_prompt,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "options": {
                "num_ctx": self.num_ctx,
            },
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "").strip()

            # GPU cooldown: sleep after each call so the GPU
            # gets idle time between inference bursts.
            if self.cooldown > 0:
                time.sleep(self.cooldown)

            return result
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? (Start it with: ollama serve)"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                "The model may be too large or the server is busy."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def list_models(self) -> list[dict]:
        """List available models from Ollama.

        Returns:
            List of model info dicts.
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list Ollama models: {e}")

    def check_available(self, model: str | None = None) -> bool:
        """Check if a model is available locally.

        Performs exact match first, then checks if the model name
        matches as a prefix (e.g., 'llama3.1' matches 'llama3.1:8b').

        Args:
            model: Model name to check. If None, checks self.model.

        Returns:
            True if the model is available.
        """
        check = model or self.model
        models = self.list_models()
        for m in models:
            name = m.get("name", "")
            if name == check:
                return True
            if name.startswith(check + ":") or name.startswith(check + "@"):
                return True
        return False
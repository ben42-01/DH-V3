"""OpenRouter API client — drop-in replacement for OllamaClient.

Uses OpenRouter's API to access frontier models (GPT-4o-mini, Claude,
Gemini, etc.) without needing local GPU. Many models have free tiers.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    export OPENROUTER_MODEL="gpt-4o-mini"  # optional, default below

    client = OpenRouterClient(model="gpt-4o-mini")
    response = client.generate("You are a predictor.", "Sequence: 1, 2, 3")
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests


# Recommended cheap models for next-token prediction
# Prices on OpenRouter: llama-3.1-8b $0.02/M, gpt-4o-mini $0.15/M
RECOMMENDED_MODELS = [
    ("meta-llama/llama-3.1-8b-instruct", "8B params, 128K ctx, $0.02/M — best value"),
    ("openai/gpt-4o-mini", "Good pattern recognition, $0.15/M"),
    ("mistralai/mistral-nemo", "12B, 128K ctx, $0.02/M"),
    ("meta-llama/llama-3-8b-instruct", "8B, 8K ctx, $0.04/M"),
    ("microsoft/wizardlm-2-8x22b", "Huge but $0.62/M"),
    ("anthropic/claude-3-haiku", "Fast, $0.25/M"),
]


class OpenRouterClient:
    """Client for the OpenRouter API.

    Same interface as OllamaClient — drop-in replacement.
    Cooldown: waits between calls to avoid rate limits.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 16,
        timeout: int = 30,
        cooldown: float = 0.5,
    ):
        """Initialize the OpenRouter client.

        Args:
            model: Model identifier (e.g. "gpt-4o-mini", "claude-3-haiku").
                If None, reads from OPENROUTER_MODEL env var, or defaults
                to "gpt-4o-mini".
            api_key: OpenRouter API key. If None, reads from
                OPENROUTER_API_KEY env var.
            temperature: Sampling temperature.
            max_tokens: Max tokens in response.
            timeout: Request timeout in seconds.
            cooldown: Seconds to rest between API calls (rate limit safety).
        """
        self.model = (
            model
            or os.environ.get("OPENROUTER_MODEL")
            or "gpt-4o-mini"
        )
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY env var "
                "or pass api_key to the constructor."
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cooldown = cooldown

    def generate(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 5,
    ) -> str:
        """Generate a response from the LLM via OpenRouter.

        Retries on 429 (rate limit) with exponential backoff.
        All variants share one client, so requests are serial.

        Args:
            system_message: System prompt / role definition.
            user_prompt: User message / trace fragment.
            temperature: Override temperature for this call.
            max_tokens: Override max tokens for this call.
            max_retries: Max retries on rate limit errors.

        Returns:
            Generated text response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/benjamin/dh-v3",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    # Rate limited: wait with exponential backoff
                    wait = (2 ** attempt) * 2  # 2s, 4s, 8s, 16s, 32s
                    print(f"  [rate limited] retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Cooldown to avoid rate limits
                if self.cooldown > 0:
                    time.sleep(self.cooldown)

                return content

            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    "Cannot connect to OpenRouter. Check your internet connection."
                )
            except requests.exceptions.Timeout:
                raise RuntimeError(
                    f"OpenRouter request timed out after {self.timeout}s."
                )
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    last_error = e
                    continue  # Will retry
                raise RuntimeError(
                    f"OpenRouter API error: {e.response.status_code} — "
                    f"{e.response.text[:200]}"
                )
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected OpenRouter response format: {e}")

        # If we exhausted retries
        raise RuntimeError(
            f"OpenRouter 429 after {max_retries} retries. "
            f"Last error: {last_error.response.text[:200] if last_error else 'unknown'}"
        )

    def list_recommended_models(self) -> list[tuple[str, str]]:
        """List recommended cheap models with descriptions."""
        return list(RECOMMENDED_MODELS)

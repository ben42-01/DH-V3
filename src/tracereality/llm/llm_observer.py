"""LLM Observer — one variant of an LLM-based observer.

Each observer has a system message and prompt template that can be
evolved (mutated) over generations. The observer generates free-form
text responses to trace fragments.
"""

from __future__ import annotations

import copy
import json
from typing import List, Optional

from .ollama_client import OllamaClient


class LLMObserver:
    """An LLM observer variant with evolvable system message and prompt template.

    Each observer:
    - Has a unique ID
    - Has a system message (evolvable)
    - Has a prompt template (evolvable)
    - Uses an OllamaClient to generate responses
    - Tracks fitness history
    """

    def __init__(
        self,
        observer_id: int,
        system_message: str,
        prompt_template: str,
        model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 256,
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout: int = 120,
        num_ctx: int = 512,
    ):
        """Initialize an LLM observer.

        Args:
            observer_id: Unique identifier.
            system_message: System prompt/context for the LLM.
            prompt_template: Template for the user prompt. Should contain
                '{trace_fragment}' as a placeholder.
            model: Ollama model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            ollama_base_url: Ollama server URL.
            ollama_timeout: Request timeout in seconds.
            num_ctx: Context window size in tokens. Lower = less GPU
                compute per inference call. 512 is fine for short traces.
        """
        self.id = observer_id
        self.system_message = system_message
        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx

        self.client = OllamaClient(
            base_url=ollama_base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=ollama_timeout,
            num_ctx=num_ctx,
        )

        self.fitness_history: List[float] = []
        self.output_history: List[str] = []

    def respond(self, trace_fragment: str) -> str:
        """Generate a response to a trace fragment.

        Args:
            trace_fragment: String representation of the trace fragment
                (e.g., "3, 7, 2, 9, 1, 1, 8").

        Returns:
            Free-form text response from the LLM.
        """
        user_prompt = self.prompt_template.replace("{trace_fragment}", trace_fragment)
        response = self.client.generate(
            system_message=self.system_message,
            user_prompt=user_prompt,
        )
        self.output_history.append(response)
        return response

    def respond_to_trace_list(self, trace_states: list[int]) -> str:
        """Generate a response to a list of state indices.

        Converts the list to a string and calls respond().

        Args:
            trace_states: List of state indices.

        Returns:
            Free-form text response.
        """
        fragment = ", ".join(str(s) for s in trace_states)
        return self.respond(fragment)

    def mutate_prompt_template(self, mutation_rate: float = 0.3) -> None:
        """Mutate the prompt template by making small text changes.

        Mutation operations:
        - Word insertion (add a word like "please", "now", "carefully")
        - Word deletion (remove a word)
        - Synonym replacement
        - Punctuation changes

        Args:
            mutation_rate: Probability of applying a mutation.
        """
        import random
        import re

        if random.random() > mutation_rate:
            return

        template = self.prompt_template
        words = template.split()

        if not words:
            return

        op = random.choice(["insert", "delete", "swap", "punctuation"])

        if op == "insert" and len(words) < 50:
            insert_words = [
                "please", "now", "carefully", "briefly", "quickly",
                "simply", "naturally", "casually",
            ]
            insert_word = random.choice(insert_words)
            pos = random.randint(0, len(words))
            words.insert(pos, insert_word)
            template = " ".join(words)

        elif op == "delete" and len(words) > 5:
            # Don't delete the {trace_fragment} placeholder
            non_placeholder_indices = [
                i for i, w in enumerate(words)
                if "{trace_fragment}" not in w
            ]
            if non_placeholder_indices:
                idx = random.choice(non_placeholder_indices)
                words.pop(idx)
                template = " ".join(words)

        elif op == "swap" and len(words) > 3:
            idx1, idx2 = random.sample(range(len(words)), 2)
            # Don't swap the placeholder
            if "{trace_fragment}" not in words[idx1] and "{trace_fragment}" not in words[idx2]:
                words[idx1], words[idx2] = words[idx2], words[idx1]
                template = " ".join(words)

        elif op == "punctuation":
            if template.endswith("."):
                template = template[:-1] + "?"
            elif template.endswith("?"):
                template = template[:-1] + "..."
            elif template.endswith("..."):
                template = template[:-3] + "."
            else:
                template = template + "."

        self.prompt_template = template

    def mutate_system_message(self, mutation_rate: float = 0.2) -> None:
        """Mutate the system message.

        Similar operations as prompt template mutation, but more conservative
        since the system message defines the observer's core behavior.

        Args:
            mutation_rate: Probability of applying a mutation.
        """
        import random

        if random.random() > mutation_rate:
            return

        msg = self.system_message
        words = msg.split()

        if not words:
            return

        op = random.choice(["insert", "swap"])

        if op == "insert" and len(words) < 40:
            insert_words = [
                "observe", "naturally", "spontaneously", "casually",
                "freely", "openly", "simply", "organically",
            ]
            insert_word = random.choice(insert_words)
            pos = random.randint(0, len(words))
            words.insert(pos, insert_word)
            msg = " ".join(words)

        elif op == "swap" and len(words) > 3:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            msg = " ".join(words)

        self.system_message = msg

    def mutate(self, mutation_rate: float = 0.3) -> None:
        """Apply all mutation operations.

        Args:
            mutation_rate: Probability of each mutation being applied.
        """
        self.mutate_prompt_template(mutation_rate)
        self.mutate_system_message(mutation_rate * 0.7)  # System mutations less frequent

    def clone(self, new_id: int) -> "LLMObserver":
        """Create a deep copy of this observer with a new ID.

        Args:
            new_id: ID for the clone.

        Returns:
            A new LLMObserver with identical settings.
        """
        return LLMObserver(
            observer_id=new_id,
            system_message=copy.deepcopy(self.system_message),
            prompt_template=copy.deepcopy(self.prompt_template),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            ollama_base_url=self.client.base_url,
            ollama_timeout=self.client.timeout,
            num_ctx=self.num_ctx,
        )

    def record_fitness(self, fitness: float) -> None:
        """Record a fitness value."""
        self.fitness_history.append(fitness)

    @property
    def current_fitness(self) -> float:
        return self.fitness_history[-1] if self.fitness_history else 0.0

    @property
    def avg_fitness(self) -> float:
        if not self.fitness_history:
            return 0.0
        return sum(self.fitness_history) / len(self.fitness_history)

    def to_dict(self) -> dict:
        """Serialize observer state to a dictionary."""
        return {
            "id": self.id,
            "system_message": self.system_message,
            "prompt_template": self.prompt_template,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "fitness_history": self.fitness_history,
            "recent_outputs": self.output_history[-5:] if self.output_history else [],
        }
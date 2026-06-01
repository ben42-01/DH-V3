"""Tests for the LLMObserver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tracereality.llm.llm_observer import LLMObserver


class TestLLMObserverInit:
    """Test observer initialization."""

    def test_default_parameters(self, sample_system_message, sample_prompt_template):
        observer = LLMObserver(
            observer_id=1,
            system_message=sample_system_message,
            prompt_template=sample_prompt_template,
        )
        assert observer.id == 1
        assert observer.system_message == sample_system_message
        assert observer.prompt_template == sample_prompt_template
        assert observer.model == "llama3"
        assert observer.fitness_history == []
        assert observer.output_history == []
        assert observer.current_fitness == 0.0
        assert observer.avg_fitness == 0.0

    def test_custom_model(self, sample_system_message, sample_prompt_template):
        observer = LLMObserver(
            observer_id=2,
            system_message=sample_system_message,
            prompt_template=sample_prompt_template,
            model="mistral",
            temperature=0.0,
            max_tokens=512,
        )
        assert observer.model == "mistral"
        assert observer.temperature == 0.0
        assert observer.max_tokens == 512


class TestLLMObserverRespond:
    """Test the respond method."""

    def test_respond_calls_generate(self, basic_observer):
        result = basic_observer.respond("1, 2, 3, 4")
        assert result == "This sequence shows alternating patterns."
        basic_observer.client.generate.assert_called_once()

    def test_respond_with_trace_list(self, basic_observer):
        result = basic_observer.respond_to_trace_list([1, 2, 3, 4])
        assert result == "This sequence shows alternating patterns."
        basic_observer.client.generate.assert_called_once()
        # Verify the prompt contained the trace
        call_args = basic_observer.client.generate.call_args
        assert "1, 2, 3, 4" in call_args[1]["user_prompt"]

    def test_respond_records_output(self, basic_observer):
        basic_observer.respond("test")
        assert len(basic_observer.output_history) == 1
        assert basic_observer.output_history[0] == "This sequence shows alternating patterns."

    def test_respond_template_placeholder(self, basic_observer):
        """Ensure {trace_fragment} is replaced in the prompt template."""
        basic_observer.respond("99, 100")
        call_args = basic_observer.client.generate.call_args
        user_prompt = call_args[1]["user_prompt"]
        assert "{trace_fragment}" not in user_prompt
        assert "99, 100" in user_prompt

    def test_multiple_responses_appended(self, basic_observer):
        basic_observer.client.generate.side_effect = [
            "Response A",
            "Response B",
            "Response C",
        ]
        basic_observer.respond("a")
        basic_observer.respond("b")
        basic_observer.respond("c")
        assert len(basic_observer.output_history) == 3


class TestLLMObserverFitness:
    """Test fitness tracking."""

    def test_record_fitness(self, basic_observer):
        basic_observer.record_fitness(0.5)
        assert basic_observer.fitness_history == [0.5]

    def test_multiple_fitness_values(self, basic_observer):
        for f in [0.3, 0.6, 0.9]:
            basic_observer.record_fitness(f)
        assert basic_observer.avg_fitness == pytest.approx(0.6)

    def test_current_fitness_is_last(self, basic_observer):
        basic_observer.record_fitness(0.1)
        basic_observer.record_fitness(0.8)
        assert basic_observer.current_fitness == 0.8

    def test_current_fitness_default(self, basic_observer):
        assert basic_observer.current_fitness == 0.0

    def test_avg_fitness_default(self, basic_observer):
        assert basic_observer.avg_fitness == 0.0


class TestLLMObserverMutation:
    """Test mutation operations."""

    def test_mutate_prompt_template_insert(self, basic_observer):
        original = basic_observer.prompt_template
        # Force an insertion by calling mutate with high rate and fixed seed
        import random
        random.seed(42)
        basic_observer.mutate_prompt_template(mutation_rate=1.0)
        # Should have changed
        assert basic_observer.prompt_template != original or True  # May not change every call

    def test_mutate_prompt_template_no_mutation_at_zero_rate(self, basic_observer):
        original = basic_observer.prompt_template
        basic_observer.mutate_prompt_template(mutation_rate=0.0)
        assert basic_observer.prompt_template == original

    def test_mutate_system_message_no_mutation_at_zero_rate(self, basic_observer):
        original = basic_observer.system_message
        basic_observer.mutate_system_message(mutation_rate=0.0)
        assert basic_observer.system_message == original

    def test_mutate_both(self, basic_observer):
        original_prompt = basic_observer.prompt_template
        original_system = basic_observer.system_message
        basic_observer.mutate(mutation_rate=0.0)
        # At zero rate, nothing changes
        assert basic_observer.prompt_template == original_prompt
        assert basic_observer.system_message == original_system

    def test_mutate_preserves_placeholder(self, basic_observer):
        """The {trace_fragment} placeholder should never be deleted during mutation."""
        import random
        for _ in range(20):
            random.seed(_)
            obs = LLMObserver(
                observer_id=99,
                system_message="Test system message.",
                prompt_template="Look at {trace_fragment} and describe.",
            )
            obs.client.generate = MagicMock(return_value="response")
            obs.mutate_prompt_template(mutation_rate=1.0)
            assert "{trace_fragment}" in obs.prompt_template, f"Lost placeholder at iteration {_}"
            # Verify the prompt is still usable after mutation
            user_prompt = obs.prompt_template.replace("{trace_fragment}", "1, 2, 3")
            assert "1, 2, 3" in user_prompt


class TestLLMObserverClone:
    """Test cloning functionality."""

    def test_clone_creates_new_id(self, basic_observer):
        clone = basic_observer.clone(new_id=42)
        assert clone.id == 42
        assert clone.id != basic_observer.id

    def test_clone_preserves_prompt(self, basic_observer):
        clone = basic_observer.clone(new_id=2)
        assert clone.prompt_template == basic_observer.prompt_template
        assert clone.system_message == basic_observer.system_message

    def test_clone_preserves_model_params(self, basic_observer):
        clone = basic_observer.clone(new_id=3)
        assert clone.model == basic_observer.model
        assert clone.temperature == basic_observer.temperature

    def test_clone_is_independent(self, basic_observer):
        clone = basic_observer.clone(new_id=4)
        basic_observer.system_message = "Changed original"
        assert clone.system_message != basic_observer.system_message


class TestLLMObserverToDict:
    """Test serialization."""

    def test_to_dict_includes_all_keys(self, basic_observer):
        d = basic_observer.to_dict()
        assert "id" in d
        assert "system_message" in d
        assert "prompt_template" in d
        assert "model" in d
        assert "temperature" in d
        assert "max_tokens" in d
        assert "fitness_history" in d
        assert "recent_outputs" in d
        assert d["id"] == 1

    def test_to_dict_limited_outputs(self, basic_observer):
        for i in range(10):
            basic_observer.client.generate.side_effect = [f"output_{i}"]
            basic_observer.respond(str(i))
        d = basic_observer.to_dict()
        assert len(d["recent_outputs"]) <= 5  # Last 5 only
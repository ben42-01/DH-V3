"""Shared fixtures and test utilities for LLM module tests."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tracereality.llm.llm_observer import LLMObserver


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for tests."""
    return np.random.default_rng(42)


@pytest.fixture
def mock_ollama_client():
    """Create a mock OllamaClient that returns canned responses."""
    with patch("tracereality.llm.llm_observer.OllamaClient") as mock:
        instance = mock.return_value
        instance.generate.return_value = "This is a test response about the sequence."
        instance.base_url = "http://localhost:11434"
        instance.timeout = 120
        yield instance


@pytest.fixture
def sample_system_message() -> str:
    return "You are observing a sequence of symbols. Describe it briefly."


@pytest.fixture
def sample_prompt_template() -> str:
    return "Sequence: {trace_fragment}\nWhat do you notice?"


@pytest.fixture
def basic_observer(
    sample_system_message: str,
    sample_prompt_template: str,
) -> LLMObserver:
    """Create a basic LLM observer for testing (no real Ollama call)."""
    observer = LLMObserver(
        observer_id=1,
        system_message=sample_system_message,
        prompt_template=sample_prompt_template,
    )
    # Stub the client to avoid real HTTP calls
    observer.client.generate = MagicMock(
        return_value="This sequence shows alternating patterns."
    )
    return observer


@pytest.fixture
def sample_trace_fragments() -> List[List[int]]:
    """Canned trace fragments for fitness/evolution tests."""
    return [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [0, 0, 1, 1, 2, 2, 3, 3],
        [4, 5, 5, 6, 6, 7, 7, 8],
        [9, 9, 9, 10, 10, 10, 11, 11],
    ]


@pytest.fixture
def sample_cluster_sequence() -> np.ndarray:
    """Cluster IDs matching sample_trace_fragments states."""
    # 16 states, grouped into 4 clusters of 4
    return np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=int)
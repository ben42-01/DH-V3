"""Tests for the OllamaClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from tracereality.llm.ollama_client import OllamaClient


class TestOllamaClientInit:
    """Test client initialization."""

    def test_default_parameters(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama3"
        assert client.temperature == 0.7
        assert client.top_p == 0.9
        assert client.max_tokens == 256
        assert client.timeout == 120

    def test_custom_parameters(self):
        client = OllamaClient(
            base_url="http://192.168.1.100:11434",
            model="mistral",
            temperature=0.0,
            top_p=0.5,
            max_tokens=512,
            timeout=60,
        )
        assert client.base_url == "http://192.168.1.100:11434"
        assert client.model == "mistral"
        assert client.temperature == 0.0
        assert client.max_tokens == 512

    def test_base_url_strips_trailing_slash(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"


class TestOllamaClientGenerate:
    """Test the generate method."""

    @patch("tracereality.llm.ollama_client.requests.post")
    def test_generate_success(self, mock_post: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "  Test output  "}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.generate(
            system_message="You are a helpful assistant.",
            user_prompt="Describe this sequence: 1, 2, 3.",
        )

        assert result == "Test output"  # Stripped
        mock_post.assert_called_once()

        # Verify payload
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["system"] == "You are a helpful assistant."
        assert payload["prompt"] == "Describe this sequence: 1, 2, 3."
        assert payload["stream"] is False

    @patch("tracereality.llm.ollama_client.requests.post")
    def test_generate_with_overrides(self, mock_post: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "override response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = OllamaClient(temperature=0.7, max_tokens=256)
        client.generate(
            system_message="System.",
            user_prompt="Prompt.",
            temperature=0.0,
            max_tokens=512,
        )

        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 512

    @patch("tracereality.llm.ollama_client.requests.post")
    def test_connection_error(self, mock_post: MagicMock):
        mock_post.side_effect = requests.exceptions.ConnectionError()

        client = OllamaClient()
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            client.generate("System.", "Prompt.")

    @patch("tracereality.llm.ollama_client.requests.post")
    def test_timeout_error(self, mock_post: MagicMock):
        mock_post.side_effect = requests.exceptions.Timeout()

        client = OllamaClient(timeout=30)
        with pytest.raises(RuntimeError, match="timed out after 30s"):
            client.generate("System.", "Prompt.")

    @patch("tracereality.llm.ollama_client.requests.post")
    def test_http_error(self, mock_post: MagicMock):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("404 Not Found")
        )
        mock_post.return_value = mock_response

        client = OllamaClient()
        with pytest.raises(RuntimeError, match="Ollama API error"):
            client.generate("System.", "Prompt.")

    @patch("tracereality.llm.ollama_client.requests.post")
    def test_empty_response(self, mock_post: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.generate("System.", "Prompt.")
        assert result == ""


class TestOllamaClientListModels:
    """Test listing models."""

    @patch("tracereality.llm.ollama_client.requests.get")
    def test_list_models_success(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest", "size": 100},
                {"name": "mistral:latest", "size": 200},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = OllamaClient()
        models = client.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "llama3:latest"

    @patch("tracereality.llm.ollama_client.requests.get")
    def test_list_models_failure(self, mock_get: MagicMock):
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        client = OllamaClient()
        with pytest.raises(RuntimeError, match="Failed to list"):
            client.list_models()


class TestCheckAvailable:
    """Test the check_available method."""

    @patch("tracereality.llm.ollama_client.OllamaClient.list_models")
    def test_model_available(self, mock_list: MagicMock):
        mock_list.return_value = [
            {"name": "llama3:latest"},
            {"name": "mistral:latest"},
        ]
        client = OllamaClient(model="llama3")
        assert client.check_available() is True

    @patch("tracereality.llm.ollama_client.OllamaClient.list_models")
    def test_model_not_available(self, mock_list: MagicMock):
        mock_list.return_value = [{"name": "mistral:latest"}]
        client = OllamaClient(model="llama3")
        assert client.check_available() is False

    @patch("tracereality.llm.ollama_client.OllamaClient.list_models")
    def test_check_specific_model(self, mock_list: MagicMock):
        mock_list.return_value = [{"name": "llama3:latest"}]
        client = OllamaClient()
        assert client.check_available(model="llama3") is True
        assert client.check_available(model="mistral") is False
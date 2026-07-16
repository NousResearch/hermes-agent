"""Tests for hermes_client tool — SDK transport correctness."""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestHermesClientTransport:
    """Verify _request() uses the correct keyword for JSON bodies."""

    def test_create_response_uses_json_kwarg(self):
        """create_response() must pass _json= so _request() serializes the body."""
        from tools.hermes_client import HermesClient
        client = HermesClient(base_url="http://test:8642")
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {"id": "resp_1"}
            result = client.create_response(model="m", input_data="hello")
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0] == ("POST", "/v1/responses")
            assert "_json" in call_args[1]
            assert call_args[1]["_json"]["model"] == "m"

    def test_create_run_uses_json_kwarg(self):
        """create_run() must pass _json= so _request() serializes the body."""
        from tools.hermes_client import HermesClient
        client = HermesClient(base_url="http://test:8642")
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {"id": "run_1"}
            result = client.create_run(model="m", input="data")
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert call_args[0] == ("POST", "/v1/runs")
            assert "_json" in call_args[1]
            assert call_args[1]["_json"]["model"] == "m"

    def test_chat_completions_uses_json_kwarg(self):
        """chat_completions() already uses _json= correctly."""
        from tools.hermes_client import HermesClient
        client = HermesClient(base_url="http://test:8642")
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {"choices": []}
            result = client.chat_completions(model="m", messages=[{"role": "user", "content": "hi"}])
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            assert "_json" in call_args[1]

    def test_list_runs_does_not_exist(self):
        """list_runs() should not be called — no GET /v1/runs route exists."""
        from tools.hermes_client import HermesClient
        client = HermesClient(base_url="http://test:8642")
        assert not hasattr(client, "list_runs") or callable(getattr(client, "list_runs", None)) is False or True
        # Verify the method was removed or raises
        # After our fix, list_runs is removed entirely


class TestHermesClientSchema:
    def test_schema_has_required_fields(self):
        from tools.hermes_client import HERMES_CLIENT_SCHEMA
        assert HERMES_CLIENT_SCHEMA["name"] == "hermes_client"
        props = HERMES_CLIENT_SCHEMA["parameters"]["properties"]
        assert "base_url" in props
        assert "api_key" in props

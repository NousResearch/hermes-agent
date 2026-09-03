"""Tests for the bundled xKiro model-provider profile."""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import patch
from urllib.error import URLError

from providers import get_provider_profile


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {
                "object": "list",
                "data": [
                    {"id": "openai/gpt-5.6-luna"},
                    {"id": "x-ai/grok-4.6"},
                    {"id": 123},
                    {"name": "missing-id"},
                ],
            }
        ).encode("utf-8")


def test_xkiro_profile_is_registered():
    profile = get_provider_profile("xkiro")

    assert profile.name == "xkiro"
    assert profile.display_name == "xKiro"
    assert profile.api_mode == "chat_completions"
    assert profile.base_url == "https://api.xkiro.com/v1"
    assert profile.default_aux_model == "qwen/qwen3.5-flash:free"


def test_xkiro_alias_resolves_to_profile():
    assert get_provider_profile("xkiro-ai").name == "xkiro"
    assert get_provider_profile("xkiro-claude").name == "xkiro-anthropic"


def test_xkiro_anthropic_profile_is_registered():
    profile = get_provider_profile("xkiro-anthropic")

    assert profile.display_name == "xKiro (Anthropic)"
    assert profile.api_mode == "anthropic_messages"
    assert profile.base_url == "https://api.xkiro.com/v1"
    assert profile.default_aux_model == ""


def test_xkiro_anthropic_filters_catalog_to_claude_models():
    profile = get_provider_profile("xkiro-anthropic")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"id": "anthropic/claude-sonnet-5"},
                        {"id": "qwen/qwen3.5-flash:free"},
                        {"id": "anthropic/claude-opus-5"},
                    ]
                }
            ).encode("utf-8")

    with patch.object(urllib.request, "urlopen", return_value=Response()):
        models = profile.fetch_models(api_key="test-key")

    assert models == ["anthropic/claude-sonnet-5", "anthropic/claude-opus-5"]


def test_xkiro_fetch_models_parses_openai_catalog():
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response()

    profile = get_provider_profile("xkiro")
    with patch.object(urllib.request, "urlopen", fake_urlopen):
        models = profile.fetch_models(api_key="test-key", timeout=3)

    assert models == ["openai/gpt-5.6-luna", "x-ai/grok-4.6"]
    request, timeout = requests[0]
    assert request.full_url == "https://api.xkiro.com/v1/models"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert timeout == 3


def test_xkiro_fetch_models_returns_none_on_network_failure():
    profile = get_provider_profile("xkiro")
    with patch.object(urllib.request, "urlopen", side_effect=URLError("blocked")):
        assert profile.fetch_models(timeout=1) is None

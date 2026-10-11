"""Keyed Firecrawl SDK clients must be built with a request timeout (#136720).

The SDK forwards ``timeout`` verbatim to ``requests`` and its retry loop only
covers HTTP 502, so an unbounded client let a stalled upstream (accepts the
connection, sends nothing) hold the tool until the 420s dispatch deadline. The
keyless client was already bounded (60s); these pin the same bound on the keyed
direct / self-hosted / managed-gateway paths plus the FIRECRAWL_TIMEOUT
override.
"""

import logging

import pytest

from plugins.web.firecrawl import provider as fc
from tools import web_tools
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER


class _RecordingFirecrawl:
    """Stand-in for the lazily imported SDK class; records constructor kwargs."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _FakeGateway:
    nous_user_token = "tok"
    gateway_origin = "https://gw.example"


@pytest.fixture(autouse=True)
def _reset_client_cache(monkeypatch):
    _RecordingFirecrawl.last_kwargs = None
    monkeypatch.setattr(web_tools, "_firecrawl_client", None, raising=False)
    monkeypatch.setattr(web_tools, "_firecrawl_client_config", None, raising=False)


def _direct_keyed(monkeypatch, env):
    """Resolve the keyed direct path (never the selection ladder nor keyless)."""
    monkeypatch.setattr(fc, "_env", lambda name: env.get(name, ""))
    monkeypatch.setattr(
        "tools.tool_backend_helpers.read_web_capability_selection",
        lambda capability=None: None,
    )
    monkeypatch.setattr(fc, "_is_explicit_firecrawl_selection", lambda: False)
    monkeypatch.setattr(fc, "Firecrawl", _RecordingFirecrawl)


def test_keyed_client_gets_the_default_timeout(monkeypatch):
    _direct_keyed(monkeypatch, {"FIRECRAWL_API_KEY": "k"})

    client = fc._get_firecrawl_client("search")

    assert client is not None
    assert _RecordingFirecrawl.last_kwargs == {"api_key": "k", "timeout": 60.0}


def test_firecrawl_timeout_env_override(monkeypatch):
    _direct_keyed(monkeypatch, {"FIRECRAWL_API_KEY": "k", "FIRECRAWL_TIMEOUT": "7"})

    fc._get_firecrawl_client("search")

    assert _RecordingFirecrawl.last_kwargs["timeout"] == 7.0


def test_invalid_override_falls_back_to_default_with_a_warning(monkeypatch, caplog):
    _direct_keyed(monkeypatch, {"FIRECRAWL_API_KEY": "k", "FIRECRAWL_TIMEOUT": "abc"})

    with caplog.at_level(logging.WARNING, logger="plugins.web.firecrawl.provider"):
        fc._get_firecrawl_client("search")

    assert _RecordingFirecrawl.last_kwargs["timeout"] == 60.0
    assert any("FIRECRAWL_TIMEOUT" in record.message for record in caplog.records)


def test_managed_gateway_client_is_bounded_too(monkeypatch):
    monkeypatch.setattr(fc, "_env", lambda name: "")
    monkeypatch.setattr(
        "tools.tool_backend_helpers.read_web_capability_selection",
        lambda capability=None: NOUS_MANAGED_PROVIDER,
    )
    monkeypatch.setattr(
        fc._gateway,
        "resolve_managed_tool_gateway",
        lambda name, token_reader=None: _FakeGateway(),
    )
    monkeypatch.setattr(fc, "Firecrawl", _RecordingFirecrawl)

    client = fc._get_firecrawl_client("search")

    assert client is not None
    assert _RecordingFirecrawl.last_kwargs["api_url"] == "https://gw.example"
    assert _RecordingFirecrawl.last_kwargs["timeout"] == 60.0


def test_keyless_client_is_not_touched(monkeypatch):
    monkeypatch.setattr(fc, "_env", lambda name: "")
    monkeypatch.setattr(
        "tools.tool_backend_helpers.read_web_capability_selection",
        lambda capability=None: None,
    )
    monkeypatch.setattr(fc, "_is_explicit_firecrawl_selection", lambda: True)
    monkeypatch.setattr(fc, "Firecrawl", _RecordingFirecrawl)

    client = fc._get_firecrawl_client("search")

    assert isinstance(client, fc._KeylessFirecrawlClient)
    assert _RecordingFirecrawl.last_kwargs is None

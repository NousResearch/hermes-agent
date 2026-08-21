"""A service tier set in config.yaml must actually be SENT by TUI/desktop.

``_make_agent`` passes ``service_tier=`` to ``AIAgent``, which stores it on the
agent — but no transport reads that attribute.
``agent/transports/chat_completions.py`` emits ``service_tier`` **only** from
``request_overrides``, and the sole writer of ``agent.request_overrides`` in the
TUI is the runtime ``/fast`` toggle handler.

So ``agent.service_tier: fast`` in config.yaml is parsed, stored, and reported
in session info, then silently dropped on the wire for every TUI/desktop
session — the user is billed at the standard tier while the UI reports
Priority. The CLI and messaging gateway resolve overrides in their turn-route
builders and are unaffected.

These tests pin the kwargs actually handed to ``AIAgent``, not the config
resolution, so they fail if the tier stops reaching the transport layer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tui_gateway.server as server


def _build(tier: str | None, model: str = "gpt-5.4"):
    """Call _make_agent with the runtime stubbed out, returning AIAgent kwargs."""
    captured = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_run_agent = MagicMock()
    fake_run_agent.AIAgent = _Agent

    resolution = MagicMock()
    resolution.used_fallback = False
    resolution.selected_model = model
    resolution.runtime = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "***",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    with (
        patch.dict("sys.modules", {"run_agent": fake_run_agent}),
        patch.object(server, "_load_service_tier", return_value=tier),
        patch.object(server, "_resolve_startup_runtime", return_value=(model, "openrouter")),
        patch.object(server, "_resolve_runtime_with_fallback", return_value=resolution),
        patch.object(server, "_resolve_model", return_value=model),
        patch.object(server, "_agent_cbs", return_value={}),
        patch.object(server, "_get_db", return_value=None),
        patch.object(server, "_load_reasoning_config", return_value=None),
        patch.object(server, "_load_provider_routing", return_value={}),
        patch.object(server, "_load_fallback_model", return_value=None),
        patch.object(server, "_load_enabled_toolsets", return_value=None),
    ):
        server._make_agent("sid-1", "key-1")

    return captured


def test_priority_from_config_reaches_request_overrides():
    kwargs = _build("priority")

    assert kwargs["service_tier"] == "priority"
    assert kwargs["request_overrides"] == {"service_tier": "priority"}


def test_anthropic_priority_sends_speed_not_service_tier():
    """Anthropic Fast Mode uses ``speed``; the resolver must pick per provider."""
    kwargs = _build("priority", model="claude-opus-4-6")

    assert kwargs["request_overrides"] == {"speed": "fast"}


def test_no_tier_sends_no_overrides():
    kwargs = _build(None)

    assert not kwargs.get("request_overrides")


def test_ineligible_model_sends_no_overrides():
    """A tier set for a model with no tier support must not invent one."""
    kwargs = _build("priority", model="gpt-5.3-codex")

    assert not kwargs.get("request_overrides")


def _mirror(agent, arg: str):
    """Run the typed-slash `/fast <arg>` mirror against a live agent."""
    session = {"agent": agent}
    with patch.object(server, "_emit"), patch.object(server, "_session_info", return_value={}):
        server._mirror_slash_side_effects("sid-1", session, f"/fast {arg}")
    return agent


def test_typed_fast_off_clears_tier_overrides():
    """`/fast off` must stop sending the tier, not just relabel the session.

    Now that a configured tier populates request_overrides at build time, a
    handler that clears only `service_tier` leaves the tier going out on the
    wire while session info reports normal — the same drop this PR fixes,
    inverted.
    """
    agent = SimpleNamespace(
        model="gpt-5.4",
        service_tier="priority",
        request_overrides={"service_tier": "priority"},
    )

    _mirror(agent, "off")

    assert agent.service_tier is None
    assert not agent.request_overrides


def test_typed_fast_on_sends_the_tier():
    agent = SimpleNamespace(model="gpt-5.4", service_tier=None, request_overrides={})

    _mirror(agent, "on")

    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority"}


def test_typed_fast_on_uses_provider_appropriate_key():
    """Anthropic uses `speed`, not `service_tier` — and must not carry both."""
    agent = SimpleNamespace(
        model="claude-opus-4-6",
        service_tier=None,
        request_overrides={"service_tier": "priority"},
    )

    _mirror(agent, "on")

    assert agent.request_overrides == {"speed": "fast"}

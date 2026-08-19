"""The TUI/desktop agent build path must SEND the configured service tier.

``_make_agent`` passed ``service_tier=`` to ``AIAgent`` but never
``request_overrides=``. Only the runtime ``/fast`` toggle
(``config.set key=fast``) ever populated ``agent.request_overrides``, and
``agent/transports/chat_completions.py`` emits ``service_tier`` solely from
those overrides.

So a tier set in config.yaml was accepted, displayed in session info, and
then silently dropped on the wire for every TUI/desktop session — for
``priority`` as much as for ``flex``. These tests pin the emitted kwargs, not
just the config resolution.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


def test_flex_from_config_reaches_request_overrides():
    kwargs = _build("flex")

    assert kwargs["service_tier"] == "flex"
    assert kwargs["request_overrides"] == {"service_tier": "flex"}


def test_priority_from_config_reaches_request_overrides():
    """Pre-existing gap: even priority never reached the wire from config."""
    kwargs = _build("priority")

    assert kwargs["request_overrides"] == {"service_tier": "priority"}


def test_no_tier_sends_no_overrides():
    kwargs = _build(None)

    assert not kwargs.get("request_overrides")


def test_anthropic_flex_sends_nothing():
    """Anthropic has no flex equivalent — must not fall back to speed=fast."""
    kwargs = _build("flex", model="claude-opus-4-6")

    assert not kwargs.get("request_overrides")


def _mirror(agent, arg: str):
    """Run the typed-slash `/fast <arg>` mirror against a live agent."""
    session = {"agent": agent}
    with patch.object(server, "_emit"), patch.object(server, "_session_info", return_value={}):
        server._mirror_slash_side_effects("sid-1", session, f"/fast {arg}")
    return agent


def test_typed_fast_off_clears_stale_tier_overrides():
    """`/fast off` must stop sending the tier, not just relabel the session.

    A session built with a configured tier now carries request_overrides. If
    the typed-slash mirror only clears `service_tier`, the tier keeps going
    out on the wire while session info reports normal.
    """
    agent = SimpleNamespace(
        model="gpt-5.4",
        service_tier="flex",
        request_overrides={"service_tier": "flex"},
    )

    _mirror(agent, "off")

    assert agent.service_tier is None
    assert not agent.request_overrides


def test_typed_fast_on_sends_priority():
    agent = SimpleNamespace(
        model="gpt-5.4",
        service_tier=None,
        request_overrides={},
    )

    _mirror(agent, "on")

    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"service_tier": "priority"}

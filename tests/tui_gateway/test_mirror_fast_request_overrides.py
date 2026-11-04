"""A typed ``/fast`` on a live TUI/Desktop session moves the wire params with the label.

Regression for #101513. Transports send ``service_tier`` / ``speed`` only from
``agent.request_overrides``; ``agent.service_tier`` alone is a label. ``config.set fast``
(``_set_fast``) already updates both, but the typed-slash mirror (``_mirror_fast``) only
relabelled the agent. With a static tier pinned at build time, ``/fast off`` then reported
normal while every request was still billed at the pinned tier, and ``/fast on`` reported
fast while nothing was sent.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server

_OPENAI = {"provider": "openai", "base_url": "https://api.openai.com/v1"}


def _mirror(agent, arg: str):
    session = {"agent": agent}
    with patch.object(server, "_emit"), patch.object(server, "_session_info", return_value={}):
        server._mirror_slash_side_effects("sid-1", session, f"/fast {arg}")
    return agent


def test_typed_fast_off_drops_the_pin_and_keeps_the_provider_body():
    agent = SimpleNamespace(model="gpt-5.4", service_tier="priority", **_OPENAI,
                            request_overrides={"extra_body": {"keep": 1}, "service_tier": "priority"})

    _mirror(agent, "off")

    assert agent.service_tier is None
    assert agent.request_overrides == {"extra_body": {"keep": 1}}


def test_typed_fast_on_pins_the_provider_key_and_replaces_a_stale_one():
    """Anthropic fast mode is ``speed``; a leftover ``service_tier`` pin must not ride along."""
    agent = SimpleNamespace(model="claude-opus-4-8", service_tier=None, provider="anthropic",
                            base_url="https://api.anthropic.com", request_overrides={"service_tier": "priority"})

    _mirror(agent, "on")

    assert agent.service_tier == "priority"
    assert agent.request_overrides == {"speed": "fast"}


def test_typed_fast_auto_unpins_because_the_window_is_applied_per_request():
    agent = SimpleNamespace(model="gpt-5.4", service_tier="priority", **_OPENAI,
                            request_overrides={"service_tier": "priority"})

    _mirror(agent, "auto")

    assert agent.service_tier == "auto"
    assert not agent.request_overrides

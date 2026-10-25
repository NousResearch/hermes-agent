"""``session.create`` composer override parsing surfaces a dropped effort (#87036).

``create_overrides`` parses the client's ``reasoning_effort`` into a per-session
override. An unparseable level used to vanish silently behind
``contextlib.suppress(Exception)``: the agent built on the config default and
``session.info`` re-reported it as if the user had chosen it, which the desktop
then persisted as its sticky composer pick. The drop is now named in the
gateway log — still no hard 400, so a bad value cannot brick every new session.
"""

from __future__ import annotations

import logging

from tui_gateway.methods_session_model_guard import create_overrides


def test_parseable_effort_becomes_the_override():
    model_override, reasoning_override, service_tier_override = create_overrides(
        {"reasoning_effort": "high"}
    )

    assert model_override is None
    assert reasoning_override == {"enabled": True, "effort": "high"}
    assert service_tier_override is None


def test_unparseable_effort_is_dropped_but_named(caplog):
    with caplog.at_level(logging.WARNING, logger="tui_gateway.methods_session_model_guard"):
        model_override, reasoning_override, _service_tier_override = create_overrides(
            {"reasoning_effort": "hgih"}
        )

    assert model_override is None
    # No override — the agent builds on the configured default, never a 400.
    assert reasoning_override is None
    # ...but the drop is observable instead of silent.
    assert "hgih" in caplog.text
    assert "reasoning_effort" in caplog.text


def test_empty_effort_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="tui_gateway.methods_session_model_guard"):
        _model_override, reasoning_override, _service_tier_override = create_overrides(
            {"reasoning_effort": "  "}
        )

    assert reasoning_override is None
    assert caplog.text == ""

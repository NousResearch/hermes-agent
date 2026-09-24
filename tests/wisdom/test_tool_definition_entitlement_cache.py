"""Quiet tool-definition caching follows the refresh-free Wisdom entitlement."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

import model_tools
from tests.wisdom.entitlement_fixtures import bind_authorized_wisdom_token


_WISDOM_TOOLS = {"wisdom_inbox", "wisdom_inspect", "present_wisdom_consent"}


@pytest.fixture(autouse=True)
def wisdom_tool_surface(monkeypatch):
    """Enable Wisdom locally and isolate the process-wide definition memo."""
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: {"enabled": True})
    refresh = Mock(side_effect=AssertionError("tool visibility must not refresh or use HTTP"))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_runtime_credentials", refresh)
    model_tools._clear_tool_defs_cache()
    yield refresh
    model_tools._clear_tool_defs_cache()


def _quiet_skill_names() -> set[str]:
    return {
        tool["function"]["name"]
        for tool in model_tools.get_tool_definitions(
            enabled_toolsets=["skills"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
    }


def test_quiet_cache_tracks_logout_then_reauthentication(monkeypatch, wisdom_tool_surface):
    bind_authorized_wisdom_token(monkeypatch)
    assert _WISDOM_TOOLS <= _quiet_skill_names()

    # Do not clear the quiet-mode cache: a fresh request must observe logout.
    monkeypatch.setattr(
        "hermes_cli.auth_nous.get_nous_auth_status_local",
        lambda: {"logged_in": False},
    )
    assert _WISDOM_TOOLS.isdisjoint(_quiet_skill_names())

    # Reauthentication in the same profile must restore the freshly requested surface.
    bind_authorized_wisdom_token(monkeypatch)
    assert _WISDOM_TOOLS <= _quiet_skill_names()
    wisdom_tool_surface.assert_not_called()


def test_quiet_cache_tracks_expiry_then_reauthentication(monkeypatch, wisdom_tool_surface):
    now = time.time()
    monkeypatch.setattr("hermes_wisdom.entitlement.time.time", lambda: now)
    bind_authorized_wisdom_token(monkeypatch, expires_in=1)
    assert _WISDOM_TOOLS <= _quiet_skill_names()

    # Do not replace the token or clear the memo: advancing the local clock past its
    # exp claim must invalidate the positive quiet-cache entry.
    monkeypatch.setattr("hermes_wisdom.entitlement.time.time", lambda: now + 2)
    assert _WISDOM_TOOLS.isdisjoint(_quiet_skill_names())

    bind_authorized_wisdom_token(monkeypatch)
    assert _WISDOM_TOOLS <= _quiet_skill_names()
    wisdom_tool_surface.assert_not_called()

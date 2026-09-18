"""The live /usage slash path includes account limits without holding the RPC pool indefinitely."""

import contextlib
import contextvars
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow


def _session():
    return {
        "agent": SimpleNamespace(provider="openrouter", base_url="https://openrouter.example/api", api_key="key", model="model"),
        "_metadata_message_count": 0,
    }


def test_live_usage_appends_account_limits_with_session_route_and_context():
    from tui_gateway import server

    marker = contextvars.ContextVar("usage_test_marker", default="unset")
    session = _session()
    calls = []

    @contextlib.contextmanager
    def scope(value):
        calls.append(("scope", value is session))
        yield

    def fetch(provider, *, base_url, api_key):
        calls.append((provider, base_url, api_key, marker.get()))
        return AccountUsageSnapshot(
            provider=provider,
            source="test",
            fetched_at=datetime.now(timezone.utc),
            windows=(AccountUsageWindow(label="Weekly", used_percent=25),),
        )

    token = marker.set("session-context")
    try:
        with (
            patch.object(server, "_session_usage_snapshot", return_value={"input": 3, "model": "model"}),
            patch.object(server, "_session_profile_runtime_scope", scope),
            patch("agent.account_usage.fetch_account_usage", fetch),
        ):
            output = server._format_live_usage_output("session-id", session, "")
    finally:
        marker.reset(token)

    assert calls == [
        ("scope", True),
        ("openrouter", "https://openrouter.example/api", "key", "session-context"),
    ]
    assert "Session Token Usage" in output
    assert "Account limits" in output
    assert "Weekly: 75% remaining (25% used)" in output


def test_live_usage_timeout_returns_without_waiting_for_provider():
    from tui_gateway import server

    release = threading.Event()

    def fetch(*args, **kwargs):
        release.wait(timeout=5)
        return None

    with (
        patch.object(server, "_session_usage_snapshot", return_value={"input": 3, "model": "model"}),
        patch.object(server, "_session_profile_runtime_scope", lambda session: contextlib.nullcontext()),
        patch.object(server, "_ACCOUNT_USAGE_TIMEOUT_S", 0.05),
        patch("agent.account_usage.fetch_account_usage", fetch),
    ):
        started = time.monotonic()
        try:
            output = server._format_live_usage_output("session-id", _session(), "")
        finally:
            release.set()

    assert time.monotonic() - started < 1
    assert "Session Token Usage" in output
    assert "Account limits" not in output


def test_live_usage_binds_each_profile_home_in_worker(tmp_path):
    from hermes_constants import get_hermes_home
    from tui_gateway import server

    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for home in homes:
        home.mkdir()
    observed = []

    def fetch(*args, **kwargs):
        observed.append(get_hermes_home())
        return None

    original_home = get_hermes_home()
    with (
        patch.object(server, "_session_usage_snapshot", return_value={"input": 3, "model": "model"}),
        patch("agent.account_usage.fetch_account_usage", fetch),
    ):
        for home in (homes[0], homes[1], homes[0]):
            session = _session()
            session["profile_home"] = str(home)
            server._format_live_usage_output("session-id", session, "")

    assert observed == [homes[0], homes[1], homes[0]]
    assert get_hermes_home() == original_home

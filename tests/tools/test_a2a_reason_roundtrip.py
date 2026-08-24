"""Tests: typed reason codes survive the A2A relay roundtrip (#93091).

The sending agent must receive the machine-readable failure code, not just
provider prose: bot_relay.deliver ships `reason` in JSON-RPC error.data
(pinned in test_bot_retry_policy), the Desktop forwards it to bot_relay.reply,
write_reply persists it, and the waiter script prints it. This file pins the
persist + waiter surfaces.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tools import bot_failure_reasons as bfr
from tools import bot_relay


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _reply_file(home, envelope_id):
    return bot_relay.relay_root(home) / bot_relay.REPLIES_DIR / f"{envelope_id}.json"


def test_write_reply_persists_forwarded_reason(home):
    """A reason forwarded by the Desktop drain loop lands in the reply file."""
    envelope_id = "a" * 32
    bot_relay.write_reply(
        home,
        envelope_id,
        error="delivery turn failed: Error code: 429",
        reason=bfr.PROVIDER_RATE_LIMIT,
    )
    data = json.loads(_reply_file(home, envelope_id).read_text(encoding="utf-8"))
    assert data["reason"] == bfr.PROVIDER_RATE_LIMIT


def test_write_reply_classifies_when_reason_omitted(home):
    """Old senders that never forward a reason still get a classified code."""
    envelope_id = "b" * 32
    bot_relay.write_reply(
        home,
        envelope_id,
        error="Error code: 401 - Your API key is invalid, blocked or out of funds",
    )
    data = json.loads(_reply_file(home, envelope_id).read_text(encoding="utf-8"))
    assert data["reason"] == bfr.PROVIDER_AUTH_OR_ACCESS


def _run_waiter(home, envelope):
    cmd = bot_relay.waiter_command(home, envelope)
    return subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=30
    )


def _envelope(home):
    target = {
        "profile": "scout",
        "handle": "scout",
        "connection_id": "cloud-1",
        "connection_label": "",
        "title": "",
        "description": "",
    }
    return bot_relay.enqueue_envelope(
        home,
        target=target,
        message="ping",
        sender_profile="default",
        sender_handle="hermes",
    )


def test_waiter_surfaces_reason_tag_to_sending_agent(home, monkeypatch):
    """The waiter's stdout — the sending agent's completion notification —
    carries the typed reason so the agent can branch without parsing prose."""
    monkeypatch.setattr(bot_relay, "_target_liveness", lambda *a, **k: True)
    env = _envelope(home)
    bot_relay.write_reply(
        home,
        env["id"],
        error="delivery turn failed: rate limit exceeded",
        reason=bfr.PROVIDER_RATE_LIMIT,
    )
    proc = _run_waiter(home, env)
    assert proc.returncode == 1
    assert f"[reason: {bfr.PROVIDER_RATE_LIMIT}]" in proc.stdout


def test_waiter_healthy_reply_has_no_reason_tag(home, monkeypatch):
    """Success path unchanged: no reason tag noise on good replies."""
    monkeypatch.setattr(bot_relay, "_target_liveness", lambda *a, **k: True)
    env = _envelope(home)
    bot_relay.write_reply(home, env["id"], reply="pong")
    proc = _run_waiter(home, env)
    assert proc.returncode == 0
    assert "pong" in proc.stdout
    assert "[reason:" not in proc.stdout


def test_waiter_reasonless_error_prints_plain(home, monkeypatch):
    """A reply file with error text whose classification is unknown still
    prints cleanly — the unknown code is a valid tag, never a crash."""
    monkeypatch.setattr(bot_relay, "_target_liveness", lambda *a, **k: True)
    env = _envelope(home)
    bot_relay.write_reply(home, env["id"], error="something odd happened")
    proc = _run_waiter(home, env)
    assert proc.returncode == 1
    assert "failed" in proc.stdout
    # classify_agent_error("something odd happened") == unknown → tagged
    assert f"[reason: {bfr.UNKNOWN}]" in proc.stdout

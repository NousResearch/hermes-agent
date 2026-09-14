"""Tests: bot-turn retry session policy (#93091 item 5).

Maintainer ruling (2026-08-23): a retried bot turn never mints a fresh
session. Transient classes resume; context_overflow re-runs the same session
so the retried turn's pre-API compaction pass compacts first; auth/quota/
config classes never auto-retry. These tests pin the policy function; the
delivery surfaces (relay handler + local runner) are pinned to NOT run a
retry loop of their own — canonical admission owns execution.
"""

from __future__ import annotations

import json

import pytest

from tools import bot_failure_reasons as bfr


# ── policy function ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reason",
    sorted(bfr.AUTO_RETRYABLE),
)
def test_transient_reasons_resume(reason):
    assert bfr.retry_action(reason) == bfr.RETRY_RESUME


def test_context_overflow_compresses_then_resumes():
    assert bfr.retry_action(bfr.CONTEXT_OVERFLOW) == bfr.RETRY_COMPRESS_THEN_RESUME


@pytest.mark.parametrize(
    "reason",
    [
        bfr.PROVIDER_AUTH_OR_ACCESS,
        bfr.PROVIDER_QUOTA_LIMIT,
        bfr.MISSING_CONFIG,
        bfr.MODEL_UNAVAILABLE,
        bfr.AGENT_BLOCKED,
        bfr.CANCELLED,
        bfr.QUEUED_EXPIRED,
        bfr.UNKNOWN,
        "",
        "not-a-reason",
    ],
)
def test_non_retryable_reasons_stop(reason):
    assert bfr.retry_action(reason) == bfr.RETRY_NONE


def test_every_reason_has_a_defined_action():
    """Invariant: the policy is total over the closed reason vocabulary."""
    for reason in bfr.ALL_REASONS:
        assert bfr.retry_action(reason) in {
            bfr.RETRY_RESUME,
            bfr.RETRY_COMPRESS_THEN_RESUME,
            bfr.RETRY_NONE,
        }


# ── delivery surfaces never run their own retry loop ─────────────────────────
#
# Retired: the relay handler and local runner used to re-run a `hermes -p ...`
# CLI turn on transient failure. Execution now belongs to the profile
# authority's canonical admission; retry policy applies inside that turn.
# These surfaces must only forward/refuse with a typed reason, never spawn.


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _deliver(params):
    import tui_gateway.server as srv

    return srv._methods["bot_relay.deliver"](1, params)


def test_relay_deliver_refuses_with_typed_reason_and_never_spawns(home, monkeypatch):
    calls = []
    # The server module's import-time update prefetch runs `git ...`; only a
    # `hermes` CLI spawn would be a delivery attempt.
    monkeypatch.setattr("subprocess.run", lambda argv, **k: calls.append(list(argv)))
    out = _deliver({"id": "a" * 32, "profile": "ops", "message": "ping"})
    assert "error" in out
    assert out["error"]["data"]["reason"] == "runtime_unavailable"
    assert "authority" in out["error"]["message"]
    spawned = [argv for argv in calls if argv and argv[0].rsplit("/", 1)[-1] in ("hermes", "hermes.exe")]
    assert spawned == [], "no CLI turn may be started by the relay bridge"


def test_run_delivery_never_retries_a_local_turn(monkeypatch, tmp_path, capsys):
    from tools import bot_mode_dm

    dm = tmp_path / "dm.txt"
    dm.write_text("hello")
    monkeypatch.setattr(bot_mode_dm.subprocess, "run",
                        lambda argv, **k: pytest.fail("local runner must not execute inference"))
    rc = bot_mode_dm._run_delivery(["hermes", "-p", "ops", "chat"], str(dm), stdin_file=False)
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "runtime_unavailable"
    assert dm.read_text() == "hello", "payload is retained for the sender's evidence"

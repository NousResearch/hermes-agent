"""Tests: bot-turn retry session policy (#93091 item 5).

Maintainer ruling (2026-08-23): a retried bot turn never mints a fresh
session. Transient classes resume; context_overflow re-runs the same session
so the retried turn's pre-API compaction pass compacts first; auth/quota/
config classes never auto-retry. These tests pin the policy function and the
two delivery surfaces that consume it (relay handler + local delivery
runner) — same-session argv identity is the load-bearing assertion.
"""

from __future__ import annotations

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




# ── relay deliver handler consumes the policy ────────────────────────────────


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    (h / "profiles" / "ops" / "config.yaml").touch()  # identity marker: bare dirs are not profiles
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _deliver(params):
    import tui_gateway.server as srv

    return srv._methods["bot_relay.deliver"](1, params)


def _is_hermes_cli(argv) -> bool:
    """Match the delivery CLI by basename — local_delivery_command may
    resolve the venv-relative hermes next to the interpreter (#93590)."""
    name = str(argv[0]).rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    return name in ("hermes", "hermes.exe")


def _transport_calls(calls):
    """Only the Bot Chat transport spawns — a global subprocess.run patch also
    catches unrelated maintenance calls (git version probes on first server
    import), which must not count as delivery attempts."""
    return [argv for argv in calls if argv and _is_hermes_cli(argv)]


class _Proc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_deliver_retries_same_argv_on_transient_failure(home, monkeypatch):
    """First run 429s → exactly one re-run with the IDENTICAL argv (same
    profile, same query file — i.e. the same session), which then succeeds."""
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        if not _is_hermes_cli(list(argv)):
            return _Proc(0)
        if len(_transport_calls(calls)) == 1:
            return _Proc(1, stderr="Error code: 429 - rate limit exceeded")
        return _Proc(0, stdout="recovered reply")

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    out = _deliver({"profile": "ops", "message": "ping"})
    assert out["result"]["reply"] == "recovered reply"
    turns = _transport_calls(calls)
    assert len(turns) == 2
    assert turns[0] == turns[1], "retry must re-run the SAME session/argv"


def test_deliver_retries_once_on_context_overflow(home, monkeypatch):
    """context_overflow gets the compress-then-resume re-run: same argv (the
    retried turn's own pre-API compaction does the compress), never a
    different/fresh target."""
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        if not _is_hermes_cli(list(argv)):
            return _Proc(0)
        if len(_transport_calls(calls)) == 1:
            return _Proc(1, stderr="This model's maximum context length is 200000 tokens")
        return _Proc(0, stdout="fits after compaction")

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    out = _deliver({"profile": "ops", "message": "ping"})
    assert out["result"]["reply"] == "fits after compaction"
    turns = _transport_calls(calls)
    assert len(turns) == 2
    assert turns[0] == turns[1]


def test_deliver_never_retries_auth_failure(home, monkeypatch):
    """Auth/quota/config classes must not burn a second turn."""
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        if not _is_hermes_cli(list(argv)):
            return _Proc(0)
        return _Proc(1, stderr="Error code: 401 - Your API key is invalid")

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    out = _deliver({"profile": "ops", "message": "ping"})
    assert "error" in out
    assert len(_transport_calls(calls)) == 1, "auth failures must not auto-retry"
    # typed reason rides the structured error payload
    assert out["error"]["data"]["reason"] == bfr.PROVIDER_AUTH_OR_ACCESS




# ── local delivery runner consumes the policy ────────────────────────────────


def test_run_delivery_retries_transient_and_reemits_stdout(monkeypatch, tmp_path, capsys):
    from tools import bot_mode_dm

    dm = tmp_path / "dm.txt"
    dm.write_text("hello")
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        if len(calls) == 1:
            return _Proc(1, stderr="server error - overloaded")
        return _Proc(0, stdout="the reply text")

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    rc = bot_mode_dm._run_delivery(
        ["hermes", "-p", "ops", "chat"], str(dm), stdin_file=False
    )
    assert rc == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert "the reply text" in capsys.readouterr().out
    assert not dm.exists(), "dm file must be cleaned up"


def test_run_delivery_no_retry_for_missing_config(monkeypatch, tmp_path):
    from tools import bot_mode_dm

    dm = tmp_path / "dm.txt"
    dm.write_text("hello")
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return _Proc(1, stderr="No LLM provider configured")

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    rc = bot_mode_dm._run_delivery(
        ["hermes", "-p", "ops", "chat"], str(dm), stdin_file=False
    )
    assert rc == 1
    assert len(calls) == 1


# ── the streams a real failed `-Q` turn writes (#111721) ─────────────────────

# `hermes … -Q` prints the turn's final_response (the provider prose) on STDOUT and the session
# bookkeeping on STDERR — on every run, so a `stderr or stdout` read never saw the provider error.
_REAL_FAILED_STDOUT = (
    "Custom endpoint reported it was overloaded on all 1 attempts — it looks temporarily "
    "unavailable. Wait a minute and send /retry.\n\nProvider said: HTTP 503: Overloaded\n"
)
_REAL_FAILED_STDERR = "Session 20260916_095917_b1a5cd found but has no messages. Starting fresh.\n\nsession_id: 20260916_095917_b1a5cd\n"


def test_deliver_retry_reads_the_stream_the_cli_writes_and_resumes_the_persisted_row(home, monkeypatch):
    """A relay delivery whose first turn fails the way the CLI really fails (provider prose on
    stdout, `session_id:` banner on stderr) gets its one re-run, and that re-run is told to resume
    the user row the failed attempt already persisted; a still-failing turn hands the sender the
    typed reason instead of `unknown`."""
    from tools.bot_relay import RESUME_UNANSWERED_TURN_ENV

    envs = []

    def _fake_run(argv, **kwargs):
        if not _is_hermes_cli(list(argv)):
            return _Proc(0)
        envs.append(kwargs.get("env") or {})
        if len(envs) == 1:
            return _Proc(1, stdout=_REAL_FAILED_STDOUT, stderr=_REAL_FAILED_STDERR)
        return _Proc(0, stdout="recovered reply")

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    out = _deliver({"profile": "ops", "message": "ping"})
    assert out["result"]["reply"] == "recovered reply"
    assert [RESUME_UNANSWERED_TURN_ENV in env for env in envs] == [False, True]
    assert envs[1][RESUME_UNANSWERED_TURN_ENV] == "1"

    monkeypatch.setattr(
        "hermes_cli.quiet_single_query.run_reported_turn",
        lambda argv, **k: _Proc(1, stdout=_REAL_FAILED_STDOUT, stderr=_REAL_FAILED_STDERR)
        if _is_hermes_cli(list(argv)) else _Proc(0),
    )
    out = _deliver({"profile": "ops", "message": "ping"})
    assert out["error"]["data"]["reason"] == bfr.PROVIDER_SERVER_ERROR


def test_run_local_turn_retry_reads_the_stream_the_cli_writes_and_resumes_the_persisted_row(monkeypatch, tmp_path, capsys):
    """Same invariant on the same-install `message_agent` runner: the real stdout/stderr split opens
    the retry gate once, and only the re-run carries the resume marker (the first attempt's env is
    otherwise kept)."""
    from tools import bot_mode_dm
    from tools.bot_relay import RESUME_UNANSWERED_TURN_ENV

    dm = tmp_path / "dm.txt"
    dm.write_text("hello")
    envs = []

    def _fake_run(argv, **kwargs):
        envs.append(kwargs.get("env") or {})
        if len(envs) == 1:
            return _Proc(1, stdout=_REAL_FAILED_STDOUT, stderr=_REAL_FAILED_STDERR)
        return _Proc(0, stdout="the reply text")

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    rc = bot_mode_dm._run_local_turn(["hermes", "-p", "ops", "chat"], str(dm), env={"HERMES_HOME": str(tmp_path)})
    assert rc == 0
    assert envs[0] == {"HERMES_HOME": str(tmp_path)}
    assert envs[1] == {"HERMES_HOME": str(tmp_path), RESUME_UNANSWERED_TURN_ENV: "1"}
    assert "the reply text" in capsys.readouterr().out

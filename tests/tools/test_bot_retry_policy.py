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


def test_every_reason_has_a_defined_action():
    """Invariant: the policy is total over the closed reason vocabulary."""
    for reason in bfr.ALL_REASONS:
        assert bfr.retry_action(reason) in {
            bfr.RETRY_RESUME,
            bfr.RETRY_COMPRESS_THEN_RESUME,
            bfr.RETRY_NONE,
        }


# ── relay deliver handler consumes the policy ────────────────────────────────


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
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

    monkeypatch.setattr("subprocess.run", _fake_run)
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

    monkeypatch.setattr("subprocess.run", _fake_run)
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

    monkeypatch.setattr("subprocess.run", _fake_run)
    out = _deliver({"profile": "ops", "message": "ping"})
    assert "error" in out
    assert len(_transport_calls(calls)) == 1, "auth failures must not auto-retry"
    # typed reason rides the structured error payload
    assert out["error"]["data"]["reason"] == bfr.PROVIDER_AUTH_OR_ACCESS


def test_deliver_failure_carries_typed_reason(home, monkeypatch):
    """A still-failing retryable error surfaces its classified reason."""
    monkeypatch.setattr(
        "subprocess.run",
        lambda argv, **k: _Proc(1, stderr="502 server error - overloaded")
        if _is_hermes_cli(list(argv))
        else _Proc(0),
    )
    out = _deliver({"profile": "ops", "message": "ping"})
    assert "error" in out
    assert out["error"]["data"]["reason"] == bfr.PROVIDER_SERVER_ERROR


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


# ── the stream a failed turn actually writes to ──────────────────────────────
#
# The CLI splits a failed turn across BOTH streams: provider prose on stdout,
# session bookkeeping (plus the typed refusal marker) on stderr. Every test
# above hands the error text to stderr and leaves stdout empty, which no real
# run ever does — so `stderr or stdout` looked correct while classifying only
# bookkeeping in production. These pin the real layout.

# Verbatim shape of a real failed delivery: stderr carries only bookkeeping.
REAL_STDERR = (
    'Session 20260908_050239_c0dd40 found but has no messages. Starting fresh.\n'
    '\nsession_id: 20260908_050239_c0dd40\n'
)
REAL_STDOUT_500 = 'API call failed after 3 retries: HTTP 500: server error: upstream overloaded\n'
REAL_STDOUT_OVERFLOW = "This model's maximum context length is 200000 tokens\n"


@pytest.mark.parametrize(
    "stdout,expected",
    [
        (REAL_STDOUT_500, bfr.PROVIDER_SERVER_ERROR),
        (REAL_STDOUT_OVERFLOW, bfr.CONTEXT_OVERFLOW),
        ('API call failed after 3 retries: Error code: 429 - rate limit exceeded\n', bfr.PROVIDER_RATE_LIMIT),
    ],
)
def test_failure_reason_reads_stdout_when_stderr_is_only_bookkeeping(stdout, expected):
    """The provider prose is on stdout; stderr's session lines must not mask it."""
    assert bfr.transport_failure_reason(_Proc(1, stdout=stdout, stderr=REAL_STDERR)) == expected


def test_failure_reason_still_reads_a_typed_refusal_on_stderr():
    """format_refusal_stderr deliberately rides stderr — that contract is unchanged."""
    proc = _Proc(1, stdout='', stderr='No LLM provider configured\n')
    assert bfr.transport_failure_reason(proc) == bfr.MISSING_CONFIG


def test_failure_reason_is_unknown_when_neither_stream_says_anything():
    assert bfr.transport_failure_reason(_Proc(1, stdout='', stderr=REAL_STDERR)) == 'unknown'


def test_run_delivery_retries_a_transient_failure_reported_on_stdout(monkeypatch, tmp_path):
    """The regression: a real 500 (prose on stdout, bookkeeping on stderr) must retry.

    Before the fix this ran the turn exactly once — the documented policy retry
    (#93091 item 5) never fired for any real provider failure.
    """
    from tools import bot_mode_dm

    dm = tmp_path / "dm.txt"
    dm.write_text("hello")
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        if len(calls) == 1:
            return _Proc(1, stdout=REAL_STDOUT_500, stderr=REAL_STDERR)
        return _Proc(0, stdout="the reply text", stderr=REAL_STDERR)

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    rc = bot_mode_dm._run_delivery(["hermes", "-p", "ops", "chat"], str(dm), stdin_file=False)

    assert rc == 0
    assert len(calls) == 2, "a transient failure reported on stdout must be retried"
    assert calls[0] == calls[1], "the retry re-runs the SAME session/argv"


def test_run_delivery_still_refuses_to_retry_auth_reported_on_stdout(monkeypatch, tmp_path):
    """Reading stdout must not turn a permanent class into a retry."""
    from tools import bot_mode_dm

    dm = tmp_path / "dm.txt"
    dm.write_text("hello")
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return _Proc(1, stdout="API call failed: authentication_error\n", stderr=REAL_STDERR)

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    bot_mode_dm._run_delivery(["hermes", "-p", "ops", "chat"], str(dm), stdin_file=False)
    assert len(calls) == 1


def test_the_retry_tells_the_reruns_process_to_adopt_the_persisted_row(monkeypatch, tmp_path):
    """The first attempt already persisted the user row, so the retried process must be told to
    adopt it (agent.turn_context.RESUME_UNANSWERED_TURN_ENV) instead of appending a second copy.
    The FIRST attempt must not carry the flag — nothing has been persisted for it yet."""
    from tools import bot_mode_dm

    dm = tmp_path / "dm.txt"
    dm.write_text("hello")
    envs = []

    def _fake_run(argv, **kwargs):
        envs.append((kwargs.get("env") or {}).get("HERMES_RESUME_UNANSWERED_TURN"))
        if len(envs) == 1:
            return _Proc(1, stdout=REAL_STDOUT_500, stderr=REAL_STDERR)
        return _Proc(0, stdout="the reply text", stderr=REAL_STDERR)

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    bot_mode_dm._run_delivery(["hermes", "-p", "ops", "chat"], str(dm), stdin_file=False)

    assert envs == [None, "1"], f"first attempt clean, retry opts in; got {envs}"


# ── the relay handler has the SAME retry gate ────────────────────────────────
#
# ``bot_relay.deliver`` (cross-connection DMs) classified with its own
# ``(stderr or stdout)`` helper, so its policy re-run was dead in production for exactly
# the same reason, and the typed ``reason`` it hands back to the sender was always
# 'unknown'. Both now go through ``transport_failure_reason``.


def test_the_relay_deliver_retries_a_failure_reported_on_stdout(home, monkeypatch):
    """Before: the relay never re-ran a transient failure, because stderr's session banner
    always won the classification."""
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(((kwargs.get("env") or {}).get("HERMES_RESUME_UNANSWERED_TURN")))
        if not _is_hermes_cli(list(argv)):
            return _Proc(0)
        if len(calls) == 1:
            return _Proc(1, stdout=REAL_STDOUT_500, stderr=REAL_STDERR)
        return _Proc(0, stdout="recovered reply", stderr=REAL_STDERR)

    monkeypatch.setattr("subprocess.run", _fake_run)
    out = _deliver({"profile": "ops", "message": "ping"})

    assert out["result"]["reply"] == "recovered reply"
    assert calls == [None, "1"], f"first attempt clean, re-run opts in; got {calls}"


def test_the_relay_deliver_reports_a_typed_reason_not_unknown(home, monkeypatch):
    """A permanent failure whose prose is on stdout must still reach the sender typed."""
    def _fake_run(argv, **kwargs):
        if not _is_hermes_cli(list(argv)):
            return _Proc(0)
        return _Proc(1, stdout="API call failed: authentication_error\n", stderr=REAL_STDERR)

    monkeypatch.setattr("subprocess.run", _fake_run)
    out = _deliver({"profile": "ops", "message": "ping"})

    assert "error" in out
    assert out["error"]["data"]["reason"] == bfr.PROVIDER_AUTH_OR_ACCESS

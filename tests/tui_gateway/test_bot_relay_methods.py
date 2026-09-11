"""Tests: bot_relay.* JSON-RPC handlers (tui_gateway/methods_bot_relay.py).

The Desktop's relay door on each connected gateway. Contracts:
- roster.sync persists validated rows and reports the accepted count;
- outbox.drain returns queued envelopes exactly once;
- deliver validates the target profile against THIS install and runs the
  one-turn Bot Chat transport (subprocess is faked here — the argv contract
  is what's pinned);
- reply writes the waiter's file and rejects malformed envelope ids.
"""

from __future__ import annotations

import json

import pytest

import tui_gateway.server as srv
from hermes_cli.dashboard_auth.ws_tickets import INTERNAL_PROVIDER, INTERNAL_USER_ID
from tools import bot_relay


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def test_roster_sync_persists_and_counts(home):
    out = _result(
        srv._methods["bot_relay.roster.sync"](
            1,
            {
                "agents": [
                    {"profile": "scout", "handle": "scout", "connection_id": "cloud-1"},
                    {"profile": "", "connection_id": "cloud-1"},  # dropped
                ]
            },
        )
    )
    assert out["count"] == 1
    assert [r["profile"] for r in bot_relay.read_remote_roster(home)] == ["scout"]


def test_outbox_drain_returns_each_envelope_once(home):
    target = {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
              "connection_label": "", "title": "", "description": ""}
    env = bot_relay.enqueue_envelope(
        home, target=target, message="m", sender_profile="default", sender_handle="hermes"
    )
    first = _result(srv._methods["bot_relay.outbox.drain"](1, {}))
    assert [e["id"] for e in first["envelopes"]] == [env["id"]]
    second = _result(srv._methods["bot_relay.outbox.drain"](2, {}))
    assert second["envelopes"] == []


def test_deliver_validates_profile_and_runs_transport(home, monkeypatch):
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "pong from ops"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    out = _result(
        srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping"})
    )
    assert out["reply"] == "pong from ops"
    # Decoding is pinned (#93590 sibling defect): without encoding= the
    # child's UTF-8 output is decoded with the locale codec — cp1252/GBK on
    # Windows — mangling non-ASCII replies; errors="replace" keeps a bad
    # byte from raising instead of delivering.
    assert calls["kwargs"]["encoding"] == "utf-8"
    assert calls["kwargs"]["errors"] == "replace"
    argv = calls["argv"]
    # argv[0] may be a resolved venv path (#93590) — match by basename.
    assert argv[1:3] == ["-p", "ops"]
    assert argv[0].rsplit("\\", 1)[-1].rsplit("/", 1)[-1] in ("hermes", "hermes.exe")
    assert "Bot Chat" in argv and "--query-file" in argv

    # 'hermes' alias resolves to default
    _result(srv._methods["bot_relay.deliver"](2, {"profile": "hermes", "message": "x"}))
    assert calls["argv"][1:3] == ["-p", "default"]

    # unknown profile refuses without spawning
    calls.clear()
    err = srv._methods["bot_relay.deliver"](3, {"profile": "ghost", "message": "x"})
    assert "error" in err and "ghost" in err["error"]["message"]
    assert not calls


def test_deliver_requires_params(home):
    err = srv._methods["bot_relay.deliver"](1, {"profile": "", "message": ""})
    assert "error" in err


def test_deliver_lands_in_live_bot_chat_instead_of_subprocess(home, monkeypatch):
    """#100523: a Desktop-owned Bot Chat receives the DM as a normal user turn.

    With the target's Bot Chat live in this gateway, the subprocess transport
    would be fenced out by the single-owner lease and drop the payload. The
    handler must route through prompt.submit (the composer's choke point) and
    never spawn the CLI.
    """
    spawned = []
    submitted = []

    class _Proc:
        returncode, stdout, stderr = 0, "pong", ""

    def _fake_run(argv, *a, **k):
        # The server module's import-time update prefetch runs `git ...` on a
        # daemon thread; only the relay's `hermes` CLI spawn is under test.
        if argv and argv[0] != "git":
            spawned.append(argv)
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setitem(
        srv._methods, "prompt.submit", lambda rid, p: submitted.append(p) or srv._ok(rid, {"status": "streaming"})
    )
    monkeypatch.setattr(srv, "_profile_home", lambda name: home / "profiles" / name)
    monkeypatch.setitem(
        srv._sessions,
        "live-ops",
        {"profile_home": str(home / "profiles" / "ops"), "pending_title": "Bot Chat", "history": []},
    )
    out = _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping"}))
    # queued=True is the invariant: a DM never interrupts a turn in flight.
    assert submitted == [{"session_id": "live-ops", "text": "ping", "queued": True}]
    assert not spawned
    assert "reply" in out

    # A live session titled anything else for the same profile does not qualify:
    # the subprocess path runs exactly as before.
    srv._sessions["live-ops"]["pending_title"] = "Scratch"
    submitted.clear()

    out = _result(srv._methods["bot_relay.deliver"](2, {"profile": "ops", "message": "ping"}))
    assert out["reply"] == "pong" and spawned and not submitted


def test_reply_roundtrip_and_id_validation(home):
    envelope_id = "c" * 32
    _result(srv._methods["bot_relay.reply"](1, {"id": envelope_id, "reply": "hi"}))
    path = bot_relay.relay_root(home) / bot_relay.REPLIES_DIR / f"{envelope_id}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["reply"] == "hi"

    err = srv._methods["bot_relay.reply"](2, {"id": "../evil"})
    assert "error" in err


@pytest.mark.platforms("linux")
def test_deliver_write_failure_still_removes_tempfile(home, monkeypatch, tmp_path):
    """A failed payload write must not leak the relay DM tempfile."""
    import glob
    import os
    import tempfile as _tempfile

    made = []
    real_mkstemp = _tempfile.mkstemp

    def _tracking_mkstemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        fd, path = real_mkstemp(*args, **kwargs)
        made.append(path)
        return fd, path

    class _BrokenWriter:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def write(self, content):
            raise OSError("disk full")

    monkeypatch.setattr("tempfile.mkstemp", _tracking_mkstemp)
    monkeypatch.setattr("os.fdopen", lambda *a, **k: _BrokenWriter())
    err = srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "x"})
    assert "error" in err
    assert made, "mkstemp was never reached"
    assert not glob.glob(str(tmp_path / "hermes-relay-dm-*")), "tempfile leaked"


@pytest.fixture
def fake_runs(monkeypatch):
    """Fake ``subprocess.run`` that records each call's kwargs; ``outcomes`` holds (returncode, stderr) per call."""
    calls, outcomes = [], []

    def _fake_run(argv, **kwargs):
        calls.append(kwargs)
        code, err = outcomes.pop(0) if outcomes else (0, "")

        class _Proc:
            returncode, stdout, stderr = code, "ok" if code == 0 else "", err

        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    return calls, outcomes


@pytest.mark.parametrize("sender, expected", [
    ({"from_profile": "scout", "from_handle": "scout"}, {"id": "bot:scout", "name": "scout", "is_bot": True}),
    ({"from_profile": "scout", "from_handle": "scout", "from_connection": "cloud-1"},
     {"id": "bot:cloud-1/scout", "name": "scout", "is_bot": True}),
    ({}, None),
], ids=["sender fields", "sender on another connection", "no sender fields"])
def test_deliver_child_env_carries_the_envelope_sender_on_every_attempt(home, monkeypatch, fake_runs, sender, expected):
    """HERMES_TURN_AUTHOR on the child comes from the envelope's sender fields alone: the retry gets the same
    author, and without sender fields a stale author on the gateway's own environment never reaches the child."""
    from agent.turn_author import TURN_AUTHOR_ENV

    calls, outcomes = fake_runs
    outcomes.extend([(1, "HTTP 429 rate limit"), (0, "")])
    monkeypatch.setenv("HERMES_RELAY_TEST_MARKER", "kept")
    monkeypatch.setenv(TURN_AUTHOR_ENV, json.dumps({"id": "bot:stale", "name": "stale", "is_bot": True}))

    _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping", **sender}))

    envs = [c["env"] for c in calls]
    assert len(envs) == 2
    assert [json.loads(e[TURN_AUTHOR_ENV]) if TURN_AUTHOR_ENV in e else None for e in envs] == [expected, expected]
    assert all(e["HERMES_RELAY_TEST_MARKER"] == "kept" for e in envs)


class _Client:
    def __init__(self, auth_identity=None):
        self.auth_identity = auth_identity

    def write(self, obj):
        return True

    def close(self):
        return None


@pytest.fixture
def bound_client(monkeypatch):
    """Bind a fake calling transport for the handler; yields a setter for its ``auth_identity``."""
    client = _Client()
    token = srv.bind_transport(client)
    try:
        yield client
    finally:
        srv.reset_transport(token)


SENDER = {"from_profile": "scout", "from_handle": "scout", "from_connection": "cloud-1"}
SENDER_AUTHOR = {"id": "bot:cloud-1/scout", "name": "scout", "is_bot": True}


@pytest.mark.parametrize("identity", [
    None,
    {"user_id": INTERNAL_USER_ID, "provider": INTERNAL_PROVIDER},
], ids=["no identity", "server-internal identity"])
def test_deliver_accepts_a_sender_from_an_admitted_non_login_client(home, fake_runs, bound_client, identity):
    """The Desktop and server-internal callers carry no login identity; their sender fields become the author."""
    from agent.turn_author import TURN_AUTHOR_ENV

    calls, _outcomes = fake_runs
    bound_client.auth_identity = identity

    _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping", **SENDER}))

    assert [json.loads(c["env"][TURN_AUTHOR_ENV]) for c in calls] == [SENDER_AUTHOR]


def test_deliver_refuses_a_sender_from_a_logged_in_client(home, fake_runs, bound_client):
    """A browser login never relays for another connection, so its from_* fields are refused before any turn runs.
    Without sender fields the same client still delivers, unattributed."""
    from agent.turn_author import TURN_AUTHOR_ENV

    calls, _outcomes = fake_runs
    bound_client.auth_identity = {"user_id": "alice", "provider": "google"}

    for sender in ({"from_profile": "scout"}, {"from_connection": "cloud-1"}, SENDER):
        err = srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping", **sender})
        assert err["error"]["code"] == 4095
    assert not calls

    _result(srv._methods["bot_relay.deliver"](2, {"profile": "ops", "message": "ping"}))
    assert len(calls) == 1 and TURN_AUTHOR_ENV not in calls[0]["env"]

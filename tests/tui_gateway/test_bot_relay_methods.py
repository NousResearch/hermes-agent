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
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    out = _result(
        srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping"})
    )
    assert out["reply"] == "pong from ops"
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


def test_reply_roundtrip_and_id_validation(home):
    envelope_id = "c" * 32
    _result(srv._methods["bot_relay.reply"](1, {"id": envelope_id, "reply": "hi"}))
    path = bot_relay.relay_root(home) / bot_relay.REPLIES_DIR / f"{envelope_id}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["reply"] == "hi"

    err = srv._methods["bot_relay.reply"](2, {"id": "../evil"})
    assert "error" in err


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

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
import time
from contextlib import contextmanager

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


def test_deliver_idempotency_replays_without_second_agent_turn(home, monkeypatch):
    calls = []

    class _Proc:
        returncode, stdout, stderr = 0, "one result", ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a) or _Proc())
    params = {
        "profile": "ops",
        "message": "ping",
        "message_id": "a" * 32,
        "idempotency_key": "mission:42:message:1",
        "envelope_schema": "asm-hermes-a2a-envelope/v2",
    }
    params["envelope"] = {
        "schema": params["envelope_schema"],
        "message_id": params["message_id"],
        "idempotency_key": params["idempotency_key"],
        "type": "REQUEST",
        "from_agent": "hermes",
        "to_agent": "ops",
        "target_connection": "cloud-1",
        "target_profile": "ops",
        "target_handle": "ops",
        "message": "ping",
        "scope": {"mutation": "none", "production": "none"},
        "expires_at": time.time() + 60,
        "authority_effect": "none",
    }
    first = _result(srv._methods["bot_relay.deliver"](1, params))
    second = _result(srv._methods["bot_relay.deliver"](2, params))
    assert first["reply"] == "one result"
    assert first["target_receipt"]["schema"] == "asm-hermes-a2a-target-receipt/v1"
    assert first["target_receipt"]["target_connection"] == "cloud-1"
    assert "already delivered" in second["reply"]
    assert second["replayed"] is True
    relay_calls = [call for call in calls if call and call[0][1:3] == ["-p", "ops"]]
    assert len(relay_calls) == 1

    readback = _result(srv._methods["bot_relay.receipt.read"](4, params))
    assert readback["receipt"] == first["target_receipt"]

    # Envelope expiry governs admission, not a completed receipt's durable
    # reconnect readback. The fingerprint is intentionally independent of TTL.
    expired_readback = {
        **params,
        "envelope": {**params["envelope"], "expires_at": time.time() - 1},
    }
    assert _result(srv._methods["bot_relay.receipt.read"](5, expired_readback))["receipt"] == first["target_receipt"]

    changed = dict(params)
    changed["envelope"] = {**params["envelope"], "mission_id": "different-mission"}
    err = srv._methods["bot_relay.deliver"](3, changed)
    assert err["error"]["data"]["reason"] == "idempotency_conflict"
    relay_calls = [call for call in calls if call and call[0][1:3] == ["-p", "ops"]]
    assert len(relay_calls) == 1


def test_receipt_readback_rejects_changed_target_identity(home, monkeypatch):
    class _Proc:
        returncode, stdout, stderr = 0, "one result", ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    params = {
        "profile": "ops",
        "message": "ping",
        "message_id": "c" * 32,
        "idempotency_key": "mission:42:receipt-identity",
        "envelope_schema": "asm-hermes-a2a-envelope/v2",
        "envelope": {
            "schema": "asm-hermes-a2a-envelope/v2",
            "message_id": "c" * 32,
            "idempotency_key": "mission:42:receipt-identity",
            "type": "REQUEST",
            "from_agent": "hermes",
            "to_agent": "ops",
            "target_connection": "cloud-1",
            "target_profile": "ops",
            "target_handle": "ops",
            "message": "ping",
            "scope": {"mutation": "none", "production": "none"},
            "expires_at": time.time() + 60,
            "authority_effect": "none",
        },
    }
    _result(srv._methods["bot_relay.deliver"](1, params))
    changed = {**params, "envelope": {**params["envelope"], "target_connection": "other-1"}}
    err = srv._methods["bot_relay.receipt.read"](2, changed)
    assert err["error"]["data"]["reason"] == "target_receipt_mismatch"


def test_deliver_holds_ambiguous_prior_attempt(home):
    fingerprint = json.dumps(
        {
            "target_profile": "ops", "message": "ping", "type": "legacy",
            "from_agent": "", "to_agent": "ops", "mission_id": "",
            "work_item_id": "", "request": {}, "scope": {},
            "authority_effect": "none",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    bot_relay.begin_idempotent_delivery(home, "mission:42:ambiguous", "b" * 32, fingerprint)
    err = srv._methods["bot_relay.deliver"](
        1,
        {
            "profile": "ops",
            "message": "ping",
            "message_id": "b" * 32,
            "idempotency_key": "mission:42:ambiguous",
        },
    )
    assert err["error"]["data"]["reason"] == "duplicate_ambiguous"


def test_deliver_rejects_unknown_structured_schema(home, monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    err = srv._methods["bot_relay.deliver"](
        1, {"profile": "ops", "message": "ping", "envelope_schema": "asm-hermes-a2a-envelope/v999"}
    )
    assert "unsupported" in err["error"]["message"]
    assert not calls


@pytest.mark.parametrize(
    ("envelope_update", "code", "reason"),
    [
        ({"expires_at": time.time() - 1}, 4098, "queued_expired"),
        ({"scope": {"mutation": "write", "production": "none"}}, 4099, "authority_missing"),
        ({"authority_effect": "grant"}, 4099, "authority_escalation"),
    ],
)
def test_deliver_preserves_typed_structured_validation_reasons(
    home, monkeypatch, envelope_update, code, reason,
):
    class _Proc:
        returncode, stdout, stderr = 0, "unused", ""

    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a) or _Proc())
    message_id = "e" * 32
    envelope = {
        "schema": "asm-hermes-a2a-envelope/v2",
        "message_id": message_id,
        "idempotency_key": "mission:typed-validation",
        "type": "REQUEST",
        "from_agent": "hermes",
        "to_agent": "ops",
        "target_connection": "cloud-1",
        "target_profile": "ops",
        "target_handle": "ops",
        "message": "ping",
        "scope": {"mutation": "none", "production": "none"},
        "expires_at": time.time() + 60,
        "authority_effect": "none",
    }
    envelope.update(envelope_update)
    err = srv._methods["bot_relay.deliver"](
        1,
        {
            "profile": "ops",
            "message": "ping",
            "message_id": message_id,
            "idempotency_key": envelope["idempotency_key"],
            "envelope_schema": envelope["schema"],
            "envelope": envelope,
        },
    )
    assert err["error"]["code"] == code
    assert err["error"]["data"]["reason"] == reason
    assert not calls


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


def test_structured_live_delivery_waits_for_exact_durable_target_message(home, monkeypatch):
    """A prompt.submit ACK is not a target receipt until SessionDB has the exact row."""
    submitted = []
    message_id = "f" * 32
    params = {
        "profile": "ops",
        "message": "persist me",
        "message_id": message_id,
        "idempotency_key": "mission:live-persist",
        "envelope_schema": "asm-hermes-a2a-envelope/v2",
        "envelope": {
            "schema": "asm-hermes-a2a-envelope/v2",
            "message_id": message_id,
            "idempotency_key": "mission:live-persist",
            "type": "REQUEST",
            "from_agent": "hermes",
            "to_agent": "ops",
            "target_connection": "cloud-1",
            "target_profile": "ops",
            "target_handle": "ops",
            "message": "persist me",
            "scope": {"mutation": "none", "production": "none"},
            "expires_at": time.time() + 60,
            "authority_effect": "none",
        },
    }

    class _DB:
        reads = 0

        def get_messages_as_conversation(self, key, *, include_row_ids=False):
            assert key == "live-session"
            assert include_row_ids is True
            self.reads += 1
            if self.reads == 1:
                return []
            return [{
                "role": "user",
                "content": "persist me",
                "message_id": message_id,
                "_row_id": 17,
            }]

    db = _DB()

    @contextmanager
    def _db_for_session(_session):
        yield db

    monkeypatch.setattr(srv, "_session_db", _db_for_session)
    monkeypatch.setattr(srv, "_profile_home", lambda name: home / "profiles" / name)
    monkeypatch.setattr(srv, "LIVE_TARGET_PERSIST_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(srv, "LIVE_TARGET_PERSIST_POLL_SECONDS", 0.001)
    monkeypatch.setitem(
        srv._methods,
        "prompt.submit",
        lambda rid, payload: submitted.append(payload) or srv._ok(rid, {"status": "streaming"}),
    )
    monkeypatch.setitem(
        srv._sessions,
        "live-ops-receipt",
        {
            "profile_home": str(home / "profiles" / "ops"),
            "session_key": "live-session",
            "pending_title": "Bot Chat",
            "history": [],
        },
    )

    out = _result(srv._methods["bot_relay.deliver"](1, params))
    assert submitted == [{
        "session_id": "live-ops-receipt",
        "text": "persist me",
        "queued": True,
        "_relay_message_id": message_id,
    }]
    assert out["target_receipt"]["message_id"] == message_id
    assert db.reads >= 2


def test_structured_live_delivery_does_not_upgrade_sender_ack_to_receipt(home, monkeypatch):
    """Same text without the exact platform id remains pending, never success."""
    message_id = "a" * 32
    params = {
        "profile": "ops",
        "message": "same text",
        "message_id": message_id,
        "idempotency_key": "mission:live-no-proof",
        "envelope_schema": "asm-hermes-a2a-envelope/v2",
        "envelope": {
            "schema": "asm-hermes-a2a-envelope/v2",
            "message_id": message_id,
            "idempotency_key": "mission:live-no-proof",
            "type": "REQUEST",
            "from_agent": "hermes",
            "to_agent": "ops",
            "target_connection": "cloud-1",
            "target_profile": "ops",
            "target_handle": "ops",
            "message": "same text",
            "scope": {"mutation": "none", "production": "none"},
            "expires_at": time.time() + 60,
            "authority_effect": "none",
        },
    }

    class _DB:
        def get_messages_as_conversation(self, _key, *, include_row_ids=False):
            return [{"role": "user", "content": "same text", "message_id": "b" * 32}]

    @contextmanager
    def _db_for_session(_session):
        yield _DB()

    monkeypatch.setattr(srv, "_session_db", _db_for_session)
    monkeypatch.setattr(srv, "_profile_home", lambda name: home / "profiles" / name)
    monkeypatch.setattr(srv, "LIVE_TARGET_PERSIST_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(srv, "LIVE_TARGET_PERSIST_POLL_SECONDS", 0.001)
    monkeypatch.setitem(
        srv._methods,
        "prompt.submit",
        lambda rid, payload: srv._ok(rid, {"status": "streaming"}),
    )
    monkeypatch.setitem(
        srv._sessions,
        "live-ops-no-proof",
        {
            "profile_home": str(home / "profiles" / "ops"),
            "session_key": "live-session-no-proof",
            "pending_title": "Bot Chat",
            "history": [],
        },
    )

    err = srv._methods["bot_relay.deliver"](1, params)
    assert err["error"]["code"] == 4101
    assert err["error"]["data"]["reason"] == "target_receipt_pending"


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

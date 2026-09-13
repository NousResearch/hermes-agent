"""Tests: bot_relay.* JSON-RPC handlers (tui_gateway/methods_bot_relay.py).

The Desktop's relay door on each connected gateway. Contracts:
- roster.sync persists validated rows and reports the accepted count;
- outbox.drain replays a canonical envelope (same id) until its reply lands;
- deliver requires a stable envelope id, resolves the target profile's home
  on THIS install and forwards to that profile's authority — never a CLI turn;
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


def test_outbox_drain_replays_canonical_envelope_until_reply_acknowledged(home):
    """A drained envelope is not "delivered" — its stable id is replayed on every
    drain until the target's terminal reply is written under that id, so a
    Desktop that lost the first drain result cannot drop the DM."""
    target = {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
              "connection_label": "", "title": "", "description": ""}
    env = bot_relay.enqueue_envelope(
        home, target=target, message="m", sender_profile="default", sender_handle="hermes"
    )
    first = _result(srv._methods["bot_relay.outbox.drain"](1, {}))
    assert [e["id"] for e in first["envelopes"]] == [env["id"]]
    second = _result(srv._methods["bot_relay.outbox.drain"](2, {}))
    assert [e["id"] for e in second["envelopes"]] == [env["id"]], "same id, never a duplicate envelope"
    assert second["envelopes"][0]["message"] == "m"
    _result(srv._methods["bot_relay.reply"](3, {"id": env["id"], "reply": "done"}))
    assert _result(srv._methods["bot_relay.outbox.drain"](4, {}))["envelopes"] == []


def test_deliver_forwards_stable_id_to_target_profile_authority(home, monkeypatch):
    """The relay door is a transport bridge: it resolves the target profile's
    own home and forwards the envelope id + message to THAT authority. It
    never runs a CLI turn of its own."""
    from tools import bot_live_delivery as live

    forwarded = []
    spawned = []

    def _fake_run(argv, *a, **k):
        # The server module's import-time update prefetch runs `git ...`; only
        # a `hermes` spawn would be a delivery attempt.
        if argv and argv[0] != "git":
            spawned.append(argv)
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def _fake_authority(target_home, params):
        forwarded.append((target_home, dict(params)))
        return {"status": "queued", "delivery_id": params["id"], "reply": ""}

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(live, "authority_delivery", _fake_authority)
    envelope_id = "b" * 32
    out = _result(srv._methods["bot_relay.deliver"](1, {"id": envelope_id, "profile": "ops", "message": "ping"}))
    assert out["status"] == "queued" and out["delivery_id"] == envelope_id
    assert forwarded[-1][0] == home / "profiles" / "ops"
    assert forwarded[-1][1]["id"] == envelope_id and forwarded[-1][1]["profile"] == "ops"

    # Exact retry carries the SAME id to the same authority — no second envelope.
    _result(srv._methods["bot_relay.deliver"](2, {"id": envelope_id, "profile": "ops", "message": "ping"}))
    assert [p["id"] for _h, p in forwarded] == [envelope_id, envelope_id]

    # 'hermes' alias resolves to the default profile's home.
    _result(srv._methods["bot_relay.deliver"](3, {"id": "c" * 32, "profile": "hermes", "message": "x"}))
    assert forwarded[-1][0] == home and forwarded[-1][1]["profile"] == "default"
    assert not spawned


def test_deliver_unreachable_authority_is_a_typed_refusal(home, monkeypatch):
    from tools import bot_live_delivery as live

    def _down(target_home, params):
        raise ValueError("profile authority is not ready")

    monkeypatch.setattr(live, "authority_delivery", _down)
    err = srv._methods["bot_relay.deliver"](1, {"id": "d" * 32, "profile": "ghost", "message": "x"})
    assert err["error"]["data"]["reason"] == "runtime_unavailable"
    assert "not ready" in err["error"]["message"]


@pytest.mark.parametrize("params", [
    {"profile": "", "message": ""},
    {"profile": "ops", "message": "no envelope id"},
    {"id": "../evil", "profile": "ops", "message": "x"},
    {"id": "e" * 32, "profile": "../ops", "message": "x"},
])
def test_deliver_requires_stable_id_and_valid_profile(home, monkeypatch, params):
    from tools import bot_live_delivery as live

    monkeypatch.setattr(live, "authority_delivery",
                        lambda *a, **k: pytest.fail("malformed requests never reach an authority"))
    err = srv._methods["bot_relay.deliver"](1, params)
    assert err["error"]["data"]["reason"] == "invalid_params"


def test_reply_roundtrip_and_id_validation(home):
    envelope_id = "c" * 32
    _result(srv._methods["bot_relay.reply"](1, {"id": envelope_id, "reply": "hi"}))
    path = bot_relay.relay_root(home) / bot_relay.REPLIES_DIR / f"{envelope_id}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["reply"] == "hi"

    err = srv._methods["bot_relay.reply"](2, {"id": "../evil"})
    assert "error" in err


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


@pytest.mark.parametrize("identity, refused", [
    (None, False),
    ({"user_id": INTERNAL_USER_ID, "provider": INTERNAL_PROVIDER}, False),
    ({"user_id": "alice", "provider": "google"}, True),
])
def test_relay_sender_attribution_obeys_transport_identity(home, monkeypatch, bound_client, identity, refused):
    from tools import bot_live_delivery as live
    forwarded = []
    monkeypatch.setattr(live, "authority_delivery",
                        lambda home, params: forwarded.append(params) or {"status": "queued"})
    bound_client.auth_identity = identity
    result = srv._methods["bot_relay.deliver"](1, {
        "id": "f" * 32, "profile": "ops", "message": "ping", **SENDER})
    if refused:
        assert result["error"]["code"] == 4095
        assert not forwarded
    else:
        assert _result(result)["status"] == "queued"
        assert forwarded[0]["author"] == SENDER_AUTHOR
        assert not any(key in forwarded[0] for key in SENDER)

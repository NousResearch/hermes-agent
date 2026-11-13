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
from pathlib import Path

import pytest

import tui_gateway.server as srv
from hermes_cli.dashboard_auth.ws_tickets import INTERNAL_PROVIDER, INTERNAL_USER_ID
from tools import bot_relay


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    (h / "profiles" / "ops" / "config.yaml").write_text("{}\n")  # identity marker: a bare dir is no target
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


def test_outbox_drain_settles_a_claimed_envelope_nobody_answered_with_a_typed_timeout(home):
    """A Desktop that disconnects between ``outbox.drain`` and ``bot_relay.deliver`` leaves the envelope in
    ``claimed/`` with no reply (#111021, #111207). It is replayed on every drain — same id, so the target
    authority admits one turn — until the waiter's ``REPLY_WAIT_SECONDS`` run out; then the drain writes a
    ``delivery_timeout`` reply so the sender learns, and never hands the envelope out again."""
    import time

    target = {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
              "connection_label": "", "title": "", "description": ""}
    lost = bot_relay.enqueue_envelope(home, target=target, message="lost", sender_profile="w", sender_handle="w")
    done = bot_relay.enqueue_envelope(home, target=target, message="done", sender_profile="w", sender_handle="w")
    drain = srv._methods["bot_relay.outbox.drain"]
    assert sorted(e["id"] for e in _result(drain(1, {}))["envelopes"]) == sorted([lost["id"], done["id"]])
    bot_relay.write_reply(home, done["id"], reply="answered")
    # An answered envelope is never re-offered; the unanswered one rides every drain.
    assert [e["id"] for e in _result(drain(2, {}))["envelopes"]] == [lost["id"]]
    lost_path = bot_relay.relay_root(home) / bot_relay.CLAIMED_DIR / f"{lost['id']}.json"
    stale = json.loads(lost_path.read_text(encoding="utf-8"))
    stale["created_at"] = int(time.time()) - bot_relay.REPLY_WAIT_SECONDS - 1
    lost_path.write_text(json.dumps(stale), encoding="utf-8")
    assert _result(drain(3, {}))["envelopes"] == []
    reply = json.loads((bot_relay.relay_root(home) / bot_relay.REPLIES_DIR / f"{lost['id']}.json").read_text(encoding="utf-8"))
    assert reply["reason"] == "delivery_timeout" and reply["error"]
    assert _result(drain(4, {}))["envelopes"] == []


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
    err = srv._methods["bot_relay.deliver"](1, {"id": "d" * 32, "profile": "ops", "message": "x"})
    assert err["error"]["data"]["reason"] == "runtime_unavailable"
    assert "not ready" in err["error"]["message"]
    # A name that is not a live profile (#99392: infra dirs and bare shells are not teammates)
    # is refused before any authority is consulted.
    (home / "profiles" / "sessions").mkdir()
    for ghost in ("ghost", "sessions"):
        err = srv._methods["bot_relay.deliver"](2, {"id": "e" * 32, "profile": ghost, "message": "x"})
        assert err["error"]["data"]["reason"] == "unknown_profile", ghost


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


@pytest.mark.parametrize("identity", [
    None,
    {"user_id": INTERNAL_USER_ID, "provider": INTERNAL_PROVIDER},
], ids=["no identity", "server-internal identity"])
def test_deliver_accepts_a_sender_from_an_admitted_non_login_client(home, monkeypatch, bound_client, identity):
    """A caller with no identity, or one holding the ``?internal=`` credential, keeps its sender fields.

    NOT the Desktop: it mints a ws-ticket carrying the signed-in ``{user_id, provider}`` on every
    gateway that requires sign-in (``hermes_cli/dashboard_auth/routes.py``), so it is a login
    identity and takes the principal-author branch below."""
    from tools import bot_live_delivery as live
    forwarded = []
    monkeypatch.setattr(live, "authority_delivery",
                        lambda home, params: forwarded.append(params) or {"status": "queued"})
    bound_client.auth_identity = identity
    result = srv._methods["bot_relay.deliver"](1, {
        "id": "f" * 32, "profile": "ops", "message": "ping", **SENDER})
    assert _result(result)["status"] == "queued"
    assert forwarded[0]["author"] == SENDER_AUTHOR
    assert not any(key in forwarded[0] for key in SENDER)


def test_deliver_from_a_logged_in_client_is_attributed_to_its_principal_never_to_the_claimed_sender(
        home, monkeypatch, bound_client):
    """A logged-in client's sender fields are not trusted — but the dm is neither refused nor left unattributed.

    Refusing the CALL took cross-machine relay offline for every auth-gated gateway, because the Desktop is
    itself a logged-in client there. Dropping the AUTHOR would make the turn the human's to the recipient's
    memory. So the author is derived from the caller's minted identity: stable, unspoofable, and still a bot —
    whether or not the client named a sender — and the authority receives it with the sender fields stripped."""
    from tools import bot_live_delivery as live
    forwarded = []
    monkeypatch.setattr(live, "authority_delivery",
                        lambda home, params: forwarded.append(params) or {"status": "queued"})
    bound_client.auth_identity = {"user_id": "alice", "provider": "google"}

    shapes = ({"from_profile": "scout"}, {"from_connection": "cloud-1"}, SENDER, {})
    for rid, sender in enumerate(shapes):
        _result(srv._methods["bot_relay.deliver"](rid, {"id": "f" * 32, "profile": "ops", "message": "ping", **sender}))

    assert len(forwarded) == len(shapes), "every relayed dm from a logged-in client must still be admitted"
    authors = [p["author"] for p in forwarded]
    assert all(a["is_bot"] is True for a in authors), "a relayed dm stays bot-authored for the recipient's memory"
    assert all(a["id"].startswith("bot:principal:dashboard:") and a["id"].endswith("/relay") for a in authors)
    assert all(a["name"] == "relayed teammate" for a in authors)
    assert SENDER_AUTHOR not in authors, "the claimed sender must not become the author"
    assert len({a["id"] for a in authors}) == 1, "one signed-in principal, one author — with or without sender fields"
    assert not any(key in p for p in forwarded for key in SENDER)

    bound_client.auth_identity = {"user_id": "bob", "provider": "google"}
    _result(srv._methods["bot_relay.deliver"](9, {"id": "f" * 32, "profile": "ops", "message": "ping", **SENDER}))
    assert forwarded[-1]["author"]["id"] != authors[0]["id"], "a different principal is a different author"


def test_deliver_restamps_relayed_sender_with_a_reply_safe_handle(home, monkeypatch):
    """#103731: the sender signs with its bare @handle, which for another machine's ``default`` is
    ``@hermes`` — the recipient's OWN default. The text forwarded to the target authority names the
    sender by the form this gateway resolves back to it: its title slug when the local relay roster
    carries it, else ``handle@connection``. A stamp that is not the relay's is left alone."""
    from tools import bot_live_delivery as live
    seen = []
    monkeypatch.setattr(live, "authority_delivery",
                        lambda home, params: seen.append(params["message"]) or {"status": "queued"})
    bot_relay.write_remote_roster(home, [
        {"profile": "default", "handle": "hermes", "connection_id": "vps-1", "title": "CoS Bot"},
    ])
    stamp = "Message from 🤖 CoS Bot (@hermes): are we done?"
    sender = {"from_profile": "default", "from_handle": "hermes", "from_connection": "vps-1"}
    _result(srv._methods["bot_relay.deliver"](1, {"id": "a" * 32, "profile": "ops", "message": stamp, **sender}))
    _result(srv._methods["bot_relay.deliver"](2, {"id": "b" * 32, "profile": "ops", "message": stamp, **sender,
                                                 "from_connection": "lan-2"}))
    _result(srv._methods["bot_relay.deliver"](3, {"id": "c" * 32, "profile": "ops", "message": "plain text (@hermes): x",
                                                 **sender}))
    assert seen == ["Message from 🤖 CoS Bot (@cos-bot): are we done?",
                    "Message from 🤖 CoS Bot (@hermes@lan-2): are we done?",
                    "plain text (@hermes): x"]


@pytest.mark.parametrize("subdir", ["profiles/ops", "dev"])
def test_gateway_drains_the_mailbox_the_tools_write_to(tmp_path, monkeypatch, subdir):
    """Both ends of the relay mailbox derive the install root from HERMES_HOME with ONE formula.
    The writer side (``message_agent``'s ``_hermes_root``) and the drain side
    (``methods_bot_relay._relay_root``) must agree for a ``profiles/<name>`` home AND for an
    arbitrary subdir of the native ``~/.hermes`` — a split here is silent non-delivery."""
    from tools.bot_mode_probe import _default_home, _hermes_root
    from tui_gateway import methods_bot_relay

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes" / subdir
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    writer_root = _hermes_root(Path(_default_home()))
    target = {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
              "connection_label": "", "title": "", "description": ""}
    env = bot_relay.enqueue_envelope(
        writer_root, target=target, message="m", sender_profile="default", sender_handle="hermes")

    assert methods_bot_relay._relay_root() == writer_root
    drained = _result(srv._methods["bot_relay.outbox.drain"](1, {}))
    assert [e["id"] for e in drained["envelopes"]] == [env["id"]]

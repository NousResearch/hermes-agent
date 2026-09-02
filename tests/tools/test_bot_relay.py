"""Tests: cross-connection bot relay (tools/bot_relay.py + message_agent route).

Connections ARE the peer set: every Desktop-connected gateway must be
message_agent-reachable. These tests pin the gateway-side plumbing —
roster validation, target resolution (incl. ambiguity), outbox claim
atomicity, reply write validation — and the two behavior contracts the
relay adds to message_agent:

- a target resolving against the Desktop-synced relay roster is queued as
  an envelope and acknowledged like any DM (fire-and-forget, waiter spawned);
- the legacy-SOUL dedupe (empty protocol section) NO LONGER strips the tool:
  the injection/execution gates key on managed-install, not section text.
"""

import json
import re
from pathlib import Path

import pytest

from tools import bot_relay
from tools.bot_mode_dm import (
    MESSAGE_AGENT_TOOL_NAME,
    ensure_message_agent_tool,
    message_agent_tool,
)


@pytest.fixture()
def root(tmp_path):
    return tmp_path


def _rows():
    return [
        {
            "profile": "default",
            "handle": "hermes",
            "connection_id": "cloud-1",
            "connection_label": "Hermes Cloud",
            "title": "Moxie",
            "description": "Main cloud agent",
        },
        {
            "profile": "researcher",
            "handle": "researcher",
            "connection_id": "ssh-vps",
            "connection_label": "VPS",
        },
    ]


# ── roster ───────────────────────────────────────────────────────────────────


def test_roster_roundtrip_and_validation(root):
    rows = _rows() + [
        {"profile": "", "handle": "x", "connection_id": "c"},  # no profile
        {"profile": "bad name!", "connection_id": "c"},  # bad charset
        "not-a-dict",
        {"profile": "default", "handle": "hermes", "connection_id": "cloud-1"},  # dupe
    ]
    count = bot_relay.write_remote_roster(root, rows)
    assert count == 2
    back = bot_relay.read_remote_roster(root)
    assert [r["profile"] for r in back] == ["default", "researcher"]
    assert back[0]["title"] == "Moxie"


def test_roster_read_missing_and_corrupt(root):
    assert bot_relay.read_remote_roster(root) == []
    base = bot_relay.relay_root(root)
    base.mkdir(parents=True)
    (base / bot_relay.ROSTER_FILE).write_text("{corrupt", encoding="utf-8")
    assert bot_relay.read_remote_roster(root) == []


def test_resolve_remote_target_forms(root):
    bot_relay.write_remote_roster(root, _rows())
    roster = bot_relay.read_remote_roster(root)
    assert bot_relay.resolve_remote_target("researcher", roster)["connection_id"] == "ssh-vps"
    assert bot_relay.resolve_remote_target("@hermes", roster)["profile"] == "default"
    # profile name resolves too
    assert bot_relay.resolve_remote_target("default", roster)["connection_id"] == "cloud-1"
    # exact connection-qualified form
    assert bot_relay.resolve_remote_target("hermes@cloud-1", roster)["profile"] == "default"
    assert bot_relay.resolve_remote_target("hermes@nope", roster) is None
    assert bot_relay.resolve_remote_target("ghost", roster) is None


def test_resolve_ambiguous_handle_across_connections(root):
    rows = _rows() + [
        {"profile": "researcher", "handle": "researcher", "connection_id": "cloud-1"}
    ]
    bot_relay.write_remote_roster(root, rows)
    roster = bot_relay.read_remote_roster(root)
    assert bot_relay.resolve_remote_target("researcher", roster) == "ambiguous"
    match = bot_relay.resolve_remote_target("researcher@ssh-vps", roster)
    assert match["connection_id"] == "ssh-vps"
    forms = bot_relay.remote_target_forms(roster)
    assert "researcher@ssh-vps" in forms and "researcher@cloud-1" in forms
    assert "hermes" in forms  # unique handle stays bare


# ── outbox / replies ─────────────────────────────────────────────────────────


def test_enqueue_claim_is_atomic_and_single_shot(root):
    bot_relay.write_remote_roster(root, _rows())
    roster = bot_relay.read_remote_roster(root)
    target = bot_relay.resolve_remote_target("researcher", roster)
    env = bot_relay.enqueue_envelope(
        root, target=target, message="hi", sender_profile="work", sender_handle="work"
    )
    assert re.match(r"^[0-9a-f]{32}$", env["id"])
    claimed = bot_relay.claim_pending_envelopes(root)
    assert [e["id"] for e in claimed] == [env["id"]]
    assert claimed[0]["target_connection"] == "ssh-vps"
    assert claimed[0]["message"] == "hi"
    # second drain: nothing (no double delivery)
    assert bot_relay.claim_pending_envelopes(root) == []


def test_write_reply_validates_envelope_id(root):
    with pytest.raises(ValueError):
        bot_relay.write_reply(root, "../../etc/passwd", reply="x")
    path = bot_relay.write_reply(root, "a" * 32, reply="pong")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["reply"] == "pong" and not data["error"]


def test_write_reply_reason_passthrough_and_classification(root):
    # explicit reason is persisted verbatim
    path = bot_relay.write_reply(root, "c" * 32, error="boom", reason="delivery_timeout")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["reason"] == "delivery_timeout" and data["error"] == "boom"
    # no reason given → classified from error text
    path = bot_relay.write_reply(root, "d" * 32, error="Error code: 429 - rate limit")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["reason"] == "provider_rate_limit"
    # success reply carries an empty reason
    path = bot_relay.write_reply(root, "e" * 32, reply="ok")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["reason"] == "" and data["reply"] == "ok"


def test_waiter_command_quotes_and_targets_reply_file(root):
    env = {"id": "b" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}
    cmd = bot_relay.waiter_command(root, env)
    assert ("b" * 32) in cmd and "-c" in cmd
    assert "rm -rf" not in cmd  # sanity: single quoted -c payload


def test_waiter_picks_up_reply_within_a_sub_second_cadence(root):
    """The reply file is written once; the waiter must notice it fast, not
    on a multi-second sleep (dead air the sender's completion notification
    inherits on every cross-machine reply)."""
    import shlex
    import subprocess
    import threading
    import time

    env = {"id": "c" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}
    reply_path = bot_relay.relay_root(root) / bot_relay.REPLIES_DIR / f"{env['id']}.json"
    reply_path.parent.mkdir(parents=True, exist_ok=True)

    def write_reply():
        time.sleep(0.3)
        reply_path.write_text(json.dumps({"reply": "pong"}), encoding="utf-8")

    threading.Thread(target=write_reply, daemon=True).start()
    started = time.monotonic()
    proc = subprocess.run(shlex.split(bot_relay.waiter_command(root, env)), capture_output=True, text=True, timeout=10)
    elapsed = time.monotonic() - started
    assert proc.returncode == 0 and "pong" in proc.stdout
    assert elapsed < 1.5, f"waiter took {elapsed:.2f}s to notice a reply written at 0.3s"


def test_roster_rejects_connection_id_outside_handle_charset(root):
    bad = [
        {"profile": "researcher", "handle": "researcher", "connection_id": "vps'); print(1)"},
        {"profile": "researcher", "handle": "researcher", "connection_id": "foo'bar"},
        {"profile": "researcher", "handle": "researcher", "connection_id": "ssh vps"},
        {"profile": "researcher", "handle": "researcher", "connection_id": "a" * 65},
    ]
    assert bot_relay.write_remote_roster(root, bad) == 0
    good = {
        "profile": "researcher",
        "handle": "researcher",
        "connection_id": "ssh-vps",
    }
    assert bot_relay.write_remote_roster(root, [good]) == 1


def test_waiter_command_repr_encodes_hostile_connection_id(root):
    import ast
    import shlex

    inj = "x'); open(r'/tmp/pwned','w').write('pwned'); print('x"
    env = {
        "id": "c" * 32,
        "target_handle": "researcher",
        "target_connection": inj,
    }
    cmd = bot_relay.waiter_command(root, env)
    parts = shlex.split(cmd)
    code = parts[parts.index("-c") + 1]
    compile(code, "<waiter>", "exec")
    tree = ast.parse(code)
    opens = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "open"
    ]
    # Only json.load(open(p, ...)) is a real open(); the payload must stay data.
    assert len(opens) == 1

    # A quote in the id used to SyntaxError the waiter. It must compile.
    quoted = bot_relay.waiter_command(
        root,
        {"id": "a" * 32, "target_handle": "h", "target_connection": "foo'bar"},
    )
    qcode = shlex.split(quoted)[shlex.split(quoted).index("-c") + 1]
    compile(qcode, "<waiter-quote>", "exec")


# ── message_agent integration: relay route + legacy-SOUL gate fix ───────────

import textwrap


def _managed_home(tmp_path, *, legacy_soul=False):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    d = home / "profiles" / "researcher"
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.yaml").write_text(
        textwrap.dedent(
            """\
            description: teammate for tests
            ui_meta:
              hermes-bots:
                shape: cloud
            """
        ),
        encoding="utf-8",
    )
    if legacy_soul:
        (home / "SOUL.md").write_text(
            "# Soul\n\n## Messaging other agents\nold shellout protocol\n",
            encoding="utf-8",
        )
    return home


class _FakeDB:
    def __init__(self, home, title):
        self.db_path = str(home / "state.db")
        self._title = title

    def get_session_title(self, _sid):
        return self._title


class _FakeAgent:
    def __init__(self, home, title="Bot Chat"):
        self._session_db = _FakeDB(home, title)
        self.session_id = "sess-1"
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self.tools: list = []
        self.valid_tool_names: set = set()


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    from tools import bot_mode_probe

    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def test_tool_injects_despite_legacy_soul_protocol(tmp_path):
    """The legacy-SOUL dedupe empties the SECTION, never the TOOL.

    Regression: upgraded installs whose SOUL.md still carries the old
    plugin-appended protocol silently lost message_agent because the gate
    keyed on section non-emptiness.
    """
    from tools import bot_mode_probe

    home = _managed_home(tmp_path, legacy_soul=True)
    # Premise: the dedupe really does empty the section for this profile...
    assert bot_mode_probe.get_bot_mode_protocol_section(home) == ""
    # ...but the install is managed, so the tool must still inject.
    agent = _FakeAgent(home)
    assert ensure_message_agent_tool(agent) is True
    assert [t["function"]["name"] for t in agent.tools] == [MESSAGE_AGENT_TOOL_NAME]


def test_relay_route_queues_envelope_and_spawns_waiter(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    bot_relay.write_remote_roster(home, [
        {"profile": "default", "handle": "hermes", "connection_id": "cloud-1",
         "connection_label": "Hermes Cloud", "title": "Moxie"},
    ])

    spawned = {}

    def _fake_spawn(command, label, *, task_id, agent):
        spawned["command"] = command
        spawned["label"] = label
        return json.dumps({"status": "sent", "to": label})

    monkeypatch.setattr("tools.bot_mode_dm._spawn_delivery", _fake_spawn)
    agent = _FakeAgent(home)
    out = json.loads(message_agent_tool(target="hermes", message="ping", agent=agent))
    assert out.get("status") == "sent"
    assert "Hermes Cloud" in spawned["label"]
    # envelope landed in the outbox with attribution prefixed
    pending = bot_relay.claim_pending_envelopes(home)
    assert len(pending) == 1
    assert pending[0]["target_connection"] == "cloud-1"
    assert pending[0]["target_profile"] == "default"
    assert pending[0]["message"].startswith("Message from 🤖 hermes (@hermes): ping")
    # waiter watches this envelope's reply file
    assert pending[0]["id"] in spawned["command"]


def test_relay_route_ambiguous_target_errors_with_forms(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    bot_relay.write_remote_roster(home, [
        {"profile": "scout", "handle": "scout", "connection_id": "cloud-1"},
        {"profile": "scout", "handle": "scout", "connection_id": "ssh-vps"},
    ])
    monkeypatch.setattr(
        "tools.bot_mode_dm._spawn_delivery",
        lambda *a, **k: json.dumps({"status": "sent"}),
    )
    agent = _FakeAgent(home)
    out = json.loads(message_agent_tool(target="scout", message="hi", agent=agent))
    assert "scout@cloud-1" in out.get("error", "") and "scout@ssh-vps" in out["error"]
    # connection-qualified form goes through
    out2 = json.loads(message_agent_tool(target="scout@ssh-vps", message="hi", agent=agent))
    assert out2.get("status") == "sent"


def test_unknown_target_error_mentions_connected_machines(tmp_path):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home)
    out = json.loads(message_agent_tool(target="ghost", message="hi", agent=agent))
    assert "connected machine" in out.get("error", "")


def test_protocol_section_lists_remote_teammates(tmp_path):
    from tools import bot_mode_probe

    home = _managed_home(tmp_path)
    bot_relay.write_remote_roster(home, [
        {"profile": "default", "handle": "hermes", "connection_id": "cloud-1",
         "connection_label": "Hermes Cloud", "title": "Moxie"},
    ])
    section = bot_mode_probe.get_bot_mode_protocol_section(home, force_refresh=True)
    assert "OTHER connected machines" in section
    assert "`@hermes` — on Hermes Cloud — Moxie" in section


def test_capability_fingerprint_changes_with_relay_roster(tmp_path):
    from tools import bot_mode_probe

    home = _managed_home(tmp_path)
    before = bot_mode_probe.capability_fingerprint(home)
    bot_relay.write_remote_roster(home, [
        {"profile": "default", "handle": "hermes", "connection_id": "cloud-1"},
    ])
    after = bot_mode_probe.capability_fingerprint(home)
    assert before != after  # eternal Bot Chats refresh once on roster change


# ── stale artifact sweep (housekeeping contract) ─────────────────────────────


def test_cleanup_bot_relay_artifacts_sweeps_stale_plaintext(tmp_path, monkeypatch):
    import os as _os
    import time as _time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
              "connection_label": "", "title": "", "description": ""}
    stale_env = bot_relay.enqueue_envelope(
        tmp_path, target=target, message="old secret",
        sender_profile="default", sender_handle="hermes",
    )
    fresh_env = bot_relay.enqueue_envelope(
        tmp_path, target=target, message="new secret",
        sender_profile="default", sender_handle="hermes",
    )
    base = bot_relay.relay_root(tmp_path)
    stale_reply = bot_relay.write_reply(tmp_path, stale_env["id"], reply="done")
    old = _time.time() - bot_relay.STALE_AFTER_SECONDS - 1
    _os.utime(base / bot_relay.OUTBOX_DIR / f"{stale_env['id']}.json", (old, old))
    _os.utime(stale_reply, (old, old))

    removed = bot_relay.cleanup_bot_relay_artifacts()

    assert removed == 2
    assert not (base / bot_relay.OUTBOX_DIR / f"{stale_env['id']}.json").exists()
    assert not stale_reply.exists()
    assert (base / bot_relay.OUTBOX_DIR / f"{fresh_env['id']}.json").exists()


def test_cleanup_bot_relay_artifacts_missing_dir_is_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nope"))
    assert bot_relay.cleanup_bot_relay_artifacts() == 0


# ── #93091 item 2: offline fail-fast + drain-time TTL ────────────────────────

import os as _os2
import time as _time2


def _target(conn="cloud-1", profile="scout", handle="scout"):
    return {"profile": profile, "handle": handle, "connection_id": conn,
            "connection_label": "", "title": "", "description": ""}


def test_enqueue_fails_fast_when_row_explicitly_offline(root):
    bot_relay.write_remote_roster(root, [
        {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
         "online": False},
    ])
    roster = bot_relay.read_remote_roster(root)
    assert roster[0]["online"] is False  # additive field survives normalize
    with pytest.raises(bot_relay.EnvelopeRefusedError) as ei:
        bot_relay.enqueue_envelope(
            root, target=roster[0], message="hi",
            sender_profile="default", sender_handle="hermes",
        )
    assert ei.value.reason == "runtime_offline"
    assert "offline" in str(ei.value)
    # nothing was written to the outbox
    outdir = bot_relay.relay_root(root) / bot_relay.OUTBOX_DIR
    assert not outdir.exists() or list(outdir.glob("*.json")) == []


def test_enqueue_fails_fast_when_target_absent_from_fresh_roster(root):
    bot_relay.write_remote_roster(root, _rows())  # fresh, no 'scout' row
    with pytest.raises(bot_relay.EnvelopeRefusedError) as ei:
        bot_relay.enqueue_envelope(
            root, target=_target(), message="hi",
            sender_profile="default", sender_handle="hermes",
        )
    assert ei.value.reason == "runtime_offline"


def test_enqueue_fails_open_when_liveness_unknown(root):
    # 1. no roster ever synced → unknown → enqueue
    env = bot_relay.enqueue_envelope(
        root, target=_target(), message="hi",
        sender_profile="default", sender_handle="hermes",
    )
    assert (bot_relay.relay_root(root) / bot_relay.OUTBOX_DIR / f"{env['id']}.json").exists()
    # 2. stale roster missing the target → unknown → enqueue
    bot_relay.write_remote_roster(root, _rows())
    roster_path = bot_relay.relay_root(root) / bot_relay.ROSTER_FILE
    old = _time2.time() - bot_relay.ROSTER_FRESH_SECONDS - 5
    _os2.utime(roster_path, (old, old))
    env2 = bot_relay.enqueue_envelope(
        root, target=_target(), message="hi again",
        sender_profile="default", sender_handle="hermes",
    )
    assert (bot_relay.relay_root(root) / bot_relay.OUTBOX_DIR / f"{env2['id']}.json").exists()
    # 3. fresh roster listing the target without an online flag → enqueue
    bot_relay.write_remote_roster(root, _rows())
    target = bot_relay.read_remote_roster(root)[1]  # researcher@ssh-vps
    env3 = bot_relay.enqueue_envelope(
        root, target=target, message="hello",
        sender_profile="default", sender_handle="hermes",
    )
    assert (bot_relay.relay_root(root) / bot_relay.OUTBOX_DIR / f"{env3['id']}.json").exists()


def test_drain_expires_old_envelope_with_queued_expired_reply(root):
    env = bot_relay.enqueue_envelope(
        root, target=_target(), message="too late",
        sender_profile="default", sender_handle="hermes",
    )
    base = bot_relay.relay_root(root)
    out_path = base / bot_relay.OUTBOX_DIR / f"{env['id']}.json"
    # backdate the envelope beyond the TTL
    env["created_at"] = int(_time2.time()) - bot_relay.DEFAULT_ENVELOPE_TTL_SECONDS - 10
    out_path.write_text(json.dumps(env), encoding="utf-8")

    claimed = bot_relay.claim_pending_envelopes(root)

    assert claimed == []  # not delivered
    assert not out_path.exists()  # expired outbox file removed
    reply = json.loads(
        (base / bot_relay.REPLIES_DIR / f"{env['id']}.json").read_text(encoding="utf-8")
    )
    assert reply["reason"] == "queued_expired"
    assert "expired" in reply["error"] and "NOT delivered" in reply["error"]
    assert not reply["reply"]


def test_drain_delivers_fresh_envelope_under_ttl(root):
    env = bot_relay.enqueue_envelope(
        root, target=_target(), message="on time",
        sender_profile="default", sender_handle="hermes",
    )
    claimed = bot_relay.claim_pending_envelopes(root)
    assert [e["id"] for e in claimed] == [env["id"]]
    # no spurious expiry reply for a delivered envelope
    base = bot_relay.relay_root(root)
    assert not (base / bot_relay.REPLIES_DIR / f"{env['id']}.json").exists()


def test_drain_ttl_zero_disables_expiry(root, monkeypatch):
    monkeypatch.setattr(bot_relay, "_envelope_ttl_seconds", lambda: 0)
    env = bot_relay.enqueue_envelope(
        root, target=_target(), message="never expires",
        sender_profile="default", sender_handle="hermes",
    )
    base = bot_relay.relay_root(root)
    out_path = base / bot_relay.OUTBOX_DIR / f"{env['id']}.json"
    env["created_at"] = int(_time2.time()) - 10 * 3600
    out_path.write_text(json.dumps(env), encoding="utf-8")
    claimed = bot_relay.claim_pending_envelopes(root)
    assert [e["id"] for e in claimed] == [env["id"]]


def test_ttl_config_read_is_lazy_and_defensive(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name.startswith("hermes_cli"):
            raise ImportError("config unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert bot_relay._envelope_ttl_seconds() == bot_relay.DEFAULT_ENVELOPE_TTL_SECONDS


def test_message_agent_surfaces_runtime_offline_refusal(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    bot_relay.write_remote_roster(home, [
        {"profile": "default", "handle": "hermes", "connection_id": "cloud-1",
         "connection_label": "Hermes Cloud", "online": False},
    ])
    monkeypatch.setattr(
        "tools.bot_mode_dm._spawn_delivery",
        lambda *a, **k: json.dumps({"status": "sent"}),
    )
    agent = _FakeAgent(home)
    out = json.loads(message_agent_tool(target="hermes", message="ping", agent=agent))
    assert out.get("reason") == "runtime_offline"
    assert "offline" in out.get("error", "")
    # fail-fast means no envelope was queued
    assert bot_relay.claim_pending_envelopes(home) == []

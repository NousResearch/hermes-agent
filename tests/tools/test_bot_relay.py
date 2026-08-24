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


def test_waiter_command_quotes_and_targets_reply_file(root):
    env = {"id": "b" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}
    cmd = bot_relay.waiter_command(root, env)
    assert ("b" * 32) in cmd and "-c" in cmd
    assert "rm -rf" not in cmd  # sanity: single quoted -c payload


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

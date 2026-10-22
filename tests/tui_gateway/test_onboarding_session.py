"""The onboarding chat in the gateway: ``onboarding.start`` / ``onboarding.state`` / ``onboarding.skip``,
the toolset fold-in that gives setup_beat to that chat only, the scripted opening, the turn note,
and the run record across a compression rotation."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import onboarding_record
from agent.onboarding_flow import End, Facts, Shown
from agent.onboarding_session import carry_onboarding_toolset, onboarding_prelude, onboarding_toolsets
from agent.onboarding_text import GREETING
from tui_gateway import server, server_requests

FACTS = Facts(os="macos", arch="arm64", machine_kind="Mac", plugins=("blender",))


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr("agent.onboarding_facts.read_facts", lambda **kw: FACTS)
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")  # a model to run on
    return home


@pytest.fixture
def client(root, monkeypatch):
    """session.resume and prompt.submit as a recorder: the RPCs onboarding.start drives. A submit
    stores its user turn, as the real turn does when it starts."""
    calls = []

    def resume(rid, params):
        calls.append(("resume", params))
        return {"result": {"session_id": f"rt-{params['session_id']}"}}

    def submit(rid, params):
        calls.append(("submit", params))
        server._get_db().append_message(params["session_id"].removeprefix("rt-"), "user", params["text"])
        return {"result": {"status": "streaming"}}

    monkeypatch.setitem(server._methods, "session.resume", resume)
    monkeypatch.setitem(server._methods, "prompt.submit", submit)
    return calls


def _call(method: str, params: dict | None = None) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": 3, "method": method, "params": params or {}})


def test_start_creates_one_onboarding_chat_and_plays_the_opening_once(client):
    started = _call("onboarding.start", {"locale": "en-IN"})["result"]
    stored = started["stored_session_id"]

    db = server._get_db()
    run_id = onboarding_record.load(db, stored)["run_id"]
    assert started == {"session_id": f"rt-{stored}", "stored_session_id": stored, "resumed": False, "run_id": run_id}
    assert onboarding_record.load(db, stored)["phase"] == "onboarding"
    assert client[0] == ("resume", {"profile": "default", "session_id": stored, "source": "desktop"})
    kind, submitted = client[1]
    assert kind == "submit" and submitted["display_kind"] == "hidden"
    assert "first-run onboarding" in submitted["text"] and "setup_beat" in submitted["text"]
    assert _call("onboarding.state")["result"]["session_id"] == stored

    # A start without fresh_launch (a second window, a reconnect) reopens the same chat without a
    # second opening; rerun starts a new one.
    again = _call("onboarding.start")["result"]
    assert again["stored_session_id"] == stored and again["resumed"] is True and len(client) == 3
    assert _call("onboarding.start", {"rerun": True})["result"]["stored_session_id"] != stored


def test_a_restart_before_the_opening_turn_plays_the_opening_on_relaunch(client, monkeypatch):
    def lost_submit(rid, params):
        client.append(("lost", params))
        return {"result": {"status": "streaming"}}

    storing_submit = server._methods["prompt.submit"]
    monkeypatch.setitem(server._methods, "prompt.submit", lost_submit)
    first = _call("onboarding.start")["result"]
    stored = first["stored_session_id"]
    monkeypatch.setitem(server._methods, "prompt.submit", storing_submit)

    reopened = _call("onboarding.start")["result"]
    assert reopened == {"session_id": f"rt-{stored}", "stored_session_id": stored, "resumed": True,
                        "run_id": first["run_id"]}
    assert [kind for kind, _ in client] == ["resume", "lost", "resume", "submit"]


def test_only_a_chat_with_a_run_in_progress_gets_the_onboarding_toolset(root):
    db = server._get_db()
    db.create_session("onb", "desktop")
    db.create_session("plain", "desktop")
    onboarding_record.start(db, "onb", FACTS)

    assert onboarding_toolsets(["web", "file"], db, "onb") == ["web", "file", "onboarding"]
    assert onboarding_toolsets(["web", "file"], db, "plain") == ["web", "file"]
    assert onboarding_toolsets([], db, "onb") == []
    # A refresh keeps what the agent was built with, and never adds it to another chat.
    built = SimpleNamespace(enabled_toolsets=["web", "onboarding"])
    assert carry_onboarding_toolset(["web"], built) == ["web", "onboarding"]
    assert carry_onboarding_toolset(["web"], SimpleNamespace(enabled_toolsets=["web"])) == ["web"]


def test_an_unrestricted_chat_with_a_run_keeps_every_tool_and_gets_setup_beat(root):
    """HERMES_TUI_TOOLSETS=all loads None, which selects no session-gated toolset (onboarding is one)."""
    from model_tools import _select_tool_names
    db = server._get_db()
    db.create_session("onb", "desktop")
    db.create_session("plain", "desktop")
    onboarding_record.start(db, "onb", FACTS)

    enabled = onboarding_toolsets(None, db, "onb")
    assert _select_tool_names(enabled, None, True) == _select_tool_names(None, None, True) | {"setup_beat"}
    # The context-engine gate at agent build reads the toolset name, as it reads None.
    assert "context_engine" in enabled
    assert onboarding_toolsets(None, db, "plain") is None
    # A refresh loads None again; the built agent keeps setup_beat.
    assert carry_onboarding_toolset(None, SimpleNamespace(enabled_toolsets=enabled)) == enabled
    assert carry_onboarding_toolset(None, SimpleNamespace(enabled_toolsets=None)) is None


def _agent(db, sid="onb"):
    return SimpleNamespace(valid_tool_names={"setup_beat", "terminal"}, _session_db=db, session_id=sid,
                           _gateway_turn_context_notes="")


def test_the_opening_plays_the_greeting_and_name_card_until_the_name_card_was_shown(root):
    db = server._get_db()
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", FACTS)
    agent = _agent(db)

    assert list(onboarding_prelude(agent, [])) == [(GREETING, "setup_beat", {"beat": "name"})]
    relaunch = [{"role": "assistant", "content": GREETING}]
    assert list(onboarding_prelude(agent, relaunch)) == [("", "setup_beat", {"beat": "name"})]
    onboarding_record.advance(db, "onb", Shown("name"))
    assert onboarding_prelude(agent, []) is None
    assert onboarding_prelude(SimpleNamespace(valid_tool_names={"terminal"}), []) is None


@pytest.mark.parametrize("locale, opens_with, asks", [
    ("de", "Hallo, ich bin Hermes.", "Wie darf ich Sie nennen?"),
    ("zh_TW", "你好，我是 Hermes。我們", "我該怎麼稱呼你？"),
    ("ja-JP", "こんにちは、Hermes です。", "なんとお呼びすればいいですか？"),
    ("tlh", GREETING, "What should I call you?"),
    (None, GREETING, "What should I call you?"),
])
def test_the_opening_greets_and_asks_the_name_in_the_run_s_locale(root, locale, opens_with, asks):
    from tools.setup_beat_tool import setup_beat_tool

    db = server._get_db()
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", Facts(os="macos", locale=locale))
    agent = _agent(db)

    [(hello, tool, args)] = list(onboarding_prelude(agent, []))
    assert hello.startswith(opens_with) and (tool, args) == ("setup_beat", {"beat": "name"})
    assert next(iter(onboarding_prelude(agent, [{"role": "assistant", "content": hello}])))[0] == ""
    shown = []
    setup_beat_tool(args, db=db, session_id="onb", callback=lambda payload: shown.append(payload) or None)
    assert shown[0]["question"] == asks


def test_the_turn_note_rides_a_user_turn_only_while_the_run_is_open(root):
    db = server._get_db()
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", FACTS)
    onboarding_record.advance(db, "onb", Shown("name"))
    agent = _agent(db)

    server._stage_onboarding_turn_note("ui", agent, history_empty=True)
    assert agent._gateway_turn_context_notes == ""
    server._stage_onboarding_turn_note("ui", agent, history_empty=False)
    assert 'setup_beat(beat="name")' in agent._gateway_turn_context_notes


def test_skip_setup_ends_the_run_and_closes_the_waiting_card(root, monkeypatch):
    db = server._get_db()
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", FACTS)
    sid, frames, events = "ui-onb", [], []
    monkeypatch.setitem(server._sessions, sid, {"session_key": "onb", "agent": _agent(db), "source": "desktop"})
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_write", frames.append)
    monkeypatch.setattr(server_requests, "_emit", lambda kind, _sid, payload: events.append((kind, payload)))
    reply = []
    waiter = threading.Thread(target=lambda: reply.append(server._setup_beat_request(
        sid, {"beat": "theme", "card": {"picker": "accent"}})))
    waiter.start()
    from tests.waiting import wait_until
    wait_until(lambda: server_requests.pending_kind(sid) == "setup_beat")

    assert _call("onboarding.skip", {"session_id": sid})["result"] == {"skipped": True}
    waiter.join(timeout=5)
    assert reply == [None]
    assert events == [("request.cancel", {"id": frames[0]["id"], "method": "setup_beat", "reason": "resolved"})]
    assert onboarding_record.load(db, "onb")["phase"] == "skipped"
    from hermes_cli.config import read_user_config_raw
    onboarding = read_user_config_raw(root / "config.yaml")["onboarding"]
    assert onboarding["run"] is False and onboarding["seen"]["profile_build_offered"] is True


def test_skip_withdraws_the_card_a_compute_host_child_waits_on(root, monkeypatch):
    """dashboard.turn_isolation: the opening ran in the child, which owns the setup_beat wait; the
    parent holds only a mirror, so a local cancel never reaches it."""
    forwarded = []

    class _Supervisor:
        def respond(self, sid, params, *, timeout=15.0):
            forwarded.append((sid, dict(params)))
            return {"type": "respond.ack", "response": {"result": {"cancelled": 1}}}

    db = server._get_db()
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", FACTS)
    sid = "ui-onb"
    monkeypatch.setitem(server._sessions, sid, {"session_key": "onb", "agent": None, "source": "desktop",
                                                "agent_ready": threading.Event(), "_compute_host_active": True,
                                                "history_lock": threading.Lock()})
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor())
    monkeypatch.setattr(server, "write_json", lambda _message: True)
    server._relay_compute_host_rpc({"jsonrpc": "2.0", "id": "srq-host", "method": "setup_beat",
                                    "params": {"session_id": sid, "beat": "theme", "card": {"picker": "accent"}}})

    assert _call("onboarding.skip", {"session_id": sid})["result"] == {"skipped": True}
    assert forwarded == [(sid, {"cancel": {"method": "setup_beat", "reason": "resolved"}})]
    assert onboarding_record.load(db, "onb")["phase"] == "skipped"


def test_a_compression_rotation_carries_the_run_record(root):
    db = server._get_db()
    db.create_session("parent", "desktop")
    onboarding_record.start(db, "parent", FACTS)
    db.patch_session_model_config("parent", {"codex_thread_id": "segment-only"})

    db.publish_compression_child(parent_session_id="parent", child_session_id="child", source="desktop",
                                 messages=[{"role": "user", "content": "summary"}],
                                 require_compression_lease=False)

    assert onboarding_record.load(db, "child") == onboarding_record.load(db, "parent")
    assert db.get_session_model_config_value("child", "codex_thread_id") is None
    # The parent keeps its active copy; the run is read at the tip, so skipping the child ends it.
    assert onboarding_record.active_session_id(db) == "child"
    # Deleting the parent keeps the tip, so the run stays in progress there.
    assert onboarding_record.delete_chat(db, "parent", lambda: db.delete_session("parent"))
    assert onboarding_record.active_session_id(db) == "child"
    onboarding_record.advance(db, "child", End())
    assert onboarding_record.active_session_id(db) is None


def test_skip_that_read_the_id_before_a_rotation_ends_the_run_at_the_tip(root, monkeypatch):
    """Compression published a child between Skip reading the agent's id and writing the record."""
    from hermes_cli.config import read_user_config_raw
    db = server._get_db()
    db.create_session("parent", "desktop")
    onboarding_record.start(db, "parent", FACTS)
    db.publish_compression_child(parent_session_id="parent", child_session_id="child", source="desktop",
                                 messages=[{"role": "user", "content": "summary"}],
                                 require_compression_lease=False)
    monkeypatch.setitem(server._sessions, "ui-onb",
                        {"session_key": "parent", "agent": _agent(db, "parent"), "source": "desktop"})

    assert _call("onboarding.skip", {"session_id": "ui-onb"})["result"] == {"skipped": True}
    assert onboarding_record.load(db, "child")["phase"] == "skipped"
    assert onboarding_record.active_session_id(db) is None
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is False


def test_a_skip_retried_after_a_failed_config_write_closes_the_install_flag(root, monkeypatch):
    from hermes_cli import onboarding_run
    from hermes_cli.config import read_user_config_raw
    db = server._get_db()
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", FACTS)
    real_set_run = onboarding_run.set_run

    def disk_full(value, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(onboarding_run, "set_run", disk_full)
    with pytest.raises(OSError):
        onboarding_record.advance(db, "onb", End())
    monkeypatch.setattr(onboarding_run, "set_run", real_set_run)

    assert onboarding_record.load(db, "onb")["phase"] == "skipped"
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is True
    onboarding_record.advance(db, "onb", End())
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is False


def test_an_old_ended_chat_never_closes_the_flag_of_a_newer_run(root):
    from hermes_cli.config import read_user_config_raw
    db = server._get_db()
    db.create_session("old", "desktop")
    onboarding_record.start(db, "old", FACTS)
    onboarding_record.advance(db, "old", End())
    db.create_session("new", "desktop")
    onboarding_record.start(db, "new", FACTS)

    onboarding_record.advance(db, "old", End())  # the old chat's agent still carries setup_beat
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is True
    assert onboarding_record.active_session_id(db) == "new"


def test_a_rerun_ends_the_run_it_replaces_so_finishing_the_new_one_closes_the_flag(client, root, monkeypatch):
    from hermes_cli.config import read_user_config_raw
    broadcast = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda event, payload: broadcast.append((event, payload)))
    old = _call("onboarding.start")["result"]
    new = _call("onboarding.start", {"rerun": True})["result"]["stored_session_id"]
    db = server._get_db()

    assert onboarding_record.load(db, old["stored_session_id"])["ended_by"] == "replaced"
    onboarding_record.advance(db, new, End())
    onboarding_record.advance(db, new, End())  # a repeat end is not a second event
    assert broadcast == [
        ("onboarding.ended", {"run_id": old["run_id"], "stored_session_id": old["stored_session_id"],
                              "outcome": "replaced"}),
        ("onboarding.ended", {"run_id": onboarding_record.load(db, new)["run_id"], "stored_session_id": new,
                              "outcome": "skipped"})]
    assert onboarding_record.active_session_id(db) is None
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is False


def test_a_rerun_never_writes_the_install_flag_off_and_heals_leftover_open_runs(client, monkeypatch):
    from hermes_cli import onboarding_run
    db = server._get_db()
    for leftover in ("crash-a", "crash-b"):  # two open runs, as a crash mid-rerun could leave
        db.create_session(leftover, "desktop")
        onboarding_record.start(db, leftover, FACTS)
    writes = []
    real_set_run = onboarding_run.set_run
    monkeypatch.setattr(onboarding_run, "set_run", lambda value, **kw: (writes.append(value), real_set_run(value, **kw)))

    phase = lambda sid: onboarding_record.load(db, sid)["phase"]
    assert _call("onboarding.start")["result"]["stored_session_id"] == "crash-b"
    assert (phase("crash-a"), phase("crash-b")) == ("skipped", "onboarding")
    new = _call("onboarding.start", {"rerun": True})["result"]["stored_session_id"]

    assert (phase("crash-b"), phase(new)) == ("skipped", "onboarding") and False not in writes


def test_a_rerun_closes_the_card_waiting_in_the_chat_it_replaces(client, monkeypatch):
    old = _call("onboarding.start")["result"]["stored_session_id"]
    db = server._get_db()
    sid, frames, events = "ui-old", [], []
    monkeypatch.setitem(server._sessions, sid, {"session_key": old, "agent": _agent(db, old), "source": "desktop"})
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_write", frames.append)
    monkeypatch.setattr(server_requests, "_emit", lambda kind, _sid, payload: events.append((kind, payload)))
    reply = []
    waiter = threading.Thread(target=lambda: reply.append(server._setup_beat_request(
        sid, {"beat": "theme", "card": {"picker": "accent"}})), daemon=True)
    waiter.start()
    from tests.waiting import wait_until
    wait_until(lambda: server_requests.pending_kind(sid) == "setup_beat")

    _call("onboarding.start", {"rerun": True})
    waiter.join(timeout=5)
    assert reply == [None] and events[0][0] == "request.cancel"


def test_deleting_the_onboarding_chat_ends_its_run_and_closes_the_flag(root):
    db = server._get_db()
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", FACTS)

    assert _call("session.delete", {"session_id": "onb"})["result"] == {"deleted": "onb"}
    assert _call("onboarding.state")["result"] == {"run": False, "eligible": False, "session_id": None, "run_id": None}


def test_the_onboarding_facts_are_read_in_the_default_profile(root, client, monkeypatch):
    from hermes_constants import get_hermes_home
    work = root / "profiles" / "work"
    (work / "skills").mkdir(parents=True)
    (work / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", str(work))
    monkeypatch.setenv("HERMES_HOME", str(work))  # a backend launched with -p work
    monkeypatch.setitem(server._methods, "prompt.submit", lambda rid, params: {"result": {}})  # launch db is work's
    homes = []
    monkeypatch.setattr("agent.onboarding_facts.read_facts", lambda **kw: (homes.append(get_hermes_home()), FACTS)[1])

    _call("onboarding.start")
    assert homes == [root]

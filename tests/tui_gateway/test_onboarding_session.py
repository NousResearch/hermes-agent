"""The onboarding chat in the gateway: ``onboarding.start`` / ``onboarding.state`` / ``onboarding.skip``,
the toolset fold-in that gives setup_beat to that chat only, the scripted opening, the turn note,
and the run record across a compression rotation."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import onboarding_record
from session_rules.onboarding_flow import End, Facts, Shown, apply, new_record
from agent.onboarding_session import (
    carry_onboarding_toolset, onboarding_chat_toolsets, onboarding_prelude, onboarding_toolsets,
)
from agent.onboarding_text import GREETING
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
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
        server._get_db().messages.append(params["session_id"].removeprefix("rt-"),
                                         NewMessage(role="user", content=params["text"]))
        return {"result": {"status": "streaming"}}

    monkeypatch.setitem(server._methods, "session.resume", resume)
    monkeypatch.setitem(server._methods, "prompt.submit", submit)
    from hermes_cli.process_identity import serving_local_desktop
    with serving_local_desktop(True):  # the local Desktop's requests: only it starts a first run
        yield calls


def _call(method: str, params: dict | None = None) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": 3, "method": method, "params": params or {}})


def _create(db, session_id: str) -> None:
    db.sessions.create(NewSession(session_id=session_id, source="desktop"))


def _rotate(db, parent: str, child: str) -> None:
    """Publish *child* as *parent*'s compression continuation, outside any compression lease."""
    db.lineage.publish_compression_child(parent, CompressionChild(session_id=child, source="desktop"),
                                         [NewMessage(role="user", content="summary")],
                                         require_compression_lease=False)


def _delete(db, session_id: str) -> bool:
    """Delete *session_id* through the store; True when it existed."""
    return bool(db.sessions.delete(session_id, media_root=str(Path(server._hermes_home) / "media")))


def test_start_creates_one_onboarding_chat_and_plays_the_opening_once(client):
    started = _call("onboarding.start", {"locale": "en-IN"})["result"]
    stored = started["stored_session_id"]

    db = server._get_db()
    run_id = onboarding_record.load(db, stored, persistence=db)["run_id"]
    assert started == {"session_id": f"rt-{stored}", "stored_session_id": stored, "resumed": False, "run_id": run_id}
    assert onboarding_record.load(db, stored, persistence=db)["phase"] == "onboarding"
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
    _create(db, "onb")
    _create(db, "plain")
    onboarding_record.start(db, "onb", FACTS, persistence=db)

    assert onboarding_toolsets(["web", "file"], db, "onb", persistence=db) == ["web", "file", "onboarding"]
    assert onboarding_toolsets(["web", "file"], db, "plain", persistence=db) == ["web", "file"]
    assert onboarding_toolsets([], db, "onb", persistence=db) == []
    # A refresh keeps what the agent was built with, and never adds it to another chat.
    built = SimpleNamespace(enabled_toolsets=["web", "onboarding"])
    assert carry_onboarding_toolset(["web"], built) == ["web", "onboarding"]
    assert carry_onboarding_toolset(["web"], SimpleNamespace(enabled_toolsets=["web"])) == ["web"]


def test_an_unrestricted_chat_with_a_run_keeps_every_tool_and_gets_setup_beat(root):
    """HERMES_TUI_TOOLSETS=all loads None, which selects no session-gated toolset (onboarding is one)."""
    from model_tools import _select_tool_names
    db = server._get_db()
    _create(db, "onb")
    _create(db, "plain")
    onboarding_record.start(db, "onb", FACTS, persistence=db)

    enabled = onboarding_toolsets(None, db, "onb", persistence=db)
    assert _select_tool_names(enabled, None, True) == _select_tool_names(None, None, True) | {"setup_beat"}
    # The context-engine gate at agent build reads the toolset name, as it reads None.
    assert "context_engine" in enabled
    assert onboarding_toolsets(None, db, "plain", persistence=db) is None
    # A refresh loads None again; the built agent keeps setup_beat.
    assert carry_onboarding_toolset(None, SimpleNamespace(enabled_toolsets=enabled)) == enabled
    assert carry_onboarding_toolset(None, SimpleNamespace(enabled_toolsets=None)) is None


def _agent(db, sid="onb"):
    return SimpleNamespace(valid_tool_names={"setup_beat", "terminal"}, _session_db=db, persistence=db,
                           session_id=sid, _gateway_turn_context_notes="")


def test_the_opening_plays_the_greeting_and_name_card_until_the_name_card_was_shown(root):
    db = server._get_db()
    _create(db, "onb")
    onboarding_record.start(db, "onb", FACTS, persistence=db)
    agent = _agent(db)

    assert list(onboarding_prelude(agent, [])) == [(GREETING, "setup_beat", {"beat": "name"})]
    relaunch = [{"role": "assistant", "content": GREETING}]
    assert list(onboarding_prelude(agent, relaunch)) == [("", "setup_beat", {"beat": "name"})]
    onboarding_record.advance(db, "onb", Shown("name"), persistence=db)
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
    _create(db, "onb")
    onboarding_record.start(db, "onb", Facts(os="macos", locale=locale), persistence=db)
    agent = _agent(db)

    [(hello, tool, args)] = list(onboarding_prelude(agent, []))
    assert hello.startswith(opens_with) and (tool, args) == ("setup_beat", {"beat": "name"})
    assert next(iter(onboarding_prelude(agent, [{"role": "assistant", "content": hello}])))[0] == ""
    shown = []
    setup_beat_tool(args, db=db, persistence=db, session_id="onb",
                    callback=lambda payload: shown.append(payload) or None)
    assert shown[0]["question"] == asks


def test_the_turn_note_rides_a_user_turn_only_while_the_run_is_open(root):
    db = server._get_db()
    _create(db, "onb")
    onboarding_record.start(db, "onb", FACTS, persistence=db)
    onboarding_record.advance(db, "onb", Shown("name"), persistence=db)
    agent = _agent(db)

    server._stage_onboarding_turn_note("ui", agent, history_empty=True)
    assert agent._gateway_turn_context_notes == ""
    server._stage_onboarding_turn_note("ui", agent, history_empty=False)
    assert 'setup_beat(beat="name")' in agent._gateway_turn_context_notes


def test_skip_setup_ends_the_run_and_closes_the_waiting_card(root, monkeypatch):
    db = server._get_db()
    _create(db, "onb")
    onboarding_record.start(db, "onb", FACTS, persistence=db)
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
    assert onboarding_record.load(db, "onb", persistence=db)["phase"] == "skipped"
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
    _create(db, "onb")
    onboarding_record.start(db, "onb", FACTS, persistence=db)
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
    assert onboarding_record.load(db, "onb", persistence=db)["phase"] == "skipped"


def test_a_compression_rotation_carries_the_run_record(root):
    db = server._get_db()
    _create(db, "parent")
    onboarding_record.start(db, "parent", FACTS, persistence=db)
    db.sessions.patch_model_config("parent", {"codex_thread_id": "segment-only"})

    _rotate(db, "parent", "child")

    assert onboarding_record.load(db, "child", persistence=db) == onboarding_record.load(db, "parent", persistence=db)
    assert db.sessions.get_model_config_value("child", "codex_thread_id") is None
    # The parent keeps its active copy; the run is read at the tip, so skipping the child ends it.
    assert onboarding_record.active_session_id(db, persistence=db) == "child"
    # Deleting the parent keeps the tip, so the run stays in progress there.
    assert onboarding_record.delete_chat(db, "parent", lambda: _delete(db, "parent"), persistence=db)
    assert onboarding_record.active_session_id(db, persistence=db) == "child"
    onboarding_record.advance(db, "child", End(), persistence=db)
    assert onboarding_record.active_session_id(db, persistence=db) is None


def test_skip_that_read_the_id_before_a_rotation_ends_the_run_at_the_tip(root, monkeypatch):
    """Compression published a child between Skip reading the agent's id and writing the record."""
    from hermes_cli.config import read_user_config_raw
    db = server._get_db()
    _create(db, "parent")
    onboarding_record.start(db, "parent", FACTS, persistence=db)
    _rotate(db, "parent", "child")
    monkeypatch.setitem(server._sessions, "ui-onb",
                        {"session_key": "parent", "agent": _agent(db, "parent"), "source": "desktop"})

    assert _call("onboarding.skip", {"session_id": "ui-onb"})["result"] == {"skipped": True}
    assert onboarding_record.load(db, "child", persistence=db)["phase"] == "skipped"
    assert onboarding_record.active_session_id(db, persistence=db) is None
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is False


def test_a_skip_retried_after_a_failed_config_write_closes_the_install_flag(root, monkeypatch):
    from hermes_cli import onboarding_run
    from hermes_cli.config import read_user_config_raw
    db = server._get_db()
    _create(db, "onb")
    onboarding_record.start(db, "onb", FACTS, persistence=db)
    real_set_run = onboarding_run.set_run

    def disk_full(value, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(onboarding_run, "set_run", disk_full)
    with pytest.raises(OSError):
        onboarding_record.advance(db, "onb", End(), persistence=db)
    monkeypatch.setattr(onboarding_run, "set_run", real_set_run)

    assert onboarding_record.load(db, "onb", persistence=db)["phase"] == "skipped"
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is True
    onboarding_record.advance(db, "onb", End(), persistence=db)
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is False


def test_an_old_ended_chat_never_closes_the_flag_of_a_newer_run(root):
    from hermes_cli.config import read_user_config_raw
    db = server._get_db()
    _create(db, "old")
    onboarding_record.start(db, "old", FACTS, persistence=db)
    onboarding_record.advance(db, "old", End(), persistence=db)
    _create(db, "new")
    onboarding_record.start(db, "new", FACTS, persistence=db)

    onboarding_record.advance(db, "old", End(), persistence=db)  # the old chat's agent still carries setup_beat
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is True
    assert onboarding_record.active_session_id(db, persistence=db) == "new"


def test_a_rerun_ends_the_run_it_replaces_so_finishing_the_new_one_closes_the_flag(client, root, monkeypatch):
    from hermes_cli.config import read_user_config_raw
    broadcast = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda event, payload: broadcast.append((event, payload)))
    old = _call("onboarding.start")["result"]
    new = _call("onboarding.start", {"rerun": True})["result"]["stored_session_id"]
    db = server._get_db()

    assert onboarding_record.load(db, old["stored_session_id"], persistence=db)["ended_by"] == "replaced"
    onboarding_record.advance(db, new, End(), persistence=db)
    onboarding_record.advance(db, new, End(), persistence=db)  # a repeat end is not a second event
    assert broadcast == [
        ("onboarding.ended", {"run_id": old["run_id"], "stored_session_id": old["stored_session_id"],
                              "outcome": "replaced"}),
        ("onboarding.ended", {"run_id": onboarding_record.load(db, new, persistence=db)["run_id"],
                              "stored_session_id": new, "outcome": "skipped"})]
    assert onboarding_record.active_session_id(db, persistence=db) is None
    assert read_user_config_raw(root / "config.yaml")["onboarding"]["run"] is False


def test_a_rerun_never_writes_the_install_flag_off_and_heals_leftover_open_runs(client, monkeypatch):
    from hermes_cli import onboarding_run
    db = server._get_db()
    for leftover in ("crash-a", "crash-b"):  # two open runs, as a crash mid-rerun could leave
        _create(db, leftover)
        onboarding_record.start(db, leftover, FACTS, persistence=db)
    writes = []
    real_set_run = onboarding_run.set_run
    monkeypatch.setattr(onboarding_run, "set_run", lambda value, **kw: (writes.append(value), real_set_run(value, **kw)))

    phase = lambda sid: onboarding_record.load(db, sid, persistence=db)["phase"]
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
    _create(db, "onb")
    onboarding_record.start(db, "onb", FACTS, persistence=db)

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


def test_a_start_outside_the_local_desktop_schedules_nothing(root):
    """A CLI or TUI calling onboarding.start is refused before ``onboarding.run`` is written."""
    from hermes_cli.onboarding_run import _explicit_run
    assert _call("onboarding.start")["error"]["code"] == 5080
    assert _explicit_run() is None


@pytest.fixture
def writer(root):
    """A writer's storage handle on the same store as ``server._get_db()``."""
    import persistence_assembly
    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=root)) as handle:
        yield handle


def _rotated_run(db, writer):
    _create(db, "parent")
    onboarding_record.start(db, "parent", FACTS, persistence=writer)
    _rotate(db, "parent", "child")


def test_a_writer_handle_reads_and_ends_the_run_at_the_compression_tip(root, writer):
    db = server._get_db()
    _rotated_run(db, writer)
    _create(db, "plain")

    assert (onboarding_record.load(db, "child", persistence=writer)
            == onboarding_record.load(db, "parent", persistence=db))
    assert onboarding_record.active_session_id(db, persistence=writer) == "child"
    assert onboarding_toolsets(["web"], db, "child", persistence=writer) == ["web", "onboarding"]
    assert onboarding_chat_toolsets(["web"], db, "parent", persistence=writer) == ["web", "onboarding"]
    assert onboarding_chat_toolsets(["web"], db, "plain", persistence=writer) == ["web"]
    # An End given the parent's id lands on the tip the chat reads.
    onboarding_record.advance(db, "parent", End(), persistence=writer)
    assert onboarding_record.load(db, "child", persistence=db)["phase"] == "skipped"
    assert onboarding_record.active_session_id(db, persistence=writer) is None


def test_a_writer_handle_delete_ends_the_run_on_every_copy(root, writer):
    db = server._get_db()
    _rotated_run(db, writer)

    # Deleting the older copy ends nothing; deleting the tip ends the copy left behind.
    assert onboarding_record.delete_chat(db, "parent", lambda: True, persistence=writer) is True
    assert onboarding_record.active_session_id(db, persistence=writer) == "child"
    assert onboarding_record.delete_chat(db, "child", lambda: _delete(db, "child"), persistence=writer)
    assert onboarding_record.load(db, "parent", persistence=db)["ended_by"] == "deleted"
    assert onboarding_record.active_session_id(db, persistence=writer) is None


def test_a_writer_handle_deletes_the_chat_and_its_files_after_the_commit(root, writer, monkeypatch):
    db = server._get_db()
    _create(db, "gone")
    sessions_dir = root / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    (sessions_dir / "gone.jsonl").write_text("{}\n", encoding="utf-8")
    removed = []
    monkeypatch.setattr("session_retention.emit_session_deleted",
                        lambda home, ids, **kw: removed.append((list(ids), kw["reason"], kw["primary"])))

    assert server._delete_unfinished_chat(db, "gone", sessions_dir=sessions_dir, media_root=str(root / "media"),
                                          persistence=writer) is True
    assert db.sessions.get("gone") is None and not (sessions_dir / "gone.jsonl").exists()
    assert removed == [(["gone"], "delete", "gone")]
    assert server._delete_unfinished_chat(db, "gone", sessions_dir=sessions_dir, media_root=str(root / "media"),
                                          persistence=writer) is False


def _fake_writer(advanced):
    sessions = MagicMock()
    sessions.advance_onboarding_run.return_value = advanced
    sessions.session_ids_with_model_config_key.return_value = ()
    return SimpleNamespace(read_only=False, sessions=sessions, lineage=MagicMock())


def test_a_writer_handle_event_is_applied_in_one_store_write(root, monkeypatch):
    from persistence.conversation.sessions import OnboardingAdvance
    record, event = new_record(FACTS, run_id="current"), End()
    applied = apply(record, event, Facts.from_stored(record["facts"]), now=5.0)
    handle = _fake_writer(OnboardingAdvance(session_id="onb", was_active=True, step=applied))
    ended = []
    monkeypatch.setattr(onboarding_record, "_ended_listeners", [ended.append])

    step = onboarding_record.advance(MagicMock(), "onb", event, persistence=handle)

    call = handle.sessions.advance_onboarding_run.call_args
    assert call.args == ("onb", event)
    assert step is applied and step.record["phase"] == "skipped"
    assert ended == [{"run_id": "current", "stored_session_id": "onb", "outcome": step.record["ended_by"]}]


def test_a_writer_handle_event_on_a_missing_row_finds_no_run(root):
    from persistence.conversation.sessions import OnboardingAdvance
    handle = _fake_writer(OnboardingAdvance(session_id=None, was_active=False, step=None))

    with pytest.raises(LookupError):
        onboarding_record.advance(MagicMock(), "onb", End(), persistence=handle)
    handle.sessions.advance_onboarding_run.assert_called_once()

from __future__ import annotations

import threading
from types import SimpleNamespace

import tui_gateway.server as server

from hermes_cli.native_turn_sources import NativeSessionView, NativeTurnAdmission, NativeTurnLease


def test_prompt_entrypoint_commits_after_slot_reservation_and_before_thread(monkeypatch):
    session = {"history_lock": threading.RLock(), "running": False}
    cleared = []
    model_threads = []

    def admit(_sid, current, _text, _images, _generation, *_display):
        current["running"] = True
        return [], object()

    monkeypatch.setattr(server, "_admit_prompt_turn", admit)
    monkeypatch.setattr(server, "_clear_inflight_turn", lambda current: cleared.append(current))
    monkeypatch.setattr(threading, "Thread", lambda *args, **kwargs: model_threads.append((args, kwargs)))

    observed = []

    def decline():
        observed.append(session["running"])
        return False

    assert server._run_prompt_submit(
        "rid", "sid", session, "native", image_paths=[], pre_model_admission=decline
    ) is False

    assert observed == [True]
    assert session["running"] is False
    assert cleared == [session]
    assert model_threads == []


def test_native_poller_uses_ordinary_prompt_path_and_preserves_turn_author(monkeypatch, tmp_path):
    sid = "ui-session"
    lease_owner = SimpleNamespace(lease_id="owner-1")
    agent = SimpleNamespace(session_id="native-session")
    session = {
        "agent": agent,
        "history_lock": threading.RLock(),
        "running": False,
        "active_session_lease": lease_owner,
        "queued_prompt": None,
        "queued_prompts": [],
        "profile_home": str(tmp_path),
        "profile_name": "default",
        "session_key": "route",
    }
    view = NativeSessionView(
        profile="default",
        profile_home=tmp_path,
        session_id="native-session",
        surface="tui",
        compression_lineage=("native-session",),
        session_key="route",
        owner_token="owner-1",
        metadata={"ui_session_id": sid, "client_surface": "desktop"},
    )
    calls = []
    admission = NativeTurnAdmission(
        NativeTurnLease(
            "lease-1",
            "continue exactly",
            lambda seen: calls.append(("commit", seen)) or True,
            lambda outcome, reason: calls.append((outcome, reason)),
            display_kind="coordination_resume",
        ),
        view,
    )

    monkeypatch.setattr(server, "_native_session_view", lambda *_args: view)
    monkeypatch.setattr(
        "hermes_cli.native_turn_sources.poll_native_turn", lambda seen: admission if seen == view else None
    )

    submitted = []

    def submit(rid, seen_sid, current, text, **kwargs):
        submitted.append((rid, seen_sid, current, text, kwargs))
        assert current["running"] is True
        assert kwargs["pre_model_admission"]() is True
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    server._sessions[sid] = session
    try:
        assert server._poll_native_turn_once(sid, session) is True
    finally:
        server._sessions.pop(sid, None)

    assert len(submitted) == 1
    assert submitted[0][3] == "continue exactly"
    assert submitted[0][4]["display_kind"] == "coordination_resume"
    assert submitted[0][4]["turn_author"] == {
        "type": "native",
        "source": "coordination_resume",
    }
    assert calls == [("commit", view)]


def test_native_poller_is_inert_for_busy_or_human_queued_session(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.native_turn_sources.poll_native_turn",
        lambda _view: (_ for _ in ()).throw(AssertionError("must not poll")),
    )
    for session in (
        {"running": True, "active_session_lease": object()},
        {"running": False, "active_session_lease": object(), "queued_prompt": "human"},
        {"running": False, "active_session_lease": None},
    ):
        sid = f"sid-{id(session)}"
        server._sessions[sid] = session
        try:
            assert server._poll_native_turn_once(sid, session) is False
        finally:
            server._sessions.pop(sid, None)


def test_human_reservation_wins_race_after_native_poll(monkeypatch, tmp_path):
    sid = "ui-race"
    owner = SimpleNamespace(lease_id="owner-race", released=False)
    session = {
        "agent": SimpleNamespace(session_id="native-race"),
        "history_lock": threading.Lock(),
        "running": False,
        "active_session_lease": owner,
        "queued_prompt": None,
        "queued_prompts": [],
        "profile_home": str(tmp_path),
        "session_key": "route-race",
    }
    view = NativeSessionView(
        profile="default",
        profile_home=tmp_path,
        session_id="native-race",
        surface="tui",
        compression_lineage=("native-race",),
        session_key="route-race",
        owner_token="owner-race",
    )
    aborted = []
    admission = NativeTurnAdmission(
        NativeTurnLease(
            "lease-race",
            "continue",
            lambda _view: True,
            lambda outcome, reason: aborted.append((outcome, reason)),
        ),
        view,
    )
    poll_entered = threading.Event()
    human_reserved = threading.Event()

    def poll(_view):
        poll_entered.set()
        assert human_reserved.wait(timeout=2)
        return admission

    monkeypatch.setattr(server, "_native_session_view", lambda *_args: view)
    monkeypatch.setattr("hermes_cli.native_turn_sources.poll_native_turn", poll)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native turn must not start")
        ),
    )
    result = []
    server._sessions[sid] = session
    thread = threading.Thread(
        target=lambda: result.append(server._poll_native_turn_once(sid, session))
    )
    try:
        thread.start()
        assert poll_entered.wait(timeout=2)
        with session["history_lock"]:
            session["running"] = True
        human_reserved.set()
        thread.join(timeout=2)
    finally:
        human_reserved.set()
        thread.join(timeout=2)
        server._sessions.pop(sid, None)

    assert not thread.is_alive()
    assert result == [False]
    assert aborted and aborted[0][0] == "not_sent"
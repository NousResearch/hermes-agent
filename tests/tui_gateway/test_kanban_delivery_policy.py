"""Desktop Kanban delivery policy through the production rebound server (#138836)."""

import contextlib
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from tui_gateway import server


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("display: {suppress_warning_notifications: false}\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_KEY", "desktop-policy-session")
    frames, submits = [], []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)

    def accept(rid, sid, session, text, **kwargs):
        submits.append((text, kwargs))
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", accept)
    session = {"session_key": "desktop-policy-session", "profile_home": str(home),
               "history_lock": threading.RLock(), "running": False}
    return session, frames, submits


def _rows(task_id):
    with contextlib.closing(kbc.connect()) as conn:
        return kbn.list_notify_subs(conn, task_id=task_id)


def _completed_subscription(session, *, profile="default", mode="notify+wake", thread_id=None):
    with contextlib.closing(kbc.connect()) as conn:
        task_id = kb.create_task(conn, title="explicit desktop delivery")
        kbn.add_notify_sub(conn, task_id=task_id, platform="tui", chat_id=session["session_key"],
                           notifier_profile=profile, delivery_mode=mode, thread_id=thread_id)
        before = kbn.list_notify_subs(conn, task_id=task_id)[0]
        assert kb.complete_task(conn, task_id, summary="explicit result")
    return task_id, before


def test_auto_subscribed_notify_is_passive(delivery):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.kanban_tools import _maybe_auto_subscribe

    session, frames, submits = delivery
    tokens = set_session_vars(platform="", chat_id="", thread_id="", profile="default")
    try:
        with contextlib.closing(kbc.connect()) as conn:
            task_id = kb.create_task(conn, title="passive desktop completion")
            assert _maybe_auto_subscribe(conn, task_id)
            before = kbn.list_notify_subs(conn, task_id=task_id)[0]
            assert before["delivery_mode"] == "notify"
            assert kb.complete_task(conn, task_id, summary="passive result")
    finally:
        clear_session_vars(tokens)

    server._notif_poll_kanban("desktop", session)

    assert [f["params"]["type"] for f in frames] == ["status.update"]
    assert "passive result" in frames[0]["params"]["payload"]["text"]
    assert submits == [], "notify must not start a model turn"
    assert not session["running"]
    assert not session.get("_kanban_pending")
    assert _rows(task_id)[0]["last_event_id"] > before["last_event_id"]
    server._notif_poll_kanban("desktop", session)
    assert len(frames) == 1
    assert submits == []


@pytest.mark.parametrize("probe_fails", [False, True])
def test_foreign_profile_cannot_claim_even_with_same_session_key(delivery, monkeypatch, probe_fails):
    from pathlib import Path

    session, frames, submits = delivery
    task_id, before = _completed_subscription(session, profile="reviewer")
    if probe_fails:
        monkeypatch.setattr(kbn, "count_notify_subs", _unavailable_probe)

    server._notif_poll_kanban("foreign", session)

    assert _rows(task_id) == [before], "foreign profile advanced the subscription cursor"
    assert frames == []
    assert submits == []
    assert not session["running"]
    owner = Path(session["profile_home"]) / "profiles" / "reviewer"
    owner.mkdir(parents=True)
    session["profile_home"] = str(owner)

    server._notif_poll_kanban("owner", session)

    assert _rows(task_id)[0]["last_event_id"] > before["last_event_id"]
    assert [f["params"]["session_id"] for f in frames] == ["owner"]
    assert len(submits) == 1
    assert "explicit result" in submits[0][0]
    assert session["running"]


@pytest.mark.parametrize("thread_id", ["topic-7", " "])
@pytest.mark.parametrize("probe_fails", [False, True])
def test_unsupported_thread_stays_unclaimed_while_root_delivers(delivery, monkeypatch, thread_id, probe_fails):
    session, frames, submits = delivery
    threaded, before = _completed_subscription(session, thread_id=thread_id)
    root, root_before = _completed_subscription(session)
    if probe_fails:
        monkeypatch.setattr(kbn, "count_notify_subs", _unavailable_probe)

    server._notif_poll_kanban("desktop", session)

    assert _rows(threaded) == [before], "root session stole an unsupported thread subscription"
    assert _rows(root)[0]["last_event_id"] > root_before["last_event_id"]
    assert len(frames) == 1
    assert root in frames[0]["params"]["payload"]["text"]
    assert len(submits) == 1
    assert root in submits[0][0]
    assert threaded not in submits[0][0]


def test_wake_only_has_no_passive_status(delivery):
    session, frames, submits = delivery
    task_id, before = _completed_subscription(session, mode="wake")

    server._notif_poll_kanban("desktop", session)

    assert frames == [], "wake-only must not publish a passive status"
    assert len(submits) == 1
    assert task_id in submits[0][0]
    assert _rows(task_id)[0]["last_event_id"] > before["last_event_id"]


def _unavailable_probe(**kwargs):
    raise OSError("read-only probe unavailable")


@pytest.mark.parametrize("profile", [None, ""])
def test_unstamped_legacy_root_remains_passive_for_exact_named_session(delivery, profile):
    from pathlib import Path

    session, frames, submits = delivery
    owner = Path(session["profile_home"]) / "profiles" / "reviewer"
    owner.mkdir(parents=True)
    session["profile_home"] = str(owner)
    task_id, before = _completed_subscription(session, profile=profile, mode="notify")

    server._notif_poll_kanban("desktop", session)

    assert len(frames) == 1
    assert task_id in frames[0]["params"]["payload"]["text"]
    assert not submits
    assert not session["running"]
    assert _rows(task_id)[0]["last_event_id"] > before["last_event_id"]


@pytest.mark.parametrize("suppress", [False, True])
def test_busy_mixed_modes_preserve_diagnostic_classification(delivery, suppress):
    from pathlib import Path
    from gateway.warning_notifications import DiagnosticText

    session, frames, submits = delivery
    owner = Path(session["profile_home"]) / "profiles" / "reviewer"
    owner.mkdir(parents=True)
    (owner / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}\n")
    session["profile_home"] = str(owner)
    session["running"] = True
    with contextlib.closing(kbc.connect()) as conn:
        diagnostic = kb.create_task(conn, title="diagnostic")
        kbn.add_notify_sub(conn, task_id=diagnostic, platform="tui", chat_id=session["session_key"],
                           notifier_profile="reviewer", delivery_mode="notify+wake")
        with kb.write_txn(conn):
            kb._append_event(conn, diagnostic, "crashed", {})
    passive, _ = _completed_subscription(session, profile="reviewer", mode="notify")
    active, _ = _completed_subscription(session, profile="reviewer")

    server._notif_poll_kanban("desktop", session)

    assert not submits
    pending = session["_kanban_pending"]
    assert len(pending) == 2
    assert isinstance(pending[0], DiagnosticText)
    assert diagnostic in pending[0]
    assert active in pending[1]
    assert not isinstance(pending[1], DiagnosticText)
    statuses = [f["params"]["payload"]["text"] for f in frames]
    assert len(statuses) == (2 if suppress else 3)
    assert any(passive in text for text in statuses)
    session["running"] = False

    server._notif_poll_kanban("desktop", session)

    assert passive not in submits[0][0]
    assert diagnostic in submits[0][0]
    if suppress:
        assert submits[0][1] == {"display_metadata": {"notification_category": "diagnostic"}}
        session["running"] = False
        server._notif_poll_kanban("desktop", session)
        assert active in submits[1][0]
        assert submits[1][1] == {}
    else:
        assert active in submits[0][0]
        assert submits[0][1] == {}
    assert not session["_kanban_pending"]
    assert len(frames) == len(statuses), "buffer flush must not repeat the passive notification"

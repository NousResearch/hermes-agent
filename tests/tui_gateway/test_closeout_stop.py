"""Stop fences task closeouts as well as the active Desktop/TUI model turn."""
from collections import deque
from types import SimpleNamespace
import threading

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ad
from tui_gateway import server


@pytest.mark.parametrize("ready", [False, True])
def test_stop_cancels_only_session_profile_and_fences_pending_closeout(tmp_path, monkeypatch, ready):
    def register():
        assert ad.register_work_group_member(
            work_id="work", owner_turn_id="owner", delegation_id="child",
            feature_config={"delegation": {"task_scoped_closeout": True}},
            routing={"origin_session": "session", "parent_session_id": "session", "origin_ui_session_id": "ui"},
            task={"goal": "required review"},
        )
        assert ad.seal_work_group("work", "owner")

    register()  # Identically named work in the launch profile must survive.
    profile_home = tmp_path / "other-profile"
    token = set_hermes_home_override(profile_home)
    try:
        register()
        pending = deque()
        claim = None
        if ready:
            ad.persist_group_member_completion("child", {"status": "completed"}, {"status": "completed"})
            claim = ad.claim_ready_work_group("work", "test")
            pending.append(server._InternalContinuation(
                "result", "work", 0, claim["envelope"]["delivery_id"], claim["claim_id"], str(profile_home),
            ))
    finally:
        reset_hermes_home_override(token)
    session = {
        "agent": SimpleNamespace(session_id="session"), "session_key": "session",
        "profile_home": str(profile_home), "history_lock": threading.Lock(),
        "running": False, "internal_continuations": pending,
    }
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _s: False)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda *_a, **_kw: None)

    server._interrupt_session_turn("ui", session)
    assert not session["internal_continuations"]
    assert not server._start_next_internal_continuation("rid", "ui", session)
    token = set_hermes_home_override(profile_home)
    try:
        with ad._transaction() as conn:
            assert conn.execute("SELECT state, terminal_disposition FROM async_delegation_work_groups").fetchone() == ("closed", "cancelled")
        if claim:
            assert not ad.bind_work_group_closeout_turn("work", claim["envelope"]["delivery_id"], claim["claim_id"], "late-turn")
    finally:
        reset_hermes_home_override(token)
    with ad._transaction() as conn:
        assert conn.execute("SELECT state FROM async_delegation_work_groups").fetchone() == ("sealed",)

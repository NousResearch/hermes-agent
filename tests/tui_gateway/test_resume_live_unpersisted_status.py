"""A live-but-unpersisted reattach reports the same liveness the registry payload does (#144728).

`_live_session_payload` carries `running` + `status`; the resume path for a live session with no
state.db row did not, and a client reads a missing key as "this gateway is too old".
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


def _ctx(sid: str) -> SimpleNamespace:
    return SimpleNamespace(rid=1, owns_db=False, db=None, cols=80, omit_messages=True,
                           defer_history=False, target="stored", profile=None,
                           profile_home=None, profile_resume_cwd=None, found={},
                           messages=lambda history: [], mint=lambda: ("unused", "tui", "."),
                           restore=lambda: ([], [], []), display_prefix=list,
                           inline_images=True)


def _resume(monkeypatch, session: dict) -> dict:
    sid = "live-unpersisted"
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {sid: Mock()})
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "current_transport", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test")
    return server._resume_live_unpersisted(_ctx(sid), sid, session)


@pytest.mark.parametrize("running, expected", [(True, "working"), (False, "idle")])
def test_a_live_reattach_reports_running_and_status(monkeypatch, running, expected):
    session = dict(transport=server._detached_ws_transport, running=running,
                   history_lock=threading.Lock(), history=[], session_key="stored",
                   agent=SimpleNamespace(model="test"), queued_prompt=None)

    payload = _resume(monkeypatch, session)["result"]

    assert payload["running"] is running
    assert payload["status"] == expected


def test_a_pending_approval_reads_as_waiting(monkeypatch):
    """`waiting` outranks `working`: the client must prompt, not show a spinner."""
    session = dict(transport=server._detached_ws_transport, running=True,
                   history_lock=threading.Lock(), history=[], session_key="stored",
                   agent=SimpleNamespace(model="test"), queued_prompt=None)
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "approval")

    payload = _resume(monkeypatch, session)["result"]

    assert payload["status"] == "waiting"

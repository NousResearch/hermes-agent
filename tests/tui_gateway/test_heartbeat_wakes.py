"""A background-process heartbeat is a wake for the model, not a message from the user.

Desktop and the TUI paint every ``prompt.submit`` row as a user bubble unless the backend types it
otherwise, and the process row on the status stack already tells the human the job is running —
so a heartbeat row is ``hidden`` and ``display.background_process_notifications: off`` mutes every
process-driven wake on these surfaces exactly as it does on the messaging gateway.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from tui_gateway import server

HEARTBEAT = {"type": "heartbeat", "session_id": "proc_hb", "seq": 2, "elapsed": 130.0, "interval": 60,
             "command": "npm test", "output": "1 failing\n"}
COMPLETION = {"type": "completion", "session_id": "proc_done", "command": "npm test", "exit_code": 1}
DELEGATION = {"type": "async_delegation", "delegation_id": "d1", "session_key": "s", "results": []}


@pytest.fixture
def surface(monkeypatch):
    submits: list = []
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda evt, consumer: "claimed")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a, **k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda rid, sid, session, text, **kw: submits.append((text, kw)))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_owns_notification_event", lambda sid, session, evt: True)
    return submits


def _session(profile_home=None) -> dict:
    return {"history_lock": threading.RLock(), "running": False, "history": [], "agent": None,
            "profile_home": str(profile_home) if profile_home else None}


def test_background_notification_policy_is_profile_scoped_a_b_a_without_secret_hydration(
    tmp_path, monkeypatch
):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for home, mode in ((home_a, "off"), (home_b, "concise")):
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({"display": {"background_process_notifications": mode}}),
            encoding="utf-8",
        )

    hydration_attempts: list = []

    def fail_external_secret_hydration(profile_home):
        hydration_attempts.append(profile_home)
        raise AssertionError("notification-policy read invoked external secret hydration")

    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", fail_external_secret_hydration)
    session_a = _session(home_a)
    session_b = _session(home_b)

    assert [
        server._background_notifications_off(session_a),
        server._background_notifications_off(session_b),
        server._background_notifications_off(session_a),
    ] == [True, False, True]
    assert hydration_attempts == []


def test_a_heartbeat_wake_is_typed_hidden(surface):
    session = _session()
    assert server._notif_claim_turn(session) is True

    server._notif_dispatch_event("sid", session, dict(HEARTBEAT), "beat text")

    ((text, kwargs),) = surface
    assert text == "beat text"
    assert kwargs["display_kind"] == "hidden"


def test_off_mutes_process_wakes_without_hydrating_secrets_but_subagent_results_still_land(
    surface, tmp_path, monkeypatch
):
    """A display-only policy read is profile-scoped without resolving external secret sources.

    A finished ``delegate_task(background=true)`` is a result the user asked for, never a
    process notification to opt out of.
    """
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"background_process_notifications": "off"}}), encoding="utf-8"
    )
    session = _session(tmp_path)
    registry = SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda session_id: False)
    completions: list = []
    hydration_attempts: list = []

    class ForbiddenSecretHydration(RuntimeError):
        pass

    def fail_external_secret_hydration(profile_home):
        hydration_attempts.append(profile_home)
        raise ForbiddenSecretHydration("display-only notification policy invoked external secret hydration")

    from hermes_cli import env_loader

    original_hydrator = env_loader.hydrate_profile_secret_sources
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", fail_external_secret_hydration)

    forbidden_errors = []
    try:
        for event in (HEARTBEAT, COMPLETION):
            try:
                assert server._notif_handle_event(
                    "sid", session, dict(event), set(), registry, lambda e: "t", completions
                ) is True
            except ForbiddenSecretHydration as exc:
                forbidden_errors.append(exc)
    finally:
        # Delegation delivery legitimately enters the provider-capable profile scope. Keep the
        # forbidden hydrator focused on the display-only policy read being tested.
        monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", original_hydrator)

    process_running = session["running"]
    process_surface = list(surface)
    assert server._notif_handle_event(
        "sid", session, dict(DELEGATION), set(), registry, lambda e: "t", completions
    ) is True
    assert [kw["display_kind"] for _, kw in surface] == ["async_delegation_complete"]

    assert forbidden_errors == [], "; ".join(str(exc) for exc in forbidden_errors)
    assert hydration_attempts == []
    assert completions == [], "a muted wake never reaches a turn"
    assert process_surface == [], "a muted wake has no prompt surface"
    assert process_running is False, "and never keeps the session claimed"

"""Compute-host ownership survives normal turns, but not a child-process crash."""

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor


def _session(**extra):
    return {
        "agent": None,
        "agent_ready": threading.Event(),
        "session_key": "stored-test-session",
        "history": [],
        "history_version": 0,
        "history_lock": threading.Lock(),
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        "created_at": time.time(),
        "last_active": time.time(),
        "_compute_host_active": True,
        **extra,
    }


@pytest.fixture
def completion_surface(monkeypatch):
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_compute_host_session_info", lambda _session: {})
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: None)


@pytest.mark.parametrize(
    "frame, owns_agent",
    [
        ({"type": "turn.end"}, True),
        ({"type": "turn.error", "reason": "model_error"}, True),
        ({"type": "turn.error", "reason": "crash"}, False),
    ],
)
def test_turn_completion_releases_ownership_only_when_the_child_crashes(
    completion_surface, frame, owns_agent
):
    session = _session(pending_model_switch={"raw": "next-model"})

    server._on_compute_host_turn_done("request", "session", session, frame)

    assert session["running"] is False
    assert bool(session.get("_compute_host_active")) is owns_agent
    assert ("pending_model_switch" in session) is (frame["type"] == "turn.error")


@pytest.fixture
def dispatch_surface(monkeypatch, completion_surface):
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    monkeypatch.setattr(server, "_compute_host_turn_frame", lambda *_a, **_k: {})


def test_fast_child_crash_is_not_overwritten_when_dispatch_returns(monkeypatch, dispatch_surface):
    session = _session(_compute_host_active=False)

    class FastCrashingSupervisor:
        def submit_turn(self, frame, *, on_complete):
            on_complete({"type": "turn.error", "reason": "crash"})

    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: FastCrashingSupervisor())

    response = server._submit_prompt_to_compute_host("request", "session", session, "hello")

    assert response["result"]["status"] == "streaming"
    assert session["running"] is False
    assert not session.get("_compute_host_active")


@pytest.mark.parametrize("alive", [False, True])
def test_idle_ownership_verdict_does_not_release_the_flag_when_the_supervisor_is_dead(
    monkeypatch, tmp_path, alive
):
    """An idle owned session is judged by the supervisor, and that read is side-effect free:
    releasing the pin belongs to the mutating model-pick route, never to a predicate that
    read-only callers (which may already hold the non-reentrant history_lock) also call."""
    session = _session(agent=object(), running=False)
    supervisor = HostSupervisor(
        registry_path=tmp_path / "host.json", expected_build_sha="test", autostart=False
    )
    supervisor._proc = SimpleNamespace(poll=lambda: None if alive else 1)
    assert supervisor.is_running() is alive
    monkeypatch.setattr(server, "_turn_isolation_enabled", lambda _cfg=None: True)
    monkeypatch.setattr(server, "_compute_host_supervisor", supervisor)

    assert server._session_uses_compute_host(session) is alive
    # Pure predicate: the ownership pin survives a read that answered "not the host any more".
    assert bool(session.get("_compute_host_active")) is True


def test_model_pick_does_not_defer_to_a_dead_host(monkeypatch, completion_surface):
    session = _session(agent=object(), running=False)
    monkeypatch.setitem(server._sessions, "stale-owner", session)
    monkeypatch.setattr(server, "_turn_isolation_enabled", lambda _cfg=None: True)
    monkeypatch.setattr(server, "_compute_host_supervisor", SimpleNamespace(is_running=lambda: False))
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _session: nullcontext())
    switches = []

    def apply_model(sid, target_session, value, **_kwargs):
        switches.append((sid, target_session, value))
        return {"value": value, "warning": "", "confirm_required": False, "confirm_message": ""}

    monkeypatch.setattr(server, "_apply_model_switch", apply_model)
    response = server.handle_request({
        "id": "pick", "method": "config.set",
        "params": {"session_id": "stale-owner", "key": "model", "value": "next-model"},
    })

    assert not response.get("error"), response
    assert not response["result"].get("deferred", False)
    assert switches == [("stale-owner", session, "next-model")]
    # The mutating route releases the stale ownership pin the predicate must not.
    assert "_compute_host_active" not in session
    assert "pending_model_switch" not in session


@pytest.mark.parametrize("was_owned", [False, True])
def test_send_failure_restores_prior_ownership_and_keeps_inline_fallback(
    monkeypatch, dispatch_surface, was_owned
):
    session = _session(_compute_host_active=was_owned, attached_images=["attachment.png"])

    class BrokenSupervisor:
        def submit_turn(self, frame, *, on_complete):
            on_complete({"type": "turn.error", "reason": "send_failed"})
            raise BrokenPipeError("broken pipe")

    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: BrokenSupervisor())

    response = server._submit_prompt_to_compute_host("request", "session", session, "hello")

    assert response["error"]["code"] == 5019
    assert bool(session.get("_compute_host_active")) is was_owned
    assert "_compute_host_turn_id" not in session
    assert session["running"] is True
    assert session["attached_images"] == ["attachment.png"]


def test_model_pick_keeps_the_existing_unowned_lazy_session_path(monkeypatch, completion_surface):
    session = _session(running=False, _compute_host_active=False)
    monkeypatch.setitem(server._sessions, "lazy-unowned", session)
    monkeypatch.setattr(server, "_turn_isolation_enabled", lambda _cfg=None: True)
    monkeypatch.setattr(server, "_compute_host_supervisor", None)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _session: nullcontext())
    monkeypatch.setattr(server, "_cfgset_await_agent", lambda *_args: None)
    calls = []

    def build_agent(sid, target_session):
        calls.append("build")
        target_session["agent"] = object()

    def apply_model(*_args, **_kwargs):
        calls.append("switch")
        return {"value": "next-model", "warning": "", "confirm_required": False, "confirm_message": ""}

    monkeypatch.setattr(server, "_start_agent_build", build_agent)
    monkeypatch.setattr(server, "_apply_model_switch", apply_model)
    response = server.handle_request({
        "id": "pick", "method": "config.set",
        "params": {"session_id": "lazy-unowned", "key": "model", "value": "next-model"},
    })

    assert not response.get("error"), response
    assert not response["result"].get("deferred", False)
    assert calls == ["build", "switch"]


def test_read_during_child_start_does_not_release_dispatch_ownership(monkeypatch, dispatch_surface):
    session = _session(agent=object(), _compute_host_active=False)
    observed = []

    class StartingSupervisor:
        alive = False

        def is_running(self):
            return self.alive

        def submit_turn(self, frame, *, on_complete):
            # A read can arrive after ownership is installed but before the
            # supervisor has started its replacement child.
            observed.append(server._session_uses_compute_host(session))
            self.alive = True

    supervisor = StartingSupervisor()
    monkeypatch.setattr(server, "_turn_isolation_enabled", lambda _cfg=None: True)
    monkeypatch.setattr(server, "_compute_host_supervisor", supervisor)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)

    response = server._submit_prompt_to_compute_host("request", "session", session, "hello")

    assert response["result"]["status"] == "streaming"
    assert session.get("_compute_host_active") is True
    assert observed == [True]


@pytest.mark.parametrize(
    "turn_state",
    [
        {"running": True},
        {"running": False, "_compute_host_turn_id": "dispatch-in-progress"},
    ],
)
def test_stale_release_revalidates_that_the_session_is_idle(monkeypatch, turn_state):
    session = _session(**turn_state)
    monkeypatch.setattr(server, "_compute_host_supervisor", SimpleNamespace(is_running=lambda: False))

    assert server._release_stale_compute_host_ownership(session) is False
    assert session.get("_compute_host_active") is True


def test_model_pick_defers_when_ownership_was_not_released_before_a_crash(
    monkeypatch, completion_surface
):
    session = _session(agent=object(), running=False)
    monkeypatch.setitem(server._sessions, "owner-dies-after-check", session)
    monkeypatch.setattr(server, "_turn_isolation_enabled", lambda _cfg=None: True)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _session: nullcontext())
    liveness_checks = []
    local_switches = []

    class DyingSupervisor:
        def is_running(self):
            liveness_checks.append(True)
            return len(liveness_checks) == 1

    def apply_model(*_args, **_kwargs):
        local_switches.append(True)
        return {"value": "next-model", "warning": "", "confirm_required": False, "confirm_message": ""}

    monkeypatch.setattr(server, "_compute_host_supervisor", DyingSupervisor())
    monkeypatch.setattr(server, "_apply_model_switch", apply_model)
    response = server.handle_request({
        "id": "pick", "method": "config.set",
        "params": {"session_id": "owner-dies-after-check", "key": "model", "value": "next-model"},
    })

    assert not response.get("error"), response
    assert response["result"].get("deferred") is True
    assert local_switches == []
    assert session.get("_compute_host_active") is True
    assert session["pending_model_switch"]["raw"] == "next-model"

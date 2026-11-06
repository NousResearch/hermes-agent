"""Real two-home compute-host ownership controls.

The registered session path records each session's authoritative ``profile_home``.  The
launch process must keep that binding separate from its one-time launch ``HERMES_HOME``:
an off-turn save for B must not be sent to a live child that was started for A, and a later
save for A must still use A.  This is a transport/control fixture, not a full profile
startup or concurrent-retirement proof; the child uses the production JSONL host loop and
only seeds a provider-free session from its selected home.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tui_gateway import server
from tui_gateway import host_supervisor as host_supervisor_module
from tui_gateway.host_supervisor import HostSupervisor


def _home(tmp_path: Path, name: str) -> Path:
    home = tmp_path / f"home-{name}"
    home.mkdir()
    (home / "profile-owner-marker.txt").write_text(name, encoding="utf-8")
    return home


def _bind_child_session(home: Path, sid: str) -> None:
    """Give the real child fixture its target session without a mutable parent env override."""
    (home / "profile-owner-session.txt").write_text(sid, encoding="utf-8")


def _saved_marker(ack: dict, home: Path) -> str:
    result = ack.get("result")
    assert isinstance(result, dict), ack
    saved = Path(str(result["file"]))
    assert saved.is_relative_to(home), (saved, home)
    payload = json.loads(saved.read_text(encoding="utf-8"))
    messages = payload.get("messages") if isinstance(payload, dict) else None
    assert isinstance(messages, list) and messages
    return str(messages[0]["content"])


@pytest.mark.platforms("posix", "windows")
def test_normal_bridge_keeps_real_profile_child_and_offturn_control_home_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A→B→A keeps the real child and registered sessions bound to their homes.

    The test intentionally uses the normal registered server method and bridge getter.  The only
    injection is the child command, passed through the real ``HostSupervisor`` constructor,
    so production hello validation, environment assembly, JSONL control, and shutdown paths
    remain in the exercise.  The old bridge sends B's control to the already-live A child;
    that child must reject B rather than silently save under A.  A later A control must still
    succeed.  A production fix that selects a child from the session's authoritative profile
    home will make the B assertion exercise a real B child with B's marker.
    """
    home_a = _home(tmp_path, "A")
    home_b = _home(tmp_path, "B")
    fixture_child = Path(__file__).with_name("profile_owner_child.py")
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(server, "_hermes_home", home_a)
    supervisors = []

    # Keep cleanup tied to the real constructors exercised by the normal bridge getter.  A future
    # profile-aware implementation may construct one supervisor per target; none may be left live.
    class _FixtureHostSupervisor(HostSupervisor):
        def __init__(self, **kwargs):
            kwargs = dict(kwargs)
            kwargs.setdefault("argv", [sys.executable, str(fixture_child)])
            kwargs.setdefault("cwd", repo)
            super().__init__(**kwargs)
            supervisors.append(self)

    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {
        "turn_isolation": True,
        "compute_host_heartbeat_secs": 1,
        "compute_host_respawn_max": 0,
    })
    monkeypatch.setattr(host_supervisor_module, "HostSupervisor", _FixtureHostSupervisor)
    profile_homes = {"profile-owner-a": home_a, "profile-owner-b": home_b}
    monkeypatch.setattr(server, "_profile_home", lambda profile: profile_homes.get(profile))
    for name in (
        "_enable_gateway_prompts", "_schedule_agent_build", "_schedule_session_cap_enforcement",
        "_register_session_cwd",
    ):
        monkeypatch.setattr(server, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(server, "_session_default_route", lambda _session: ("fixture-model", "fixture"))
    known_sessions = set(server._sessions)
    ack_b = None
    b_exception = None

    try:
        created = {}
        for profile, home in profile_homes.items():
            response = server._methods["session.create"](
                f"rid-create-{profile}", {"profile": profile, "source": "desktop", "cols": 80})
            assert "result" in response, response
            sid = response["result"]["session_id"]
            created[profile] = sid
            record = server._sessions[sid]
            assert record["profile_home"] == str(home)
            # The registered create path is lazy.  This fixture starts with an already-admitted,
            # provider-free compute-host transport record so session.save reaches the real child
            # protocol without building a model or provider.
            record["agent_ready"] = None
            record["_compute_host_active"] = True
            _bind_child_session(home, sid)

        env_before_controls = dict(os.environ)
        sid_a = created["profile-owner-a"]
        sid_b = created["profile-owner-b"]

        ack_a = server._methods["session.save"](f"rid-{sid_a}-first", {"session_id": sid_a})
        assert "error" not in ack_a, ack_a
        assert _saved_marker(ack_a, home_a) == "profile-marker:A"
        assert os.environ["HERMES_HOME"] == str(home_a)
        assert dict(os.environ) == env_before_controls

        # The current bridge has one live A child and no authoritative target selection for B.
        # Its refusal is the causal RED; the test must not switch HERMES_HOME to make B appear.
        try:
            ack_b = server._methods["session.save"](f"rid-{sid_b}", {"session_id": sid_b})
        except Exception as exc:  # preserve the real bridge failure until A is checked again
            b_exception = exc

        ack_a_again = server._methods["session.save"](f"rid-{sid_a}-again", {"session_id": sid_a})
        assert "error" not in ack_a_again, ack_a_again
        assert _saved_marker(ack_a_again, home_a) == "profile-marker:A"
        assert os.environ["HERMES_HOME"] == str(home_a)
        assert dict(os.environ) == env_before_controls
    finally:
        for supervisor in reversed(supervisors):
            supervisor.shutdown()
        with server._sessions_lock:
            for sid in [sid for sid in server._sessions if sid not in known_sessions]:
                server._sessions.pop(sid, None)

    assert b_exception is None, f"profile B control raised: {b_exception!r}"
    assert ack_b is not None, "profile B control returned no response"
    assert "error" not in ack_b, ack_b
    assert _saved_marker(ack_b, home_b) == "profile-marker:B"

"""config.get / config.set must honor params.profile (issue #95760).

A single dashboard backend in desktop app-global remote mode serves every
profile. projects.* RPCs already bind params.profile via @_profile_scoped;
config.get / config.set did not, so reads and persistent writes used the
launch profile's config.yaml for every focused profile.
"""

from __future__ import annotations

import os
from pathlib import Path

import hermes_yaml as yaml
import pytest

import tui_gateway.server as server
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


LAUNCH_CWD = "/workspace/default"
WORKER_CWD = "/workspace/code"
LAUNCH_ROOTS = ["/workspace/default"]
WORKER_ROOTS = ["/workspace/code"]


def _write_cfg(home: Path, cwd: str, roots: list[str]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "terminal": {"cwd": cwd},
                "desktop": {"repo_scan_roots": list(roots)},
                "display": {"busy_input_mode": "queue"},
            }
        ),
        encoding="utf-8",
    )


def _homes(tmp_path: Path) -> tuple[Path, Path]:
    launch = tmp_path / "launch"
    worker = tmp_path / "profiles" / "code"
    _write_cfg(launch, LAUNCH_CWD, LAUNCH_ROOTS)
    _write_cfg(worker, WORKER_CWD, WORKER_ROOTS)
    return launch, worker


def _reset_cfg_cache() -> None:
    server._cfg_cache = None
    server._cfg_sig = None
    server._cfg_path = None


def _bind_homes(monkeypatch, launch: Path, worker: Path) -> None:
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(
        server,
        "_profile_home",
        lambda name: worker if (name or "").strip() == "code" else None,
    )
    _reset_cfg_cache()


def _get(params: dict) -> dict:
    return server._methods["config.get"]("rid-get", params)


def _set(params: dict) -> dict:
    return server._methods["config.set"]("rid-set", params)


def _read_yaml(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


@pytest.fixture
def canonical_rpc_homes(tmp_path: Path, monkeypatch):
    """A launch home plus one real named profile, with live sessions for both."""
    launch = tmp_path / ".hermes"
    worker = launch / "profiles" / "code"
    _write_cfg(launch, LAUNCH_CWD, LAUNCH_ROOTS)
    _write_cfg(worker, WORKER_CWD, WORKER_ROOTS)
    os.utime(launch / "config.yaml", (1_700_000_001, 1_700_000_001))
    os.utime(worker / "config.yaml", (1_700_000_002, 1_700_000_002))

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_sessions", {
        "session-a": {"profile_home": None, "session_key": "session-a", "agent": None},
        "session-b": {"profile_home": str(worker), "session_key": "session-b", "agent": None},
    })
    from agent import secret_scope
    from tui_gateway import launch_profile_policy
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    _reset_cfg_cache()
    return launch, worker


def _dispatch_config(method: str, params: dict, rid: str = "config-rpc") -> dict:
    response = getattr(server, "dispatch")(
        {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
    )
    assert isinstance(response, dict)
    return response


def test_config_get_resolves_full_profile_and_mtime_a_b_a(canonical_rpc_homes):
    launch, worker = canonical_rpc_homes

    def read(session_id: str, profile: str | None = None) -> tuple[str, str, float]:
        scope = {"session_id": session_id, **({"profile": profile} if profile else {})}
        full = _dispatch_config("config.get", {**scope, "key": "full"})
        profile_result = _dispatch_config("config.get", {**scope, "key": "profile"})
        mtime = _dispatch_config("config.get", {**scope, "key": "mtime"})
        assert "error" not in full and "error" not in profile_result and "error" not in mtime
        return (
            full["result"]["config"]["terminal"]["cwd"],
            profile_result["result"]["home"],
            mtime["result"]["mtime"],
        )

    assert read("session-a") == (LAUNCH_CWD, str(launch), 1_700_000_001)
    assert read("session-b", "code") == (WORKER_CWD, str(worker), 1_700_000_002)
    assert read("session-a") == (LAUNCH_CWD, str(launch), 1_700_000_001)


def test_config_rpc_rejects_unresolved_or_conflicting_profile_scope(canonical_rpc_homes):
    launch, worker = canonical_rpc_homes

    # A sessionless Desktop draft remains valid when it names its profile explicitly.
    valid = _dispatch_config("config.set", {"profile": "code", "key": "busy", "value": "steer"})
    assert valid["result"]["value"] == "steer"
    assert _read_yaml(worker)["display"]["busy_input_mode"] == "steer"
    assert _dispatch_config("config.get", {"profile": "code", "key": "profile"})["result"]["home"] == str(worker)

    # Once the backend serves both homes, an unbound config RPC cannot silently select launch A.
    for method, params in (
        ("config.get", {"key": "full"}),
        ("config.get", {"session_id": "stale-session", "key": "full"}),
        ("config.set", {"key": "busy", "value": "interrupt"}),
        ("config.set", {"session_id": "stale-session", "key": "busy", "value": "interrupt"}),
    ):
        response = _dispatch_config(method, params)
        assert response.get("error", {}).get("code") == 4001, response
    assert _read_yaml(launch)["display"]["busy_input_mode"] == "queue"

    # An explicit B selector cannot override a live session owned by A.
    for method, params in (
        ("config.get", {"profile": "code", "session_id": "session-a", "key": "full"}),
        ("config.set", {"profile": "code", "session_id": "session-a", "key": "busy", "value": "interrupt"}),
    ):
        response = _dispatch_config(method, params)
        assert response.get("error", {}).get("code") == 4001, response
    assert _read_yaml(worker)["display"]["busy_input_mode"] == "steer"


def test_config_get_full_reads_params_profile_yaml_not_launch(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    _bind_homes(monkeypatch, launch, worker)

    launch_resp = _get({"key": "full"})
    assert launch_resp["result"]["config"]["terminal"]["cwd"] == LAUNCH_CWD
    assert launch_resp["result"]["config"]["desktop"]["repo_scan_roots"] == LAUNCH_ROOTS

    _reset_cfg_cache()
    worker_resp = _get({"key": "full", "profile": "code"})
    assert worker_resp["result"]["config"]["terminal"]["cwd"] == WORKER_CWD
    assert worker_resp["result"]["config"]["desktop"]["repo_scan_roots"] == WORKER_ROOTS

    # Launch file must be untouched by the focused-profile read.
    assert _read_yaml(launch)["terminal"]["cwd"] == LAUNCH_CWD


def test_config_set_persistent_write_lands_on_params_profile_yaml(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    _bind_homes(monkeypatch, launch, worker)

    resp = _set({"key": "busy", "value": "steer", "profile": "code"})
    assert resp["result"]["value"] == "steer"

    worker_cfg = _read_yaml(worker)
    launch_cfg = _read_yaml(launch)
    assert worker_cfg["display"]["busy_input_mode"] == "steer"
    assert launch_cfg["display"]["busy_input_mode"] == "queue"
    assert launch_cfg["terminal"]["cwd"] == LAUNCH_CWD
    assert worker_cfg["terminal"]["cwd"] == WORKER_CWD


def test_config_set_without_profile_still_writes_launch_home(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    _bind_homes(monkeypatch, launch, worker)

    resp = _set({"key": "busy", "value": "interrupt"})
    assert resp["result"]["value"] == "interrupt"
    assert _read_yaml(launch)["display"]["busy_input_mode"] == "interrupt"
    assert _read_yaml(worker)["display"]["busy_input_mode"] == "queue"


def test_launch_cwd_reads_launch_config_during_foreign_profile_scope(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    launch_cwd = tmp_path / "launch-workspace"
    worker_cwd = tmp_path / "worker-workspace"
    launch_cwd.mkdir()
    worker_cwd.mkdir()
    _write_cfg(launch, str(launch_cwd), LAUNCH_ROOTS)
    _write_cfg(worker, str(worker_cwd), WORKER_ROOTS)
    _bind_homes(monkeypatch, launch, worker)

    token = set_hermes_home_override(str(worker))
    try:
        assert server._launch_configured_cwd() == str(launch_cwd)
    finally:
        reset_hermes_home_override(token)


def test_profile_cwd_write_does_not_retarget_launch_session(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    launch_cwd = tmp_path / "launch-workspace"
    worker_cwd = tmp_path / "worker-workspace"
    launch_cwd.mkdir()
    worker_cwd.mkdir()
    _write_cfg(launch, ".", LAUNCH_ROOTS)
    _write_cfg(worker, str(worker_cwd), WORKER_ROOTS)
    _bind_homes(monkeypatch, launch, worker)
    monkeypatch.setenv("TERMINAL_CWD", str(launch_cwd))
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)

    response = _set(
        {"key": "terminal.cwd", "value": str(worker_cwd), "profile": "code"}
    )
    assert response["result"]["cwd"] == str(worker_cwd)
    assert _read_yaml(worker)["terminal"]["cwd"] == str(worker_cwd)
    assert server.os.environ["TERMINAL_CWD"] == str(launch_cwd)

    created = server._methods["session.create"]("rid-create", {"cols": 80})
    sid = created["result"]["session_id"]
    try:
        assert created["result"]["info"]["cwd"] == str(launch_cwd)
        assert server._sessions[sid]["profile_home"] is None
    finally:
        server._sessions.pop(sid, None)


def test_session_bound_profile_cwd_write_does_not_retarget_launch_process(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    launch_cwd = tmp_path / "launch-workspace"
    worker_cwd = tmp_path / "worker-workspace"
    launch_cwd.mkdir()
    worker_cwd.mkdir()
    _bind_homes(monkeypatch, launch, worker)
    monkeypatch.setenv("TERMINAL_CWD", str(launch_cwd))
    monkeypatch.setitem(
        server._sessions,
        "worker-session",
        {"agent": None, "profile_home": str(worker), "session_key": "worker-session"},
    )

    response = _set(
        {"session_id": "worker-session", "key": "terminal.cwd", "value": str(worker_cwd)}
    )

    assert response["result"]["cwd"] == str(worker_cwd)
    assert _read_yaml(worker)["terminal"]["cwd"] == str(worker_cwd)
    assert server.os.environ["TERMINAL_CWD"] == str(launch_cwd)


def test_launch_profile_cwd_write_updates_non_local_terminal_task(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    old_cwd = tmp_path / "old-workspace"
    new_cwd = tmp_path / "new-workspace"
    old_cwd.mkdir()
    new_cwd.mkdir()
    _bind_homes(monkeypatch, launch, worker)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CWD", str(old_cwd))

    response = _set({"key": "terminal.cwd", "value": str(new_cwd)})

    assert response["result"]["cwd"] == str(new_cwd)
    assert _read_yaml(launch)["terminal"]["cwd"] == str(new_cwd)
    assert server.os.environ["TERMINAL_CWD"] == str(new_cwd)
    assert server._terminal_task_cwd(None) == str(new_cwd)

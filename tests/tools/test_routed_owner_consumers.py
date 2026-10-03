"""A routed profile's failed backend startup must leave the launch profile's environment alone.

Two public consumers reach the environment cache by the RAW task id: the terminal tool's degraded
path (evicts on an infrastructure failure) and the image resolver's in-sandbox read. With a session
id shared by two profiles, the launch profile's healthy Docker sandbox sits under the bare id; a
routed profile whose own backend (SSH, or a Docker image that fails to start) is down must neither
read nor evict it. Reproducers adapted from the #126438 review.
"""

import asyncio
import base64
import json
from contextlib import contextmanager

import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import terminal_tool as tt
from tools import terminal_tool_backends as backends
from tools.environments.base import EnvironmentConnectionError
from tools.image_source import ResolveContext, SourceNotFound, resolve_image_source
from tools.terminal_scope import build_profile_terminal_scope, reset_terminal_scope, set_terminal_scope
from tools.terminal_tool_lifecycle import ensure_task_env, get_active_env

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


@contextmanager
def _profile(home):
    h = set_hermes_home_override(str(home))
    s = secret_scope.set_secret_scope({}, profile_home=str(home))
    t = set_terminal_scope(build_profile_terminal_scope(home))
    try:
        yield
    finally:
        reset_terminal_scope(t)
        secret_scope.reset_secret_scope(s)
        reset_hermes_home_override(h)


@pytest.mark.parametrize("case", ["image_read", "degraded_eviction"])
def test_routed_backend_failure_preserves_the_launch_owner(monkeypatch, tmp_path, case):
    raw = "same-session"
    launch, research = tmp_path / "launch", tmp_path / "profiles" / "research"
    for home in (launch, research):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "terminal:\n  backend: docker\n  container_persistent: false\n"
            "  docker_orphan_reaper: false\n  cwd: /workspace\n"
            f"  docker_image: fixture-{home.name}\n")
    if case == "degraded_eviction":
        (research / "config.yaml").write_text(
            "terminal:\n  backend: ssh\n  ssh_host: fixture.invalid\n"
            "  ssh_user: fixture\n  cwd: /workspace\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for name in ("_active_environments", "_last_activity", "_task_env_overrides",
                 "_container_aliases", "_session_cwd", "_creation_locks"):
        monkeypatch.setattr(tt, name, {})
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    events = []

    class _DockerDouble:
        def __init__(self, *, task_id, cwd, **kwargs):
            events.append(("create", task_id))
            if task_id != raw:
                raise RuntimeError("synthetic research image startup failure")
            self.cwd, self.env_type = cwd, "docker"

        def execute(self, command, **kwargs):
            events.append(("execute-launch", command))
            return {"returncode": 0, "output": base64.b64encode(_PNG).decode()}

        def cleanup(self):
            events.append(("cleanup-launch",))

    def _failing_ssh(**kwargs):
        events.append(("connect-research", kwargs["host"]))
        raise EnvironmentConnectionError("synthetic research SSH connection failure")

    monkeypatch.setattr(backends, "_DockerEnvironment", _DockerDouble)
    monkeypatch.setattr(backends, "_SSHEnvironment", _failing_ssh)

    secret_scope.set_multiplex_active(True)
    try:
        with _profile(launch):
            owner = ensure_task_env(raw)
            assert owner is not None and get_active_env(raw) is owner
        with _profile(research):
            if case == "image_read":
                with pytest.raises(SourceNotFound):
                    asyncio.run(resolve_image_source("/workspace/owner-probe.png", ResolveContext(task_id=raw)))
                assert not any(event[0] == "execute-launch" for event in events), "read the launch sandbox"
            else:
                result = json.loads(tt.terminal_tool("true", task_id=raw))
                assert result["status"] == "degraded"
                assert ("connect-research", "fixture.invalid") in events
                assert ("cleanup-launch",) not in events, "the routed failure tore down the launch sandbox"
                assert tt._active_environments.get(raw) is owner and raw in tt._last_activity
        with _profile(launch):
            assert get_active_env(raw) is owner
    finally:
        secret_scope.set_multiplex_active(False)

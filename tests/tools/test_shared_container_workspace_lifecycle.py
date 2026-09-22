"""Lifecycle-level regressions for the shared-container workspace mount.

The resolver-only matrix in ``test_docker_session_isolation.py`` proves the right
mount SOURCE is computed. It does not prove the shared runtime ADOPTS it: a
persistent Docker container is keyed per profile, so the env cached under that key
can belong to a sibling session's workspace and both reuse sites return it before
the fresh source is consulted. These tests drive the real acquisition path against
a REAL docker daemon and assert what ``/workspace`` is actually backed by.

Skipped when no docker daemon is reachable (the behavior under test is docker's).
"""

from __future__ import annotations

import os
import subprocess
import uuid

import pytest

from tools import terminal_tool
from tools import terminal_tool_lifecycle


def _docker_available() -> bool:
    try:
        return subprocess.run(
            ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=20,
        ).returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="no docker daemon")


@pytest.fixture
def shared_docker_env(tmp_path, monkeypatch):
    """A profile-scoped persistent docker config, isolated to this test's profile."""
    import shutil

    image = os.environ.get("HERMES_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("set HERMES_TEST_DOCKER_IMAGE to a locally available image")

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    # Reuse across processes is a second path into the same bug; pin it off so the
    # in-process cache is the only reuse surface under test here.
    monkeypatch.setenv("TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "false")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "work")
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    yield image
    # Remove anything this test created, by the profile label it used.
    subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=hermes-profile=work"],
        capture_output=True, text=True, timeout=60,
    )


def _make_workspace(tmp_path, name: str) -> str:
    ws = tmp_path / name
    ws.mkdir()
    (ws / "WHICH").write_text(name, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(ws)], check=True, timeout=60)
    return str(ws)


def _acquire(session_key: str, workspace: str):
    os.environ["HERMES_SESSION_KEY"] = session_key
    terminal_tool.register_task_env_overrides(
        session_key, {"cwd": workspace, "cwd_source": "session"},
    )
    return terminal_tool_lifecycle.ensure_task_env(session_key)


def _which(env) -> str:
    result = env.execute("cat /workspace/WHICH")
    out = result.get("output", "") if isinstance(result, dict) else str(result)
    return out.strip()


class TestSharedContainerAdoptsTheSessionWorkspace:
    def test_second_session_does_not_reuse_the_first_sessions_mount(
        self, shared_docker_env, tmp_path,
    ):
        """Session A creates the persistent env; B selects another workspace and must
        NOT keep executing inside A's bind."""
        ws_a = _make_workspace(tmp_path, "workspace-a")
        ws_b = _make_workspace(tmp_path, "workspace-b")

        env_a = _acquire("sess-a", ws_a)
        assert _which(env_a) == "workspace-a"

        env_b = _acquire("sess-b", ws_b)
        assert _which(env_b) == "workspace-b", (
            "session B was handed a container still mounted on session A's workspace"
        )

    def test_same_workspace_may_reuse_safely(self, shared_docker_env, tmp_path):
        """The sibling-safety rule: agreeing mounts must still reuse, or every turn
        would recreate the sandbox and lose in-container state."""
        ws = _make_workspace(tmp_path, "workspace-a")
        env_a = _acquire("sess-a", ws)
        env_c = _acquire("sess-c", ws)
        assert env_a is env_c, "an agreeing container must be reused, not recreated"

    def test_old_home_container_is_not_adopted_for_an_explicit_workspace(
        self, shared_docker_env, tmp_path,
    ):
        """The #119170 case: a pre-existing ``$HOME:/workspace`` container must not be
        handed to a session whose explicit workspace is elsewhere.

        The legacy container is simulated by caching an env under the profile key whose
        mount is the home directory (what a backend launched from ``$HOME`` produced),
        with an ``execute`` that fails loudly if it is ever used.
        """
        import pathlib

        ws = _make_workspace(tmp_path, "workspace-a")
        used = []

        class _OldHome:
            host_cwd = str(pathlib.Path.home())
            env_type = "docker"

            def execute(self, _cmd):
                used.append(_cmd)
                raise AssertionError("the stale $HOME container was reused")

        stale = _OldHome()
        terminal_tool._active_environments["profile:work"] = stale

        env = _acquire("sess-a", ws)
        assert env is not stale
        assert used == []
        assert _which(env) == "workspace-a"

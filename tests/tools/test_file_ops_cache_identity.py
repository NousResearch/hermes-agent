"""The file-ops cache must not survive the environment it wraps.

``_get_file_ops`` caches one ``ShellFileOperations`` per env-cache key. That
cache used to be validated by KEY MEMBERSHIP — ``if task_id in
_active_environments: return cached`` — which is true again as soon as a fresh
env re-occupies the key. So any path that tears an env down and creates a new one
under the SAME key handed the next file operation a handle wrapping the dead env,
and the operation died inside ``DockerEnvironment._run_bash`` on
``assert self._container_id`` -> "Container not started", while ``terminal`` kept
working because it resolves the env through ``_active_environments`` itself.

Measured shape of the failure (docker profile, shared container mode, two live
sessions disagreeing on the ``/workspace`` mount): 77 "Container not started"
results in one session, alongside 58 container release+recreate cycles in under
five minutes, the release path being the one that replaces the env without
clearing this cache.

These tests drive ``_get_file_ops`` itself — the layer that failed — rather than
the helper it consults.
"""

import pytest

from tools import file_tools, terminal_tool


class _FakeEnv:
    """Minimal terminal env: enough for ``_get_file_ops`` and identity checks."""

    def __init__(self, name: str):
        self.name = name
        self.env_type = "docker"
        self.cwd = "/workspace"

    def execute(self, _command, cwd=None, **kwargs):  # pragma: no cover - not called here
        return {"output": self.name, "returncode": 0}


@pytest.fixture(autouse=True)
def _clean_caches():
    """Isolate the env cache, the file-ops cache and the session records."""
    with terminal_tool._env_lock:
        before_envs = dict(terminal_tool._active_environments)
        before_activity = dict(terminal_tool._last_activity)
        terminal_tool._active_environments.clear()
        terminal_tool._last_activity.clear()
    with file_tools._file_ops_lock:
        before_ops = dict(file_tools._file_ops_cache)
        file_tools._file_ops_cache.clear()
    with terminal_tool._session_cwd_lock:
        before_cwd = dict(terminal_tool._session_cwd)
        terminal_tool._session_cwd.clear()
    yield
    with terminal_tool._env_lock:
        terminal_tool._active_environments.clear()
        terminal_tool._active_environments.update(before_envs)
        terminal_tool._last_activity.clear()
        terminal_tool._last_activity.update(before_activity)
    with file_tools._file_ops_lock:
        file_tools._file_ops_cache.clear()
        file_tools._file_ops_cache.update(before_ops)
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
        terminal_tool._session_cwd.update(before_cwd)


def _register(task_key: str, env: _FakeEnv) -> None:
    with terminal_tool._env_lock:
        terminal_tool._active_environments[task_key] = env


class TestFileOpsCacheIsValidatedByIdentity:
    def test_replaced_env_does_not_hand_back_the_dead_handle(self):
        """The regression: same key, new env, old handle must NOT be returned."""
        key = "default"
        dead = _FakeEnv("dead")
        _register(key, dead)
        dead_ops = file_tools._get_file_ops("default")
        assert dead_ops.env is dead

        # What a release+recreate does to the cache: the key is popped, a fresh env
        # occupies it again. The stale handle is left behind on purpose — that is the
        # state the buggy fast path accepted.
        live = _FakeEnv("live")
        _register(key, live)

        fresh_ops = file_tools._get_file_ops("default")
        assert fresh_ops.env is live, (
            "the file-ops cache handed back a handle wrapping a torn-down env: "
            "every file op would fail with 'Container not started'"
        )
        assert fresh_ops is not dead_ops

    def test_same_env_is_still_reused(self):
        """Control: an unchanged env must reuse its handle, or every file op would
        rebuild the wrapper (and lose its command-resolution caches)."""
        env = _FakeEnv("live")
        _register("default", env)
        first = file_tools._get_file_ops("default")
        second = file_tools._get_file_ops("default")
        assert first is second

    def test_env_cleaned_up_without_replacement_still_invalidates(self, monkeypatch):
        """The pre-existing path: env gone, key gone, handle dropped and rebuilt.

        The drop happens lazily inside the next ``_get_file_ops`` call, so the
        freshly created env is what the returned handle must wrap.
        """
        env = _FakeEnv("gone")
        _register("default", env)
        stale_ops = file_tools._get_file_ops("default")
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop("default", None)

        rebuilt = _FakeEnv("rebuilt")
        monkeypatch.setattr(
            file_tools, "_create_terminal_env_for_file_ops",
            lambda _raw, _key: ("docker", rebuilt),
        )
        ops = file_tools._get_file_ops("default")
        assert ops is not stale_ops
        assert ops.env is rebuilt

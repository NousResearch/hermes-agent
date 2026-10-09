"""Sandbox teardown runs in the owning profile's HERMES_HOME scope.

Teardown runs on the inactive-env reaper thread, at exit, or on eviction after an
infrastructure failure, none of which carry the owning profile's scope. Unscoped, the
persistent snapshot stores (``modal_snapshots.json``, ``vercel_sandbox_snapshots.json``,
``singularity_snapshots.json``) resolve the launch profile's home, so a secondary
profile's next session finds no snapshot to restore and starts from a fresh filesystem.
"""

import asyncio
import json
import threading
import types
from pathlib import Path

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.environments import singularity as singularity_env
from tools.environments.base import BaseEnvironment
from tools.environments.modal import ModalEnvironment
from tools.environments.vercel_sandbox import VercelSandboxEnvironment
from tools.terminal_tool_lifecycle import _cleanup_env, _evict_environment_for_task


class _RecordingEnv(BaseEnvironment):
    """Minimal backend whose cleanup() records the home it resolves."""

    def __init__(self):
        super().__init__(cwd="/", timeout=1)
        self.seen: list[Path] = []

    def cleanup(self):
        self.seen.append(get_hermes_home())


def _in_owner(home: Path, build):
    """Construct under *home*'s scope, as the terminal tool does for the owning profile."""
    token = set_hermes_home_override(home)
    try:
        return build()
    finally:
        reset_hermes_home_override(token)


def _in_fresh_thread(fn):
    """Run *fn* on a new thread, as the reaper does: no ContextVar is inherited."""
    errors = []

    def run():
        try:
            fn()
        except Exception as exc:  # surfaced to the test below
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=30)
    assert not errors, errors


def _profile_home(tmp_path: Path) -> Path:
    home = tmp_path / "profiles" / "reviewer"
    home.mkdir(parents=True)
    return home


def test_reaper_teardown_resolves_owning_profile(tmp_path):
    owner = _profile_home(tmp_path)
    env = _in_owner(owner, _RecordingEnv)

    _in_fresh_thread(lambda: _cleanup_env(env))

    assert env.seen == [owner]


def test_eviction_teardown_resolves_owning_profile(tmp_path, monkeypatch):
    from tools import terminal_tool

    owner = _profile_home(tmp_path)
    env = _in_owner(owner, _RecordingEnv)
    monkeypatch.setitem(terminal_tool._active_environments, "evict-me", env)

    _in_fresh_thread(lambda: _evict_environment_for_task("evict-me"))

    assert env.seen == [owner]


def test_teardown_scope_does_not_leak_into_the_calling_thread(tmp_path):
    owner = _profile_home(tmp_path)
    before = get_hermes_home()
    env = _in_owner(owner, _RecordingEnv)

    _cleanup_env(env)

    assert env.seen == [owner]
    assert get_hermes_home() == before


def test_env_without_owner_home_is_torn_down_unchanged():
    calls = []
    _cleanup_env(types.SimpleNamespace(cleanup=lambda: calls.append(get_hermes_home())))
    assert calls == [get_hermes_home()]


def _assert_store_in_owner(owner: Path, name: str, task_id: str, value: str):
    assert json.loads((owner / name).read_text()) == {task_id: value}
    assert not (get_hermes_home() / name).exists()


def test_singularity_snapshot_store_lands_in_owning_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(singularity_env, "_ensure_singularity_available", lambda: "/usr/bin/apptainer")
    monkeypatch.setattr(singularity_env, "_get_or_build_sif", lambda image, executable="apptainer": str(tmp_path / "image.sif"))
    monkeypatch.setattr(singularity_env, "_get_scratch_dir", lambda: tmp_path / "scratch")
    monkeypatch.setattr(singularity_env.SingularityEnvironment, "_start_instance", lambda self: None)
    monkeypatch.setattr(singularity_env.SingularityEnvironment, "init_session", lambda self: None)
    owner = _profile_home(tmp_path)
    env = _in_owner(owner, lambda: singularity_env.SingularityEnvironment(
        image="python:3.11", persistent_filesystem=True, task_id="task-1"))

    _in_fresh_thread(lambda: _cleanup_env(env))

    _assert_store_in_owner(owner, "singularity_snapshots.json", "task-1", str(env._overlay_dir))


def _bare(cls, owner: Path, **attrs):
    """An instance of *cls* without its SDK bring-up: only BaseEnvironment's
    constructor runs (under the owner's scope) plus the attributes cleanup() reads."""
    env = cls.__new__(cls)
    _in_owner(owner, lambda: BaseEnvironment.__init__(env, cwd="/", timeout=1))
    for name, value in attrs.items():
        setattr(env, name, value)
    return env


def test_modal_snapshot_store_lands_in_owning_profile(tmp_path):
    async def snapshot(**_kw):
        return types.SimpleNamespace(object_id="im-reviewer")

    async def terminate():
        return None

    owner = _profile_home(tmp_path)
    env = _bare(
        ModalEnvironment, owner, _task_id="task-1", _persistent=True, _sync_manager=None,
        _sandbox=types.SimpleNamespace(
            snapshot_filesystem=types.SimpleNamespace(aio=snapshot),
            terminate=types.SimpleNamespace(aio=terminate)),
        _worker=types.SimpleNamespace(run_coroutine=lambda coro, timeout: asyncio.run(coro), stop=lambda: None))

    _in_fresh_thread(lambda: _cleanup_env(env))

    store = json.loads((owner / "modal_snapshots.json").read_text())
    assert store["direct:task-1"] == "im-reviewer"
    assert not (get_hermes_home() / "modal_snapshots.json").exists()


def test_vercel_snapshot_store_lands_in_owning_profile(tmp_path):
    owner = _profile_home(tmp_path)
    env = _bare(
        VercelSandboxEnvironment, owner, _task_id="task-1", _persistent=True, _lock=threading.Lock(),
        _sync_manager=types.SimpleNamespace(sync_back=lambda: None),
        _sandbox=types.SimpleNamespace(
            snapshot=lambda: {"snapshot_id": "snap-reviewer"}, stop=lambda *a, **kw: None,
            client=types.SimpleNamespace(close=lambda: None)))

    _in_fresh_thread(lambda: _cleanup_env(env))

    _assert_store_in_owner(owner, "vercel_sandbox_snapshots.json", "task-1", "snap-reviewer")

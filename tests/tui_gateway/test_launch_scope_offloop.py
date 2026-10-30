"""The async launch-profile seam must not read profile files on the event loop.

``async_launch_profile_scope_if_multiplexed()`` is entered once per handoff-watcher tick, on the
gateway's event loop (``gateway/run_adapters.py::_handoff_watcher``). Behind it, the launch scope
reads ``<home>/.env`` and ``<home>/config.yaml``. On 2026-10-26 a wedged read on that path held the
loop past three 30 s liveness probes, so ``gateway.shutdown_watchdog`` dumped every thread stack and
exited the gateway with 75; the dump's main thread sat in

    run_adapters._handoff_watcher
      → launch_profile_policy.async_launch_profile_scope_if_multiplexed
        → terminal_scope.install_profile_terminal_scope
          → terminal_scope.build_profile_terminal_scope
            → secret_scope.load_env_file

The build now runs in a worker thread (``asyncio.to_thread``) and only the ``ContextVar`` writes
stay on the loop. The scope a body sees is byte-identical to the synchronous seam's.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.secret_scope import get_secret, set_multiplex_active
from hermes_constants import get_hermes_home_override
from tools import terminal_scope
from tools.terminal_scope import (
    TerminalPolicyRefusal,
    TerminalPolicyUnavailable,
    get_terminal_scope,
)
from tui_gateway import launch_profile_policy as lpp


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    """A real launch home on disk, this process multiplexing, HERMES_HOME pointed at it."""
    home = tmp_path / "hermes_home"
    (home / "profiles").mkdir(parents=True)
    (home / ".env").write_text("LAUNCH_ONLY_TOKEN=from-dotenv\n", encoding="utf-8")
    (home / "config.yaml").write_text("terminal:\n  env: local\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(lpp, "_snapshot", None)  # re-freeze the launch env for this home
    set_multiplex_active(True)
    try:
        yield home
    finally:
        set_multiplex_active(False)


def test_build_runs_in_a_worker_thread(launch_home, monkeypatch):
    """The file-reading halves must not run on the loop thread; the writes must."""
    threads: dict[str, int] = {}

    def fake_secret_scope(_home):
        threads["secret"] = threading.get_ident()
        return {"LAUNCH_ONLY_TOKEN": "from-dotenv"}

    def fake_terminal_scope(_home, *, env_overlay=None):
        threads["terminal"] = threading.get_ident()
        return {"TERMINAL_ENV": "local"}

    monkeypatch.setattr(lpp, "launch_secret_scope", fake_secret_scope)
    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", fake_terminal_scope)

    observed: dict[str, object] = {}

    async def body():
        threads["loop"] = threading.get_ident()
        async with lpp.async_launch_profile_scope_if_multiplexed():
            observed["terminal"] = dict(get_terminal_scope() or {})
            observed["home"] = get_hermes_home_override()
            observed["token"] = get_secret("LAUNCH_ONLY_TOKEN")

    asyncio.run(body())

    assert threads["secret"] != threads["loop"], "secret scope was built on the event loop"
    assert threads["terminal"] != threads["loop"], "terminal scope was built on the event loop"
    # ... while the context wiring landed in the context the body runs in.
    assert observed == {
        "terminal": {"TERMINAL_ENV": "local"},
        "home": str(launch_home),
        "token": "from-dotenv",
    }


def test_event_loop_keeps_running_while_the_build_stalls(launch_home, monkeypatch):
    """Regression: a slow (wedged) read must not stop the loop from running other tasks."""
    real_build = terminal_scope.build_profile_terminal_scope
    release = threading.Event()

    def stalled_build(*args, **kwargs):
        # Blocks the worker until the loop proves it can still schedule work. No wall-clock wait: if
        # the build ran on the loop this Event could never be set, and the wait times out.
        if not release.wait(2.0):
            raise AssertionError("event loop did not keep running while the build was in flight")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", stalled_build)

    async def body():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while not release.is_set():
                await asyncio.sleep(0)  # yield only
                ticks += 1
                if ticks >= 5:
                    release.set()

        task = asyncio.create_task(ticker())
        try:
            async with lpp.async_launch_profile_scope_if_multiplexed():
                pass
        finally:
            task.cancel()
        return ticks

    assert asyncio.run(body()) >= 5, "event loop was blocked for the whole build"


def test_scope_matches_the_synchronous_seam(launch_home):
    """Same data as ``launch_profile_runtime_scope`` — off-loop only moves WHERE it is built."""
    with lpp.launch_profile_runtime_scope(launch_home):
        sync_snapshot = (
            dict(get_terminal_scope() or {}),
            get_secret("LAUNCH_ONLY_TOKEN"),
            get_hermes_home_override(),
        )

    async def body():
        async with lpp.async_launch_profile_scope_if_multiplexed():
            return (
                dict(get_terminal_scope() or {}),
                get_secret("LAUNCH_ONLY_TOKEN"),
                get_hermes_home_override(),
            )

    assert asyncio.run(body()) == sync_snapshot


def test_unreadable_policy_still_installs_the_refusal_scope(launch_home, monkeypatch):
    """Fail-closed behaviour is unchanged: a build error installs a refusal, never raises."""

    def broken_build(*_args, **_kwargs):
        raise TerminalPolicyUnavailable("cannot read /hermes_home/.env")

    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", broken_build)

    async def body():
        async with lpp.async_launch_profile_scope_if_multiplexed():
            return get_terminal_scope()

    installed = asyncio.run(body())

    assert isinstance(installed, TerminalPolicyRefusal)
    assert "cannot read /hermes_home/.env" in installed.reason


def test_single_profile_host_binds_nothing(launch_home):
    """No multiplexing: the async seam stays a no-op scope, as before."""
    set_multiplex_active(False)

    async def body():
        async with lpp.async_launch_profile_scope_if_multiplexed():
            return get_terminal_scope(), get_hermes_home_override()

    assert asyncio.run(body()) == (None, None)


def test_scope_is_undone_after_the_body(launch_home):
    """A per-tick seam must not leak its scope into the next tick."""

    async def body():
        async with lpp.async_launch_profile_scope_if_multiplexed():
            assert get_hermes_home_override() == str(launch_home)
        return get_terminal_scope(), get_hermes_home_override()

    assert asyncio.run(body()) == (None, None)

"""Regression tests for #75675: TUI background terminal processes silently
fail to start.

``terminal(background=true)`` used to report the generic "Background process
started" success unconditionally, even when the underlying process registry
spawn immediately failed (``ProcessSession.exited is True``, e.g. a sandbox
``env.execute()`` launch error via ``spawn_via_env``). The caller had no way
to learn the process never actually ran, and — because ``spawn_via_env``
also never registered the dead-on-arrival session anywhere — a later
``process(action='list'/'poll')`` came back completely empty too.

This also covers the local-spawn analog: ``spawn_local`` always succeeds at
the ``Popen`` level (the login shell launches fine) even when the wrapped
command itself fails almost instantly, so ``terminal_tool`` gives it a brief
observation window (``process_registry.observe_local_startup``) before
reporting blanket success, and classifies the outcome instead of blindly
returning ``failed_start`` for anything that has already exited:
  - never launched at all (``completion_reason == "failed_start"``)
  - launched but failed immediately (real exit code != 0 -- a command error,
    not a launch failure)
  - launched and completed successfully before the window closed (exit code
    0 -- a real success, not a failure and not "still starting")
"""

import json
from unittest.mock import MagicMock

import tools.terminal_tool as terminal_tool_module
from tools.process_registry import ProcessSession


def _make_env_config(tmp_path, **overrides):
    config = {
        "env_type": "local",
        "timeout": 30,
        "cwd": str(tmp_path),
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _patch_common(monkeypatch, tmp_path, mock_env, **config_overrides):
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _make_env_config(tmp_path, **config_overrides),
    )
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", mock_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)


def _run_background(monkeypatch, tmp_path, *, spawn_session):
    """Local path, with ``spawn_local`` mocked to return a canned session."""
    mock_env = MagicMock()
    mock_env.env = {}
    _patch_common(monkeypatch, tmp_path, mock_env)

    from tools import process_registry as process_registry_module

    monkeypatch.setattr(
        process_registry_module.process_registry,
        "spawn_local",
        lambda **_kwargs: spawn_session,
    )

    return json.loads(
        terminal_tool_module.terminal_tool(command="azcopy sync foo bar", background=True)
    )


def _run_background_non_local(monkeypatch, tmp_path, *, spawn_session, register_finished=True):
    """Non-local path, with ``spawn_via_env`` mocked to return a canned
    session. Mirrors what the real ``spawn_via_env`` does for a
    dead-on-arrival session: register it into ``_finished`` so it stays
    discoverable via ``process(action='list'/'poll')`` after the call
    returns -- Teknium's review on PR #75753 flagged that the original test
    only exercised the local/``spawn_local`` path.
    """
    mock_env = MagicMock()
    mock_env.env = {}
    _patch_common(monkeypatch, tmp_path, mock_env, env_type="docker", docker_image="test-image")

    from tools import process_registry as process_registry_module

    def fake_spawn_via_env(**_kwargs):
        if register_finished and spawn_session.exited:
            process_registry_module.process_registry._finished[spawn_session.id] = spawn_session
        return spawn_session

    monkeypatch.setattr(
        process_registry_module.process_registry,
        "spawn_via_env",
        fake_spawn_via_env,
    )

    result = json.loads(
        terminal_tool_module.terminal_tool(command="azcopy sync foo bar", background=True)
    )
    return result, process_registry_module.process_registry


def test_background_reports_failure_instead_of_fake_success(monkeypatch, tmp_path):
    failed_session = ProcessSession(
        id="proc_deadonarrival",
        command="azcopy sync foo bar",
        started_at=0.0,
        pid=None,
        exited=True,
        exit_code=127,
        completion_reason="failed_start",
        termination_source="failed_start",
        output_buffer="bash: azcopy: command not found",
    )

    result = _run_background(monkeypatch, tmp_path, spawn_session=failed_session)

    assert result["error"], "a dead-on-arrival spawn must report an error"
    assert result["exit_code"] == 127
    assert result["output"] != "Background process started"
    assert result["status"] == "failed_start"
    assert "azcopy" in result["error"] or "command not found" in result["error"]


def test_background_still_reports_success_for_a_live_process(monkeypatch, tmp_path):
    live_session = ProcessSession(
        id="proc_alive",
        command="sleep 30",
        started_at=0.0,
        pid=4242,
        exited=False,
    )

    result = _run_background(monkeypatch, tmp_path, spawn_session=live_session)

    assert result["error"] is None
    assert result["exit_code"] == 0
    assert result["session_id"] == "proc_alive"
    assert result["pid"] == 4242
    assert result["output"] == "Background process started"


def test_background_non_local_failure_is_discoverable_via_list_and_poll(monkeypatch, tmp_path):
    """Teknium (PR #75753 review): the original test forced env_type='local'
    and only patched ``spawn_local``, never exercising ``spawn_via_env`` --
    the path actually named in the #75675 regression. This drives a non-local
    env_type through ``spawn_via_env`` and confirms the failed session
    remains observable through the public process list/poll surface
    afterward, not just in the immediate tool return value.
    """
    failed_session = ProcessSession(
        id="proc_sandbox_deadonarrival",
        command="azcopy sync foo bar",
        started_at=0.0,
        pid=None,
        exited=True,
        exit_code=-1,
        completion_reason="failed_start",
        termination_source="failed_start",
        output_buffer="Failed to start: sandbox unreachable",
    )

    result, registry = _run_background_non_local(
        monkeypatch, tmp_path, spawn_session=failed_session
    )

    assert result["error"], "a dead-on-arrival spawn must report an error"
    assert result["status"] == "failed_start"
    assert result["session_id"] == "proc_sandbox_deadonarrival"
    assert result["output"] != "Background process started"

    # Discoverable via process(action='list') ...
    listed = {s["session_id"]: s for s in registry.list_sessions()}
    assert "proc_sandbox_deadonarrival" in listed
    assert listed["proc_sandbox_deadonarrival"]["status"] == "exited"

    # ... and via process(action='poll').
    polled = registry.poll("proc_sandbox_deadonarrival")
    assert polled["status"] == "exited"
    assert polled["exit_code"] == -1
    assert polled["completion_reason"] == "failed_start"


# =========================================================================
# Real local spawns (no mocking of spawn_local/observe_local_startup) --
# exercise the actual startup-observation + classification path end to end.
# =========================================================================

def _run_real_local_background(monkeypatch, tmp_path, command, *, observe_timeout=5.0):
    """Runs terminal_tool(background=True) against a REAL local subprocess.

    ``observe_local_startup`` is NOT mocked away, but its effective timeout
    is widened for the test: the production default is deliberately brief
    (so healthy long-lived processes never feel a startup delay), which
    makes CI/dev-host shell-startup jitter (a login shell can take a few
    hundred ms end-to-end depending on host/profile) a source of flakiness
    if we rely on it. Calling the *real* implementation with a longer
    ``timeout`` keeps this an end-to-end test of the real code path while
    removing that timing sensitivity.
    """
    mock_env = MagicMock()
    mock_env.env = {}
    _patch_common(monkeypatch, tmp_path, mock_env)

    from tools import process_registry as process_registry_module

    real_observe = process_registry_module.process_registry.observe_local_startup
    monkeypatch.setattr(
        process_registry_module.process_registry,
        "observe_local_startup",
        lambda session, timeout=observe_timeout, interval=0.02: real_observe(
            session, timeout=timeout, interval=interval
        ),
    )

    result = None
    try:
        result = json.loads(
            terminal_tool_module.terminal_tool(command=command, background=True)
        )
        return result
    finally:
        # Best-effort cleanup: don't leak a real subprocess out of the test
        # (harmless no-op if the session already exited on its own).
        session_id = (result or {}).get("session_id")
        if session_id:
            process_registry_module.process_registry.kill_process(session_id)


def test_real_local_command_that_fails_immediately_is_not_reported_as_started(monkeypatch, tmp_path):
    """A real local command that exits nonzero almost instantly must be
    reported as a genuine command failure -- not a fake "Background process
    started" success, and not mislabeled "failed_start" (the shell/process
    genuinely launched; the command it ran just failed)."""
    result = _run_real_local_background(monkeypatch, tmp_path, "false")

    assert result["output"] != "Background process started"
    assert result["error"], "an immediately-failing command must report an error"
    assert result["exit_code"] not in (0, None)
    assert result["status"] != "failed_start"


def test_real_local_fast_success_is_not_reported_as_failed_start(monkeypatch, tmp_path):
    """A real local command that succeeds almost instantly (`true`) must be
    reported as a success -- never as failed_start, and never with an error."""
    result = _run_real_local_background(monkeypatch, tmp_path, "true")

    assert result["status"] != "failed_start"
    assert result["error"] is None
    assert result["exit_code"] == 0

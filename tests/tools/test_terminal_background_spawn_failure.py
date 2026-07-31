"""Regression tests for #75675: TUI background terminal processes silently
fail to start.

``terminal(background=true)`` used to report the generic "Background process
started" success unconditionally, even when the underlying process registry
spawn immediately failed (``ProcessSession.exited is True``, e.g. a sandbox
``env.execute()`` launch error via ``spawn_via_env``). The caller had no way
to learn the process never actually ran, and — because ``spawn_via_env``
also never registered the dead-on-arrival session anywhere — a later
``process(action='list'/'poll')`` came back completely empty too.
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


def _run_background(monkeypatch, tmp_path, *, spawn_session):
    mock_env = MagicMock()
    mock_env.env = {}

    monkeypatch.setattr(
        terminal_tool_module, "_get_env_config", lambda: _make_env_config(tmp_path)
    )
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", mock_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    from tools import process_registry as process_registry_module

    monkeypatch.setattr(
        process_registry_module.process_registry,
        "spawn_local",
        lambda **_kwargs: spawn_session,
    )

    return json.loads(
        terminal_tool_module.terminal_tool(command="azcopy sync foo bar", background=True)
    )


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

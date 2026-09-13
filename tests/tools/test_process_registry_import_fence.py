"""Importing a tool module must never open the canonical store.

Only the runtime that owns delegation state (gateway authority, interactive agent) may
restore durable completions, and only explicitly; an ordinary client CLI (``hermes cron
run``, ``hermes sessions list``) imports ``tools.process_registry`` transitively and must
stay a zero-writer against ``state.db``.
"""
import json
import os
import queue
import subprocess
import sys
import time

import pytest

_IMPORT_PROBE = r'''
import json, os, sys
home = os.environ["HERMES_HOME"]
connects = []
sys.addaudithook(lambda ev, a: connects.append(str(a[0])) if ev == "sqlite3.connect" and home in str(a[0]) else None)
import tools.process_registry  # noqa: F401
print(json.dumps(connects))
'''


def test_importing_process_registry_does_not_connect_to_state_db(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(HERMES_HOME=str(home), HOME=str(tmp_path), PYTHONPATH=os.getcwd())
    out = subprocess.run([sys.executable, "-c", _IMPORT_PROBE], env=env, capture_output=True, text=True,
                         timeout=120, stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
    assert not (home / "state.db").exists()


@pytest.mark.asyncio
async def test_gateway_startup_restores_pending_delegation_completions(tmp_path, monkeypatch):
    """The gateway authority is the owner that rehydrates durable completions, at startup."""
    from unittest.mock import MagicMock
    import gateway.run as gateway_run
    import tools.async_delegation as ad
    from tools.process_registry import process_registry

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("agent.shell_hooks.register_from_config", lambda *a, **k: None)
    monkeypatch.setattr(process_registry, "recover_from_checkpoint", lambda: 0)
    event = {"type": "async_delegation", "delegation_id": "d-owner", "session_key": "OLD", "origin_ui_session_id": "",
             "goal": "g", "status": "success", "summary": "S", "api_calls": 1, "duration_seconds": 1.0,
             "dispatched_at": time.time() - 2, "completed_at": time.time() - 1}
    ad._persist_dispatch({"delegation_id": "d-owner", "goal": "g", "context": None, "toolsets": None, "role": "leaf",
                          "model": "m", "session_key": "OLD", "origin_ui_session_id": "", "parent_session_id": "OLD",
                          "status": "running", "dispatched_at": time.time() - 2, "completed_at": None, "interrupt_fn": None})
    ad._persist_completion(event, {"summary": "S"})
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner._start_register_plugins_relay_hooks = MagicMock()
    runner.hooks = MagicMock()
    runner._recover_unclean_sessions = _async_return((0, 0))
    runner._consume_clean_shutdown_marker = _async_return(0)
    runner._suspend_stuck_loop_sessions = lambda: 0
    await runner._start_recover_previous_run()

    restored = process_registry.completion_queue.get_nowait()
    assert restored["delegation_id"] == "d-owner" and restored["restored"] is True


def _async_return(value):
    async def _f(*a, **k):
        return value
    return _f

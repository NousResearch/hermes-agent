"""Ink reconnect re-ensures a crashed owner, never an explicitly stopped one (W11)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_runtime
import hermes_constants


_SCRIPT = Path(__file__).resolve().parents[2] / "ui-tui" / "scripts" / "gateway_bootstrap.py"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("ink_gateway_bootstrap_recover_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("record, ensured", [
    ({"gateway_state": "running", "pid": 999999}, True),  # SIGKILL / OOM: dead process still claims live
    ({"gateway_state": "stopped", "pid": 999999}, False),  # `hermes gateway stop`
    ({"gateway_state": "running", "desired_state": "stopped"}, False),  # durable operator stop intent
    # An auto-started owner ended itself (idle exit) while the TUI was suspended: the next client
    # starts a fresh one. Its reason, not the bare `stopped`, tells it from an operator stop.
    ({"gateway_state": "stopped", "exit_reason": "auto-started gateway idle for 600s "
      "(gateway.unmanaged_idle_exit_seconds)"}, True),
    ({"gateway_state": "stopped", "exit_reason": "Gateway restart requested"}, False),
])
def test_recover_reensures_only_a_crashed_owner(monkeypatch, tmp_path, record, ensured):
    module = _load_bootstrap()
    home = (tmp_path / "home").resolve()
    home.mkdir()
    (home / "gateway_state.json").write_text(json.dumps(record))
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda h, timeout=5: SimpleNamespace(state="absent", endpoint=None, reason_code=None))
    calls = []

    def ensure(h, timeout=30, idle_exit=False):
        calls.append(Path(h))
        return SimpleNamespace(state="starting", endpoint=None, reason_code="deadline", detail=None)

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)

    with pytest.raises(RuntimeError):
        module.bootstrap(False, recover=True)
    assert calls == ([home] if ensured else [])
    # A plain reconnect (no recovery slot) stays discovery-only even for a crashed owner.
    calls.clear()
    with pytest.raises(RuntimeError, match="absent"):
        module.bootstrap(False)
    assert calls == []


def test_started_daemon_does_not_inherit_the_launch_time_kanban_board(monkeypatch, tmp_path):
    """`hermes --tui` pins HERMES_KANBAN_BOARD for its own launch; the shared daemon this bootstrap
    starts (spawn_unmanaged_gateway copies os.environ) must keep the profile's current board."""
    module = _load_bootstrap()
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "launch-board")
    seen = []

    def ensure(h, timeout=30, idle_exit=False):
        import os
        seen.append(os.environ.get("HERMES_KANBAN_BOARD"))
        return SimpleNamespace(state="starting", endpoint=None, reason_code="deadline", detail=None)

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)
    with pytest.raises(RuntimeError):
        module.bootstrap(True)
    assert seen == [None]

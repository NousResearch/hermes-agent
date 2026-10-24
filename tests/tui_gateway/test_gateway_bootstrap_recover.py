"""Ink reconnect re-ensures an owner that is gone, however it went (W11, P2).

Main's TUI owned its own sidecar and never depended on an external daemon, so an open TUI must come
back on its own after ANY end of the shared gateway: a crash, its idle exit, or a planned
`hermes gateway stop` while the TUI is open. The bounded recovery slot (gatewayClient.ts
OWNER_RECOVERY_LIMIT) is what keeps a crash-looping owner from being respawned forever."""

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


@pytest.mark.parametrize("record", [
    {"gateway_state": "running", "pid": 999999},  # SIGKILL / OOM: dead process still claims live
    {"gateway_state": "stopped", "pid": 999999},  # `hermes gateway stop` while the TUI is open
    {"gateway_state": "running", "desired_state": "stopped"},  # durable operator stop intent
    {"gateway_state": "stopped", "exit_reason": "auto-started gateway idle for 600s "
     "(gateway.unmanaged_idle_exit_seconds)"},
    {"gateway_state": "stopped", "exit_reason": "Gateway restart requested"},
    None,  # no record at all
])
def test_recover_reensures_an_owner_that_is_gone(monkeypatch, tmp_path, record):
    ensured = True
    module = _load_bootstrap()
    home = (tmp_path / "home").resolve()
    home.mkdir()
    if record is not None:
        (home / "gateway_state.json").write_text(json.dumps(record))
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda h, timeout=5: SimpleNamespace(state="absent", endpoint=None, reason_code=None, detail=None))
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


@pytest.mark.parametrize("observed", [
    SimpleNamespace(state="incompatible", endpoint=None, reason_code="runtime_protocol", detail=None),
    SimpleNamespace(state="starting", endpoint=None, reason_code=None, detail=None),
])
def test_recover_reensures_any_unready_owner(monkeypatch, tmp_path, observed):
    """A reconnect that finds an old (incompatible) owner, or one still starting, ensures like the
    first launch does instead of reporting the same verdict every 30 s."""
    module = _load_bootstrap()
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint", lambda h, timeout=5: observed)
    calls = []
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda h, timeout=30, idle_exit=False: (
        calls.append(h), SimpleNamespace(state="starting", endpoint=None, reason_code="deadline", detail=None))[1])
    with pytest.raises(RuntimeError):
        module.bootstrap(False, recover=True)
    assert calls == [home]


def test_failure_names_the_reason_and_its_remedy(monkeypatch, tmp_path):
    """The TUI echoes this text; a refusal's detail (what to do) must travel with the verdict."""
    module = _load_bootstrap()
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda h, timeout=30, idle_exit=False:
                        SimpleNamespace(state="incompatible", endpoint=None, reason_code="runtime_protocol",
                                        detail="the gateway belongs to another Hermes install; stop it there"))
    with pytest.raises(RuntimeError) as err:
        module.bootstrap(True)
    assert str(err.value).splitlines() == [
        "gateway incompatible: runtime_protocol",
        "gateway incompatible: the gateway belongs to another Hermes install; stop it there"]


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


@pytest.mark.parametrize("start,recover", [(True, False), (False, True)])
def test_restart_wait_owner_is_attachable_on_start_and_recover(monkeypatch, tmp_path, start, recover):
    """GW-6: an owner in a restart's after-turn wait (draining/restart_wait) still owns its turns;
    reopening the TUI to answer a parked prompt must attach, not refuse or re-ensure it."""
    import io
    import os
    module = _load_bootstrap()
    home = (tmp_path / "home").resolve()
    home.mkdir()
    endpoint = SimpleNamespace(profile_id=str(home), instance_id="draining-owner", api_origin="http://127.0.0.1:1",
                               runtime_protocol=1, control_home=str(home))
    receipt = SimpleNamespace(state="draining", endpoint=endpoint, reason_code="restart_wait", detail=None)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint", lambda h, timeout=5: receipt)
    ensured = []
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime",
                        lambda h, timeout=30, idle_exit=False: (ensured.append(h), receipt)[1])
    reply = json.dumps({"ok": True, "id": 1, "result": {
        "instance_id": endpoint.instance_id, "profile_id": endpoint.profile_id, "ticket": "grant"}}).encode()
    if os.name == "nt":
        import gateway.runtime_bootstrap_windows as windows_bootstrap
        monkeypatch.setattr(windows_bootstrap, "query_runtime_control", lambda h, req, timeout: reply)
    else:
        import hermes_cli.gateway_runtime_discovery as discovery

        class Peer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def sendall(self, _request):
                pass

            def makefile(self, _mode):
                return io.BytesIO(reply)

        monkeypatch.setattr(discovery, "connect_private", lambda h, timeout: Peer())

    result = module.bootstrap(start, recover=recover)

    assert result["instance_id"] == "draining-owner"
    assert ensured == ([home] if start else [])

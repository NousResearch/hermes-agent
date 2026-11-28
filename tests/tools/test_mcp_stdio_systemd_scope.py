"""Systemd transient-scope isolation for stdio MCP servers (regression for #159117).

Covers:
- Pure isolation-policy decisions across gateway main process, managed/cron/Kanban
  workers in the gateway tree, and interactive CLI terminals (host-independent).
- Per-server ``memory_max_mb`` parsing (top-level and nested ``lifecycle``), validation,
  and clamping against true ancestor slice/container ``memory.max`` (never sibling units).
- ``_run_stdio`` integration: ``_resolved_identity`` and ``_preflight_stdio_command`` run
  on the raw command/env before ``systemd-run --user --scope`` wraps ``StdioServerParameters``,
  graceful/cancelled teardowns stop the transient scope after child teardown, and ``atexit``
  cleans scopes in worker subprocesses that exit without calling ``shutdown_mcp_servers()``.
- Live Linux cgroup-v2 + ``systemd-run --user --scope`` isolation and ``setsid()``
  grandchild cleanup on normal worker process exit when a user systemd bus is available.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import pytest


class TestMCPSystemdScopePolicy:
    """Pure policy & config helpers run on every host without faking OS globals."""

    def test_isolates_gateway_main_process_and_gateway_tree_workers(self):
        from tools.mcp_tool_common import should_isolate_mcp_stdio_in_scope

        # 1. Supervised gateway main process itself.
        assert should_isolate_mcp_stdio_in_scope(
            is_linux=True,
            environ={"INVOCATION_ID": "abc123", "_HERMES_GATEWAY": "1"},
            is_supervised_gateway=True,
        ) is True

        # 2. Managed turn worker, cron worker, or Kanban worker in the gateway tree
        # (is_supervised_gateway is False because getpid() != gateway.pid).
        assert should_isolate_mcp_stdio_in_scope(
            is_linux=True,
            environ={"INVOCATION_ID": "abc123", "_HERMES_GATEWAY": "1"},
            is_supervised_gateway=False,
        ) is True
        assert should_isolate_mcp_stdio_in_scope(
            is_linux=True,
            environ={"INVOCATION_ID": "abc123", "HERMES_SUPERVISED_CHILD": "1"},
            is_supervised_gateway=False,
        ) is True

    def test_skips_non_linux_or_missing_systemd_or_interactive_terminal(self):
        from tools.mcp_tool_common import should_isolate_mcp_stdio_in_scope

        # Off Linux: never isolate.
        assert should_isolate_mcp_stdio_in_scope(
            is_linux=False,
            environ={"INVOCATION_ID": "abc123", "_HERMES_GATEWAY": "1"},
            is_supervised_gateway=True,
        ) is False

        # Not under systemd: never isolate.
        assert should_isolate_mcp_stdio_in_scope(
            is_linux=True,
            environ={"_HERMES_GATEWAY": "1"},
            is_supervised_gateway=True,
        ) is False

        # Interactive `hermes chat` inside GNOME Terminal inherits INVOCATION_ID from the
        # terminal's user service, but carries no Hermes gateway tree markers.
        assert should_isolate_mcp_stdio_in_scope(
            is_linux=True,
            environ={"INVOCATION_ID": "gnome-term-id"},
            is_supervised_gateway=False,
        ) is False

    def test_parses_and_clamps_per_server_memory_max_mb(self, caplog):
        from tools.mcp_tool_common import (
            _get_lifecycle_memory_max_mb,
            resolve_mcp_memory_max_bytes,
        )

        assert _get_lifecycle_memory_max_mb({}) is None
        assert _get_lifecycle_memory_max_mb({"memory_max_mb": 0}) is None
        assert _get_lifecycle_memory_max_mb({"memory_max_mb": 512}) == 512
        assert _get_lifecycle_memory_max_mb({"lifecycle": {"memory_max_mb": 256}}) == 256
        # Top-level wins when both are set.
        assert _get_lifecycle_memory_max_mb(
            {"memory_max_mb": 384, "lifecycle": {"memory_max_mb": 256}}
        ) == 384

        # Invalid values (including fractional MiB and booleans) warn and fall back to None.
        for bad in (-10, 32, 64.5, "not-a-number", True):
            caplog.clear()
            assert _get_lifecycle_memory_max_mb({"memory_max_mb": bad}) is None
            assert caplog.records

        # Clamped only by a finite enclosing slice/container memory.max.
        assert resolve_mcp_memory_max_bytes(None, 2 * 1024**3) is None
        assert resolve_mcp_memory_max_bytes(512, 2 * 1024**3) == 512 * 1024 * 1024
        assert resolve_mcp_memory_max_bytes(6144, None) == 6144 * 1024 * 1024
        assert resolve_mcp_memory_max_bytes(6144, 2 * 1024**3) == 2 * 1024**3

    def test_ancestor_cgroup_memory_max_ignores_sibling_unit_limits(self, tmp_path):
        from tools.mcp_tool_common import _ancestor_cgroup_memory_max_bytes

        def write_max(rel: str, value: str) -> None:
            target = tmp_path / rel if rel else tmp_path
            target.mkdir(parents=True, exist_ok=True)
            (target / "memory.max").write_text(value + "\n", encoding="utf-8")

        write_max("", "max")
        write_max("user.slice/user-1000.slice", str(2 * 1024**3))
        write_max("user.slice/user-1000.slice/user@1000.service/app.slice", "max")
        # Tight limits on sibling .service / .scope units must NOT clamp hermes-mcp-*.scope.
        write_max(
            "user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service",
            str(512 * 1024**2),
        )
        write_max(
            "user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-cron-1.scope",
            str(1024**3),
        )

        gw_cg = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service/payload\n"
        cron_cg = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-worker-cron-1.scope\n"
        assert _ancestor_cgroup_memory_max_bytes(gw_cg, tmp_path) == 2 * 1024**3
        assert _ancestor_cgroup_memory_max_bytes(cron_cg, tmp_path) == 2 * 1024**3

        # Sibling slices (background.slice) and nested sub-slices under app.slice must not clamp
        # systemd-run --user --scope units created directly under app.slice.
        write_max("user.slice/user-1000.slice/user@1000.service/background.slice", str(256 * 1024**2))
        write_max("user.slice/user-1000.slice/user@1000.service/app.slice/app-custom.slice", str(384 * 1024**2))
        bg_cg = "0::/user.slice/user-1000.slice/user@1000.service/background.slice/hermes-gateway.service\n"
        sub_cg = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-custom.slice/hermes-gateway.service\n"
        assert _ancestor_cgroup_memory_max_bytes(bg_cg, tmp_path) == 2 * 1024**3
        assert _ancestor_cgroup_memory_max_bytes(sub_cg, tmp_path) == 2 * 1024**3

        # System service with lingering user bus: system.slice does not enclose user.slice scopes.
        write_max("system.slice", str(512 * 1024**2))
        write_max("system.slice/hermes-gateway.service", str(256 * 1024**2))
        assert _ancestor_cgroup_memory_max_bytes("0::/system.slice/hermes-gateway.service\n", tmp_path) is None

        # Container root cgroup limit IS honored everywhere.
        write_max("", str(1536 * 1024**2))
        assert _ancestor_cgroup_memory_max_bytes("0::/\n", tmp_path) == 1536 * 1024**2
        assert _ancestor_cgroup_memory_max_bytes("0::/system.slice/hermes-gateway.service\n", tmp_path) == 1536 * 1024**2

    def test_sanitizes_unit_name_component(self):
        from tools.mcp_tool_common import _sanitize_mcp_unit_component

        assert _sanitize_mcp_unit_component("@playwright/mcp@latest") == "playwright-mcp-latest"
        assert _sanitize_mcp_unit_component("...") == "server"
        long_name = "a" * 80
        sanitized = _sanitize_mcp_unit_component(long_name)
        assert len(sanitized) <= 40
        assert sanitized == "a" * len(sanitized)


class TestMCPTransportSystemdScopeContract:
    """Contract tests on _prepare_stdio_systemd_scope, _run_stdio, and orphan/atexit reaping."""

    def test_prepare_scope_wraps_argv_and_injects_user_bus_env(self):
        import tools.process_registry as pr
        from tools.mcp_tool_transport import _prepare_stdio_systemd_scope

        scoped_cmd, scoped_args, spawn_env, scope_unit = _prepare_stdio_systemd_scope(
            "agentmail-mcp",
            "/usr/bin/node",
            ["/opt/agentmail/index.js"],
            {"PATH": "/usr/bin", "HERMES_HOME": "/tmp/h"},
            {"memory_max_mb": 256},
            is_linux=True,
            environ={"INVOCATION_ID": "unit-1", "_HERMES_GATEWAY": "1"},
            is_supervised_gateway=False,
            scope_available=lambda: True,
            enclosing_memory_max=lambda: 2 * 1024**3,
            build_scope_argv=lambda argv, suffix, *, unit_prefix, memory_max_bytes: pr._systemd_scope_argv(
                "/usr/bin/systemd-run",
                f"{unit_prefix}-{suffix}",
                *argv,
                memory_max_bytes=memory_max_bytes,
            ),
            user_bus_env=lambda env: {
                **env,
                "XDG_RUNTIME_DIR": "/run/user/1000",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
            },
        )

        assert scoped_cmd == "/usr/bin/systemd-run"
        assert "--scope" in scoped_args
        assert "--collect" in scoped_args
        assert f"MemoryMax={256 * 1024 * 1024}" in scoped_args
        sep = scoped_args.index("--")
        assert scoped_args[sep + 1:] == ["/usr/bin/node", "/opt/agentmail/index.js"]
        assert scope_unit is not None
        assert scope_unit.startswith("hermes-mcp-agentmail-mcp-")
        assert scope_unit.endswith(".scope")
        assert spawn_env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
        assert spawn_env["HERMES_HOME"] == "/tmp/h"

    def test_run_stdio_preserves_identity_and_preflight_then_stops_scope_on_clean_exit(self, monkeypatch):
        from tools import mcp_tool as _core
        from tools import mcp_tool_lifecycle as _lifecycle
        from tools import mcp_tool_transport as _transport
        from tools.mcp_tool import MCPServerTask

        preflight_seen = []
        captured_params = []
        events = []

        async def fake_preflight(server_name, command, args):
            preflight_seen.append((server_name, command, list(args)))
            return "/cached/bin/agentmail-mcp", ["--stdio"]

        def fake_prepare(server_name, command, args, safe_env, config):
            # Must receive the post-preflight command, never run before OSV/cached-npx.
            assert command == "/cached/bin/agentmail-mcp"
            assert args == ["--stdio"]
            return (
                "/usr/bin/systemd-run",
                ["--user", "--scope", "--unit", "hermes-mcp-agentmail-test", "--", command, *args],
                {**safe_env, "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
                "hermes-mcp-agentmail-test.scope",
            )

        @asynccontextmanager
        async def fake_stdio_client(server_params, errlog=None):
            captured_params.append(server_params)
            events.append("stdio_enter")
            try:
                yield ("r", "w")
            finally:
                events.append("stdio_exit")

        @asynccontextmanager
        async def fake_client_session(*_a, **_kw):
            yield SimpleNamespace()

        async def fake_serve_session(self, session, connect_timeout, mark_lifecycle=False):
            events.append("serve")
            return "recycle"

        def fake_stop_unit(unit_name):
            events.append(("stop_unit", unit_name))
            return True

        monkeypatch.setattr(_core, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(_core, "_ensure_mcp_sdk", lambda: True)
        monkeypatch.setattr(_core, "StdioServerParameters", lambda **kw: SimpleNamespace(**kw), raising=False)
        monkeypatch.setattr(_core, "stdio_client", fake_stdio_client, raising=False)
        monkeypatch.setattr(_core, "ClientSession", fake_client_session)
        monkeypatch.setattr(_core, "_preflight_stdio_command", fake_preflight)
        monkeypatch.setattr(_transport, "_prepare_stdio_systemd_scope", fake_prepare)
        monkeypatch.setattr("tools.process_registry._stop_systemd_unit", fake_stop_unit)
        monkeypatch.setattr(_lifecycle, "_kill_orphaned_mcp_children", lambda **_kw: None)
        monkeypatch.setattr(_lifecycle, "_snapshot_child_pids", lambda: set())
        monkeypatch.setattr(_transport._config, "_managed_launcher", lambda *_a: ("/managed/bin/npx", ["/managed/bin"]))
        monkeypatch.setattr("tools.hosted_tool_policy.require_host_feature", lambda *_a, **_kw: None)

        cfg = {"command": "npx", "args": ["-y", "agentmail-mcp"], "memory_max_mb": 256}
        baseline_inputs, _ = _transport._connect_inputs("agentmail", cfg)
        expected_digest = _transport._registration._identity_digest(baseline_inputs)

        server = MCPServerTask("agentmail")
        monkeypatch.setattr(MCPServerTask, "_serve_session", fake_serve_session)

        outcome = asyncio.run(server._run_stdio(cfg))
        assert outcome == "recycle"
        assert server._resolved_identity == expected_digest
        assert preflight_seen == [("agentmail", baseline_inputs[0], ["-y", "agentmail-mcp"])]
        assert len(captured_params) == 1
        assert captured_params[0].command == "/usr/bin/systemd-run"
        assert captured_params[0].env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
        # Parent stdio_client exits BEFORE the systemd unit is stopped.
        assert events == ["stdio_enter", "serve", "stdio_exit", ("stop_unit", "hermes-mcp-agentmail-test.scope")]
        assert "hermes-mcp-agentmail-test.scope" not in _lifecycle._stdio_units
        assert "hermes-mcp-agentmail-test.scope" not in _lifecycle._orphan_stdio_units

    def test_aborted_run_stdio_preserves_lost_bus_cache_and_reaper_retries(self, monkeypatch):
        from tools import mcp_tool as _core
        from tools import mcp_tool_lifecycle as _lifecycle
        from tools import mcp_tool_transport as _transport
        from tools import process_registry as _pr
        from tools.mcp_tool import MCPServerTask

        unit = "hermes-mcp-crashed-test.scope"

        async def fake_preflight(_name, command, args):
            return command, list(args)

        def fake_prepare(_name, command, args, safe_env, _cfg):
            return (
                "/usr/bin/systemd-run",
                ["--", command, *args],
                {**safe_env, "DBUS_SESSION_BUS_ADDRESS": "unix:path=/nonexistent/bus"},
                unit,
            )

        @asynccontextmanager
        async def failing_stdio_client(_params, errlog=None):
            raise RuntimeError("handshake boom")
            yield  # pragma: no cover

        monkeypatch.setattr(_core, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(_core, "_ensure_mcp_sdk", lambda: True)
        monkeypatch.setattr(_core, "StdioServerParameters", lambda **kw: SimpleNamespace(**kw), raising=False)
        monkeypatch.setattr(_core, "stdio_client", failing_stdio_client, raising=False)
        monkeypatch.setattr(_core, "_preflight_stdio_command", fake_preflight)
        monkeypatch.setattr(_transport, "_prepare_stdio_systemd_scope", fake_prepare)
        monkeypatch.setattr(_pr, "systemd_user_bus_env", lambda env=None: dict(env or {}))
        monkeypatch.setattr(_pr, "_SYSTEMD_SCOPE_AVAILABLE", True)
        monkeypatch.setattr(_lifecycle, "_snapshot_child_pids", lambda: set())
        monkeypatch.setattr("tools.hosted_tool_policy.require_host_feature", lambda *_a, **_kw: None)

        _lifecycle._stdio_units.clear()
        _lifecycle._orphan_stdio_units.clear()

        server = MCPServerTask("crashed")
        # Simulate an exception-triggered reconnect after a prior healthy session:
        # _on_transport_error leaves _ready set while run()'s finally cleared self.session.
        server._ever_connected = True
        server._ready.set()
        with patch.object(_lifecycle, "_kill_orphaned_mcp_children"):
            with pytest.raises(RuntimeError, match="handshake boom"):
                asyncio.run(server._run_stdio({"command": "/bin/echo", "args": []}))

        # Negative cache set by scoped_spawn_lost_user_bus (#110803) must flip to False even on reconnect.
        assert _pr._SYSTEMD_SCOPE_AVAILABLE is False
        assert unit not in _lifecycle._stdio_units
        assert _lifecycle._orphan_stdio_units.get(unit) == "crashed"

        # First sweep fails to stop the scope -> kept for retry; second sweep succeeds -> cleared.
        attempts = []

        def flaky_stop(u):
            attempts.append(u)
            return len(attempts) >= 2

        monkeypatch.setattr(_pr, "_stop_systemd_unit", flaky_stop)
        _lifecycle._kill_orphaned_mcp_children(server_name="crashed")
        assert attempts == [unit]
        assert _lifecycle._orphan_stdio_units.get(unit) == "crashed"

        _lifecycle._kill_orphaned_mcp_children(server_name="crashed")
        assert attempts == [unit, unit]
        assert unit not in _lifecycle._orphan_stdio_units

    def test_cancel_before_stdio_client_leaves_no_ghost_scope_unit(self, monkeypatch):
        from tools import mcp_tool as _core
        from tools import mcp_tool_lifecycle as _lifecycle
        from tools import mcp_tool_transport as _transport
        from tools.mcp_tool import MCPServerTask

        unit = "hermes-mcp-unspawned-test.scope"

        async def fake_preflight(_name, command, args):
            return command, list(args)

        def fake_prepare(_name, command, args, safe_env, _cfg):
            return "/usr/bin/systemd-run", ["--", command, *args], safe_env, unit

        def cancelling_reaper(**_kw):
            raise asyncio.CancelledError("cancelled during pre-spawn orphan sweep")

        monkeypatch.setattr(_core, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(_core, "_ensure_mcp_sdk", lambda: True)
        monkeypatch.setattr(_core, "StdioServerParameters", lambda **kw: SimpleNamespace(**kw), raising=False)
        monkeypatch.setattr(_core, "_preflight_stdio_command", fake_preflight)
        monkeypatch.setattr(_transport, "_prepare_stdio_systemd_scope", fake_prepare)
        monkeypatch.setattr(_lifecycle, "_kill_orphaned_mcp_children", cancelling_reaper)
        monkeypatch.setattr("tools.hosted_tool_policy.require_host_feature", lambda *_a, **_kw: None)

        _lifecycle._stdio_units.clear()
        _lifecycle._orphan_stdio_units.clear()

        server = MCPServerTask("unspawned")
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(server._run_stdio({"command": "/bin/echo", "args": []}))

        assert unit not in _lifecycle._stdio_units
        assert unit not in _lifecycle._orphan_stdio_units

    def test_atexit_reaper_cleans_active_units_when_worker_skips_explicit_shutdown(self, monkeypatch):
        import atexit
        from tools import mcp_tool_lifecycle as _lifecycle

        registered = []
        stopped = []
        shutdown_calls = []

        monkeypatch.setattr(_lifecycle, "_atexit_units_registered", False)
        monkeypatch.setattr(atexit, "register", lambda fn: registered.append(fn))
        monkeypatch.setattr(
            _lifecycle, "shutdown_mcp_servers", lambda **kw: shutdown_calls.append(kw),
        )
        monkeypatch.setattr("tools.process_registry._stop_systemd_unit", lambda u: (stopped.append(u), True)[1])

        _lifecycle._stdio_units.clear()
        _lifecycle._orphan_stdio_units.clear()
        _lifecycle._track_stdio_unit("hermes-mcp-worker-1.scope", "worker_srv")
        _lifecycle._track_stdio_unit("hermes-mcp-worker-2.scope", "worker_srv")
        assert registered == [_lifecycle._atexit_reap_mcp_units]

        # Simulate shutdown_mcp_servers timing out without clearing _stdio_units: fallback sweep stops them.
        _lifecycle._atexit_reap_mcp_units()
        assert shutdown_calls == [{"timeout": 3.0}]
        assert set(stopped) == {"hermes-mcp-worker-1.scope", "hermes-mcp-worker-2.scope"}
        assert not _lifecycle._stdio_units
        assert not _lifecycle._orphan_stdio_units


@pytest.mark.platforms("linux")
def test_live_systemd_service_isolates_mcp_child_and_reaps_setsid_grandchild(tmp_path):
    """End-to-end on a Linux systemd host: inside a transient .service unit, a managed-worker-like
    child places its stdio MCP server into a distinct ``hermes-mcp-*.scope`` cgroup with
    ``MemoryMax`` enforced, and exiting the worker normally (via ``atexit``, without calling
    ``shutdown_mcp_servers()`` explicitly) reaps a ``setsid()``-escaped grandchild in that scope."""
    from gateway.status import _pid_exists
    from tools.process_registry import _systemd_run_user_scope_available, systemd_user_bus_env

    if not _systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope unavailable on this Linux host")

    server_py = tmp_path / "live_mcp_server.py"
    grandchild_pid_file = tmp_path / "grandchild.pid"
    server_py.write_text(
        "\n".join([
            "import json, os, subprocess, sys",
            "from pathlib import Path",
            "for raw in sys.stdin:",
            "    raw = raw.strip()",
            "    if not raw:",
            "        continue",
            "    msg = json.loads(raw)",
            "    method = msg.get('method')",
            "    req_id = msg.get('id')",
            "    if method == 'initialize':",
            "        resp = {'jsonrpc': '2.0', 'id': req_id, 'result': {",
            "            'protocolVersion': msg.get('params', {}).get('protocolVersion', '2025-03-26'),",
            "            'capabilities': {'tools': {}},",
            "            'serverInfo': {'name': 'liveprobe', 'version': '0.1'},",
            "        }}",
            "    elif method == 'tools/list':",
            "        resp = {'jsonrpc': '2.0', 'id': req_id, 'result': {'tools': [{",
            "            'name': 'inspect_scope',",
            "            'description': 'Return cgroup and memory.max',",
            "            'inputSchema': {'type': 'object', 'properties': {'pid_out': {'type': 'string'}}, 'required': ['pid_out']},",
            "        }]}}",
            "    elif method == 'tools/call':",
            "        pid_out = msg['params']['arguments']['pid_out']",
            "        cg = Path('/proc/self/cgroup').read_text(encoding='utf-8').strip()",
            "        rel = next(l.partition('::')[2] for l in cg.splitlines() if l.startswith('0::'))",
            "        mem_max = (Path('/sys/fs/cgroup') / rel.lstrip('/') / 'memory.max').read_text(encoding='utf-8').strip()",
            "        gc = subprocess.Popen([sys.executable, '-c', 'import os, signal; os.setsid(); signal.pause()'])",
            "        Path(pid_out).write_text(str(gc.pid), encoding='utf-8')",
            "        payload = json.dumps({'cgroup': rel, 'memory_max': mem_max, 'grandchild_pid': gc.pid})",
            "        resp = {'jsonrpc': '2.0', 'id': req_id, 'result': {'content': [{'type': 'text', 'text': payload}], 'isError': False}}",
            "    elif req_id is not None:",
            "        resp = {'jsonrpc': '2.0', 'id': req_id, 'result': {}}",
            "    else:",
            "        continue",
            "    sys.stdout.write(json.dumps(resp) + '\\n')",
            "    sys.stdout.flush()",
        ]),
        encoding="utf-8",
    )

    repo_root = str(Path(__file__).resolve().parents[2])
    driver_py = tmp_path / "service_driver.py"
    out_json = tmp_path / "result.json"
    driver_py.write_text(
        "\n".join([
            "import json, os, sys",
            "from pathlib import Path",
            f"sys.path.insert(0, {repo_root!r})",
            "from tools.mcp_tool_discovery import register_mcp_servers",
            "from tools.mcp_tool_schema import mcp_prefixed_tool_name",
            "from tools.registry import registry",
            f"home = Path({str(tmp_path / 'home')!r})",
            "home.mkdir(parents=True, exist_ok=True)",
            "os.environ['HERMES_HOME'] = str(home)",
            "# Simulate agent.managed_worker: pid != gateway.pid, exits via SystemExit without calling shutdown_mcp_servers().",
            "os.environ['_HERMES_GATEWAY'] = '1'",
            "os.environ['HERMES_SUPERVISED_CHILD'] = '1'",
            f"srv_cfg = {{'command': sys.executable, 'args': [{str(server_py)!r}], 'memory_max_mb': 64}}",
            "(home / 'config.yaml').write_text(json.dumps({'mcp_servers': {'liveprobe': srv_cfg}}), encoding='utf-8')",
            "service_cg = Path('/proc/self/cgroup').read_text(encoding='utf-8').strip()",
            "register_mcp_servers({'liveprobe': srv_cfg})",
            f"raw = registry.dispatch(mcp_prefixed_tool_name('liveprobe', 'inspect_scope'), {{'pid_out': {str(grandchild_pid_file)!r}}})",
            "outer = json.loads(raw)",
            "assert 'result' in outer, outer",
            "payload = json.loads(outer['result'])",
            f"Path({str(out_json)!r}).write_text(json.dumps({{",
            "    'service_cgroup': service_cg,",
            "    'mcp': payload,",
            "}), encoding='utf-8')",
            "raise SystemExit(0)",
        ]),
        encoding="utf-8",
    )

    unit_name = f"hermes-test-gw-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    env = systemd_user_bus_env()
    proc = subprocess.run(
        ["systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet", "--unit", unit_name, "--", sys.executable, str(driver_py)],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    try:
        assert proc.returncode == 0, f"driver failed (rc={proc.returncode}): stdout={proc.stdout}\nstderr={proc.stderr}"
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert f"{unit_name}.service" in data["service_cgroup"]
        mcp_cg = data["mcp"]["cgroup"]
        assert mcp_cg.endswith(".scope") and "/hermes-mcp-liveprobe-" in mcp_cg, data
        assert f"{unit_name}.service" not in mcp_cg
        assert data["mcp"]["memory_max"] == str(64 * 1024 * 1024)
        gc_pid = int(data["mcp"]["grandchild_pid"])
        assert _pid_exists(gc_pid) is False
    finally:
        if grandchild_pid_file.exists():
            try:
                os.kill(int(grandchild_pid_file.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass

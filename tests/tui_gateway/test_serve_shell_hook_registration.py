"""The serve backend must arm config-declared shell hooks at startup.

``tui_gateway/entry.py::main`` is the stdio JSON-RPC backend that the TUI, the Desktop app
and the dashboard's PTY chat all spawn (``ui-tui/src/gatewayClient.ts`` runs
``python -m tui_gateway.entry``).  Entries under ``hooks:`` in ``config.yaml`` are
consent-gated policy: ``agent.shell_hooks.register_from_config`` is the only thing that wires
them onto the plugin manager, and every guard site (the ``pre_tool_call`` dispatch in
``hermes_cli/plugins.py`` that ``tools/file_tools_write_guards.py`` gates on) is inert until
that call happens in the process serving the turn.

The classic CLI arms them in ``hermes_cli.main._prepare_agent_startup`` and the messaging
gateway in ``gateway/run_startup``.  ``_prepare_agent_startup`` returns immediately unless
``args.command in _AGENT_COMMANDS`` (``{None, "chat", "acp", "rl"}``) — and ``serve`` is not in
that set, so ``hermes serve`` never reaches the hooks block, and nothing on the serve startup
path registered them either.

These tests drive the REAL ``entry.main()`` startup (no registration helper is called
directly) against the per-test ``HERMES_HOME`` from ``tests/conftest.py``, with only the
non-hook startup side effects (DB rows, timers, provider prewarm) neutralised — registration,
consent, the hook subprocess and the public ``pre_tool_call`` dispatch are the real
implementation.
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

_BLOCK_REASON = "serve-startup-guard"


def _write_hook_config(home: Path, *, auto_accept: bool) -> None:
    """Declare one ``pre_tool_call`` guard for ``write_file`` in ``home/config.yaml``."""
    script = home / "guard.py"
    script.write_text(
        "import json\n"
        f"print(json.dumps({{'decision': 'block', 'reason': {_BLOCK_REASON!r}}}))\n",
        encoding="utf-8",
    )
    quoted = (
        subprocess.list2cmdline([sys.executable, str(script)])
        if os.name == "nt"
        else shlex.join([sys.executable, str(script)])
    )
    cfg = {
        "hooks": {
            "pre_tool_call": [{"command": quoted, "matcher": "write_file", "fail_closed": True}]
        },
        "toolsets": ["file"],
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
    }
    if auto_accept:
        cfg["hooks_auto_accept"] = True
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _run_serve_startup(monkeypatch) -> None:
    """Run the real ``entry.main()`` startup with a client whose stdin is already closed.

    Only fixtures that are not what these tests assert on are substituted: the DB-backed
    heartbeat row / orphan sweep, the skin watcher, the MCP discovery scan and the provider
    prewarm thread must not touch the host from a unit test.  Hook registration, consent and
    the guard dispatch stay real.
    """
    from tui_gateway import entry, server

    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", False)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.prewarm_picker_cache_async", lambda: None
    )
    # A client that opened the stream and closed it: the ready frame goes nowhere (write_json
    # is stubbed to succeed so a captured stdout cannot turn into a spurious sys.exit) and the
    # read loop sees EOF on its very first readline, exactly as the backend exits in production.
    monkeypatch.setattr(entry, "write_json", lambda payload: True)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    entry.main()


def test_serve_startup_arms_declared_shell_hook(tmp_path, monkeypatch):
    """Startup alone must wire an approved ``hooks:`` entry; the guard must then block."""
    from agent import shell_hooks
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    _write_hook_config(home, auto_accept=True)
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    plugins._reset_plugin_managers_for_tests()
    shell_hooks.reset_for_tests()
    try:
        _run_serve_startup(monkeypatch)

        # Nothing built an agent on this path: whatever the guard does now came from startup.
        blocked = plugins.get_pre_tool_call_block_message(
            "write_file", {"path": str(home / "protected"), "content": "x"}
        )
        assert blocked == _BLOCK_REASON
        # ...and the declared matcher still scopes it to write_file.
        assert (
            plugins.get_pre_tool_call_block_message(
                "read_file", {"path": str(home / "protected")}
            )
            is None
        )
    finally:
        plugins._reset_plugin_managers_for_tests()
        shell_hooks.reset_for_tests()


def test_serve_startup_leaves_an_unapproved_hook_unarmed(tmp_path, monkeypatch, caplog):
    """This backend has no TTY: an unapproved hook must stay unarmed, never faked live.

    The skip must also be visible — a hook that silently does nothing is the failure mode
    this whole test file exists to prevent.
    """
    import logging

    from agent import shell_hooks
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    _write_hook_config(home, auto_accept=False)
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    plugins._reset_plugin_managers_for_tests()
    shell_hooks.reset_for_tests()
    caplog.set_level(logging.WARNING, logger="agent.shell_hooks")
    try:
        _run_serve_startup(monkeypatch)
        assert (
            plugins.get_pre_tool_call_block_message(
                "write_file", {"path": str(home / "protected"), "content": "x"}
            )
            is None
        )
        skips = [r.getMessage() for r in caplog.records if "not allowlisted" in r.getMessage()]
        assert skips, "an unapproved hook must be reported, not silently dropped"
    finally:
        plugins._reset_plugin_managers_for_tests()
        shell_hooks.reset_for_tests()

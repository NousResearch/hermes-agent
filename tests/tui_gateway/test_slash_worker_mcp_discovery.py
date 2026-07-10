"""Regression tests for MCP discovery in tui_gateway.slash_worker (#61891).

The slash-command worker is a per-session child process. MCP tools discovered
in the parent ``hermes serve`` / dashboard process do not populate the worker's
in-memory registry, so ``HermesCLI`` must start its own bounded discovery
before the first tool snapshot.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prepare_runtime_runs_before_hermes_cli_in_main():
    """MCP discovery must be armed before ``HermesCLI`` construction."""
    src = (PROJECT_ROOT / "tui_gateway" / "slash_worker.py").read_text()
    tree = ast.parse(src)

    prepare_line = None
    cli_call_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prepare_slash_worker_runtime"
        ):
            if prepare_line is None:
                prepare_line = node.lineno
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HermesCLI"
        ):
            if cli_call_line is None:
                cli_call_line = node.lineno

    assert prepare_line is not None, "slash_worker must call _prepare_slash_worker_runtime()"
    assert cli_call_line is not None, "slash_worker must construct HermesCLI"
    assert prepare_line < cli_call_line, (
        "_prepare_slash_worker_runtime() must run before HermesCLI() (issue #61891)"
    )


def test_prepare_slash_worker_runtime_starts_and_waits(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **kwargs: calls.append("start"),
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.wait_for_mcp_discovery",
        lambda timeout=None: calls.append("wait"),
    )

    from tui_gateway.slash_worker import _prepare_slash_worker_runtime

    _prepare_slash_worker_runtime()

    assert calls == ["start", "wait"]


def test_main_invokes_mcp_prepare_before_cli(monkeypatch):
    """main() wires discovery ahead of the expensive HermesCLI build."""
    calls: list[str] = []

    class _FakeProc:
        def create_time(self):
            return 0.0

    monkeypatch.setattr(
        "tui_gateway.slash_worker._start_parent_death_watchdog",
        lambda *args, **kwargs: calls.append("watchdog"),
    )
    monkeypatch.setattr(
        "tui_gateway.slash_worker._prepare_slash_worker_runtime",
        lambda: calls.append("mcp"),
    )
    monkeypatch.setattr(
        "tui_gateway.slash_worker.HermesCLI",
        lambda **kwargs: calls.append("cli") or object(),
    )
    monkeypatch.setattr(
        "tui_gateway.slash_worker.psutil.Process",
        lambda pid: _FakeProc(),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["slash_worker", "--session-key", "test-key"],
    )

    with patch("sys.stdin", []):
        from tui_gateway import slash_worker

        slash_worker.main()

    assert calls[:3] == ["watchdog", "mcp", "cli"]

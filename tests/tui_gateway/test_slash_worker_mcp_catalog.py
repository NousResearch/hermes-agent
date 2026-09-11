"""Test slash worker MCP catalog join (#92330).

A slash worker is a separate process with no late-refresh: if an MCP server's
stdio handshake takes longer than the interactive bound, /tools would print the
catalog without that server and nothing would ever correct it â€” reading as
"server broken" when it is "you asked too early". The worker drops a marker so
the CLI's /tools joins in-flight discovery with the generous bound; the TUI path
late-refreshes and must not block, hence the marker gate.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

_MARKER = os.path.join(tempfile.gettempdir(), "hermes-slash-worker-marker")


def test_slash_worker_marker_created(tmp_path, monkeypatch):
    """_prepare_slash_worker_runtime drops the marker file."""
    monkeypatch.setattr(
        "tui_gateway.slash_worker._SLASH_WORKER_MARKER",
        str(tmp_path / "marker"),
    )
    with patch("hermes_cli.mcp_startup.start_background_mcp_discovery"):
        with patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"):
            from tui_gateway.slash_worker import _prepare_slash_worker_runtime

            _prepare_slash_worker_runtime()

    assert (tmp_path / "marker").exists()


def test_slash_worker_marker_write_never_raises(tmp_path, monkeypatch):
    """A read-only temp dir must not break worker startup (best-effort marker)."""
    monkeypatch.setattr(
        "tui_gateway.slash_worker._SLASH_WORKER_MARKER",
        str(tmp_path / "denied" / "marker"),  # parent dir does not exist
    )
    with patch("hermes_cli.mcp_startup.start_background_mcp_discovery"):
        with patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"):
            from tui_gateway.slash_worker import _prepare_slash_worker_runtime

            _prepare_slash_worker_runtime()  # must not raise


def _fake_cli():
    return type("FakeCLI", (), {
        "enabled_toolsets": set(),
        "disabled_toolsets": set(),
    })()


def test_show_tools_joins_mcp_discovery_when_marker_present():
    """cli.show_tools joins in-flight discovery when the slash-worker marker exists."""
    Path(_MARKER).touch()
    try:
        joined: list = []
        with patch(
            "hermes_cli.mcp_startup.join_mcp_discovery",
            side_effect=lambda timeout=None: joined.append(timeout),
        ):
            from hermes_cli.cli_info_mixin import CLIInfoMixin

            CLIInfoMixin.show_tools(_fake_cli())

        assert joined == [30.0]
    finally:
        Path(_MARKER).unlink(missing_ok=True)


def test_show_tools_skips_join_without_marker():
    """No marker (the TUI / plain-CLI path): never blocks on discovery."""
    assert not os.path.exists(_MARKER), "test requires a clean marker state"
    joined: list = []
    with patch(
        "hermes_cli.mcp_startup.join_mcp_discovery",
        side_effect=lambda timeout=None: joined.append(timeout),
    ):
        from hermes_cli.cli_info_mixin import CLIInfoMixin

        CLIInfoMixin.show_tools(_fake_cli())

    assert joined == []

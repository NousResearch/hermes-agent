"""Integration tests for LightpandaProvider using a mock binary.

These tests exercise the *real* provider code end-to-end — no mocking of
internal methods — by pointing LIGHTPANDA_PATH at a tiny Python script that
starts a minimal CDP-compatible HTTP server.

This approach:
  - Tests the actual subprocess spawn, CDP readiness poll, URL construction,
    session tracking, and process cleanup paths.
  - Does not require the real Lightpanda binary (which may not be available on
    all CI systems or older glibc platforms).
  - Runs against whichever Python interpreter is in PATH, so it works anywhere
    a standard Python 3 is installed.

To run only these tests:
    pytest tests/tools/test_browser_lightpanda_integration.py -v
"""

import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

import pytest

# Path to the mock binary shipped alongside these tests.
_MOCK_BINARY = Path(__file__).parent / "helpers" / "mock_lightpanda.py"

# Wrapper script that calls the mock binary via the current Python interpreter,
# needed because the PATH-discovered binary must be executable directly.
_MOCK_WRAPPER = Path(__file__).parent / "helpers" / "mock_lightpanda_wrapper.sh"


def _free_port() -> int:
    """Return an unbound ephemeral TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """Return True once a TCP connection can be made to host:port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def mock_binary_path(tmp_path_factory) -> str:
    """Write a tiny shell wrapper that calls mock_lightpanda.py via Python."""
    tmp = tmp_path_factory.mktemp("mock_bin")
    wrapper = tmp / "lightpanda"
    python_exe = sys.executable
    wrapper.write_text(
        f"#!/bin/sh\nexec {python_exe} {_MOCK_BINARY} \"$@\"\n"
    )
    wrapper.chmod(0o755)
    return str(wrapper)


# ---------------------------------------------------------------------------
# Full create → use → close lifecycle
# ---------------------------------------------------------------------------

class TestLightpandaProviderIntegration:
    def test_create_session_starts_cdp_server(self, mock_binary_path, monkeypatch):
        """create_session() should start a process and expose a reachable CDP URL."""
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_PATH", mock_binary_path)
        monkeypatch.setenv("LIGHTPANDA_STARTUP_TIMEOUT", "10")

        provider = LightpandaProvider()
        assert provider.is_configured(), "should detect binary from LIGHTPANDA_PATH"

        session = provider.create_session("integ_task_01")

        try:
            # cdp_url must be a ws:// URL
            assert session["cdp_url"].startswith("ws://"), session["cdp_url"]
            # session_name must embed the task id
            assert "integ_task_01" in session["session_name"]
            # bb_session_id is the internal key used for close/cleanup
            assert session["bb_session_id"]
            assert session["features"] == {"lightpanda": True}

            # The HTTP server that backs the CDP endpoint must be reachable
            host, port_str = session["cdp_url"].removeprefix("ws://").split(":")
            port = int(port_str)
            assert _wait_for_port(host, port), f"CDP port {port} not reachable"

            # /json/version must return valid JSON
            resp = urllib.request.urlopen(
                f"http://{host}:{port}/json/version", timeout=3
            )
            import json
            data = json.loads(resp.read())
            assert "webSocketDebuggerUrl" in data

        finally:
            provider.close_session(session["bb_session_id"])

    def test_close_session_terminates_process(self, mock_binary_path, monkeypatch):
        """close_session() must terminate the lightpanda process."""
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_PATH", mock_binary_path)

        provider = LightpandaProvider()
        session = provider.create_session("integ_task_02")

        host, port_str = session["cdp_url"].removeprefix("ws://").split(":")
        port = int(port_str)
        assert _wait_for_port(host, port), "server should be up before close"

        result = provider.close_session(session["bb_session_id"])
        assert result is True

        # Give OS a moment to reclaim the port
        time.sleep(0.2)

        # The port should now be closed
        reachable = _wait_for_port(host, port, timeout=1.0)
        assert not reachable, f"port {port} still open after close_session"

        # The session must be removed from internal tracking
        assert session["bb_session_id"] not in provider._processes

    def test_session_not_in_registry_after_close(self, mock_binary_path, monkeypatch):
        """Session must be deregistered so a second close returns False."""
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_PATH", mock_binary_path)

        provider = LightpandaProvider()
        session = provider.create_session("integ_task_03")
        provider.close_session(session["bb_session_id"])

        # Second close on the same ID must be a no-op (returns False)
        assert provider.close_session(session["bb_session_id"]) is False

    def test_emergency_cleanup_stops_process(self, mock_binary_path, monkeypatch):
        """emergency_cleanup() must kill the process without raising."""
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_PATH", mock_binary_path)

        provider = LightpandaProvider()
        session = provider.create_session("integ_task_04")
        host, port_str = session["cdp_url"].removeprefix("ws://").split(":")
        port = int(port_str)
        assert _wait_for_port(host, port)

        provider.emergency_cleanup(session["bb_session_id"])  # must not raise

        time.sleep(0.2)
        reachable = _wait_for_port(host, port, timeout=1.0)
        assert not reachable, f"port {port} still open after emergency_cleanup"

    def test_multiple_concurrent_sessions_use_different_ports(
        self, mock_binary_path, monkeypatch
    ):
        """Each session must get its own process on its own port."""
        from tools.browser_providers.lightpanda import LightpandaProvider

        monkeypatch.setenv("LIGHTPANDA_PATH", mock_binary_path)

        provider = LightpandaProvider()
        sessions = []
        ports = []

        try:
            for i in range(3):
                s = provider.create_session(f"integ_concurrent_{i}")
                sessions.append(s)
                _, port_str = s["cdp_url"].removeprefix("ws://").split(":")
                ports.append(int(port_str))

            # All ports must be distinct
            assert len(set(ports)) == 3, f"expected 3 unique ports, got {ports}"

            # All sessions must be reachable simultaneously
            for s, port in zip(sessions, ports):
                assert _wait_for_port("127.0.0.1", port, timeout=3.0), (
                    f"session {s['session_name']} not reachable on port {port}"
                )

        finally:
            for s in sessions:
                provider.close_session(s["bb_session_id"])

    def test_startup_timeout_error_kills_process(self, tmp_path, monkeypatch):
        """If CDP never becomes ready within the timeout, the process must be killed."""
        import subprocess

        # Create a fake lightpanda that starts but never serves HTTP
        slow_binary = tmp_path / "lightpanda"
        slow_binary.write_text("#!/bin/sh\nsleep 30\n")
        slow_binary.chmod(0o755)

        monkeypatch.setenv("LIGHTPANDA_PATH", str(slow_binary))
        monkeypatch.setenv("LIGHTPANDA_STARTUP_TIMEOUT", "1")

        from tools.browser_providers.lightpanda import LightpandaProvider

        provider = LightpandaProvider()
        with pytest.raises(RuntimeError, match="did not become ready"):
            provider.create_session("integ_timeout")

        # No process should be tracked after the timeout error
        assert len(provider._processes) == 0

    def test_is_configured_false_without_binary(self, tmp_path, monkeypatch):
        """is_configured() must return False when no binary is available."""
        from unittest.mock import patch

        monkeypatch.delenv("LIGHTPANDA_PATH", raising=False)
        # point PATH to an empty dir so shutil.which finds nothing
        empty_bin = tmp_path / "bin"
        empty_bin.mkdir()
        monkeypatch.setenv("PATH", str(empty_bin))

        from tools.browser_providers.lightpanda import LightpandaProvider

        # Also suppress the node_modules/.bin fallback so a local install
        # does not cause a false positive.
        with patch.object(Path, "exists", return_value=False):
            assert LightpandaProvider().is_configured() is False


# ---------------------------------------------------------------------------
# CDP server response validation
# ---------------------------------------------------------------------------

class TestMockCdpServer:
    """Sanity-checks that our mock binary itself works correctly."""

    def test_mock_responds_to_json_version(self, mock_binary_path):
        """The mock server must return a /json/version payload with webSocketDebuggerUrl."""
        import json
        import subprocess
        import time

        port = _free_port()
        proc = subprocess.Popen(
            [mock_binary_path, "serve", "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert _wait_for_port("127.0.0.1", port, timeout=5.0), (
                "mock server did not start"
            )
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=3
            )
            data = json.loads(resp.read())
            assert "webSocketDebuggerUrl" in data
            assert data["webSocketDebuggerUrl"].startswith("ws://")
        finally:
            proc.terminate()
            proc.wait(timeout=5)

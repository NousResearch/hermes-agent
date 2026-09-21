"""The loopback OAuth callback latches the first terminal result (#116278).

A browser follows the ``/callback`` redirect with queryless fetches (``/favicon.ico``), and the CLI waiter
samples the result only every 500 ms. A handler that wrote every GET into the result lost the stored code
between two polls, so the user saw "Authorization Successful" while ``hermes mcp login`` timed out. These
tests drive the production entry (``_make_callback_waiter`` → ``_start_callback_server`` → handler) with a
browser stand-in that sends its requests back-to-back, well inside one poll interval.
"""
import asyncio
import io
import socket
import threading
from http.client import HTTPConnection

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

import tools.mcp_oauth as mo


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str) -> int:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        resp.read()
        return resp.status
    finally:
        conn.close()


def _wait_listening(port: int) -> None:
    for _ in range(200):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            threading.Event().wait(0.02)
    raise AssertionError("callback listener never bound")


def _drive_waiter(monkeypatch, paths: list[str]):
    """Run the real waiter on its own loop; send *paths* back-to-back once the listener is bound."""
    monkeypatch.setattr(mo.sys, "stdin", io.StringIO())  # paste reader sees EOF; the HTTP listener is under test
    port = _free_port()
    out: dict = {}

    def run():
        async def main():
            with mo.force_interactive_oauth():
                return await mo._make_callback_waiter(port, timeout=4)()
        try:
            out["result"] = asyncio.run(main())
        except Exception as exc:  # noqa: BLE001 — the timeout is the failure under test
            out["exc"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    _wait_listening(port)
    statuses = [_get(port, p) for p in paths]
    thread.join(timeout=15)
    assert not thread.is_alive(), "waiter did not finish"
    assert "exc" not in out, f"waiter raised {type(out.get('exc')).__name__}"
    return statuses, out["result"]


def test_favicon_right_after_callback_does_not_clobber_the_code(monkeypatch):
    statuses, result = _drive_waiter(
        monkeypatch, ["/callback?code=synthetic&state=s1&iss=https://as.example", "/favicon.ico"])
    assert statuses == [200, 404]
    assert (result.code, result.state, result.iss) == ("synthetic", "s1", "https://as.example")


def test_first_terminal_callback_wins_over_later_ones(monkeypatch):
    statuses, result = _drive_waiter(monkeypatch, [
        "/favicon.ico",
        "/callback?code=first&state=s1",
        "/callback?code=second&state=s2",
        "/callback?error=access_denied&state=s1",
    ])
    assert statuses == [404, 200, 200, 200]
    assert (result.code, result.state) == ("first", "s1")

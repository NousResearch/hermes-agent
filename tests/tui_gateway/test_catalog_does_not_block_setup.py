"""commands.catalog must not hold the RPC reader in front of setup.status.

The TUI fires commands.catalog without awaiting it, then awaits setup.status
before session.create. Catalog scans skills. setup.status is already a long
handler. If catalog stays inline, the reader cannot serve setup.status until
the scan finishes.

Hold the catalog scan and assert setup.status is answered while that hold is
still taken. A wall-clock budget is not the assertion.
"""

from __future__ import annotations

import threading
import time

from tui_gateway import server


class _RecordingTransport:
    def __init__(self):
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self.responses = {}

    def write(self, response):
        with self._lock:
            self.responses[response.get("id")] = response
            self._changed.set()
        return True

    def wait_for(self, request_id, timeout=2.0):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                response = self.responses.get(request_id)
                if response is not None:
                    return response
                self._changed.clear()
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"timed out waiting for response {request_id!r}"
            self._changed.wait(timeout=remaining)


def test_setup_status_answers_while_catalog_scan_is_held(monkeypatch):
    """A held catalog scan must not keep setup.status unanswered."""
    from hermes_cli import free_tier_bootstrap
    import hermes_cli.main as cli_main
    from tui_gateway import command_discovery

    started = threading.Event()
    release = threading.Event()

    def held_scan(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=5), "test failed to release the catalog hold"
        return ""

    monkeypatch.setattr(command_discovery, "_catalog_skills", held_scan)
    monkeypatch.setattr(
        command_discovery, "_catalog_quick_commands", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        command_discovery, "_catalog_plugin_commands", lambda *_a, **_k: None
    )
    monkeypatch.setattr(cli_main, "_has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(free_tier_bootstrap, "wait_for_record", lambda *_a, **_k: None)

    transport = _RecordingTransport()
    assert (
        server.dispatch(
            {"id": "catalog", "method": "commands.catalog", "params": {}}, transport
        )
        is None
    )
    assert started.wait(timeout=2), "catalog never reached the skill scan"

    try:
        assert (
            server.dispatch(
                {"id": "setup", "method": "setup.status", "params": {}}, transport
            )
            is None
        )
        setup = transport.wait_for("setup", timeout=2.0)
    finally:
        release.set()

    assert setup.get("error") is None
    assert setup["result"]["provider_configured"] is True
    # Released: the catalog still completes with its payload. Only its thread moved.
    catalog = transport.wait_for("catalog", timeout=2.0)
    assert catalog.get("error") is None
    assert "pairs" in catalog["result"]

"""gateway.ready carries the active launch profile (#36081).

``hermes -p <name> --tui`` users must see the active profile at a glance in the
status bar, without waiting for a session to exist. The name rides additively
on the startup event both transports already emit:

- stdio: ``tui_gateway/entry.main`` writes it before entering the read loop;
- WS:    ``tui_gateway/ws.handle_ws`` writes it right after connection accept.

Contract under test:

- named profile (HERMES_HOME under ``~/.hermes/profiles/<name>``) → the name;
- default home / unrecognized custom home → None, so the stock single-profile
  UX renders no profile segment (mirrors the composer prefix suppression);
- key-additive: ``skin`` / ``change_events`` consumers are untouched — old
  clients simply ignore the new key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + Path.home() — the repo's profile-test pattern."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class TestResolveLaunchProfile:
    def test_default_home_reports_none(self, hermes_home):
        from tui_gateway.server import resolve_launch_profile

        assert resolve_launch_profile() is None

    def test_unrecognized_custom_home_reports_none(self, hermes_home, monkeypatch):
        from tui_gateway.server import resolve_launch_profile

        custom = hermes_home.parent / "custom-home"
        custom.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(custom))

        assert resolve_launch_profile() is None

    def test_named_profile_reports_name(self, hermes_home, monkeypatch):
        from tui_gateway.server import resolve_launch_profile

        named = Path.home() / ".hermes" / "profiles" / "work"
        named.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(named))

        assert resolve_launch_profile() == "work"

    def test_import_failure_fails_closed_to_none(self, hermes_home, monkeypatch):
        """A profiles-module hiccup must never break the ready write."""
        from tui_gateway.server import resolve_launch_profile

        named = Path.home() / ".hermes" / "profiles" / "work"
        named.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(named))
        monkeypatch.setitem(__import__("sys").modules, "hermes_cli.profiles", None)

        assert resolve_launch_profile() is None


# ── stdio transport: entry.main's ready frame ────────────────────────────────


def _run_entry_main(monkeypatch) -> list[dict]:
    """Run entry.main() with stubbed collaborators; return every written frame.

    Harness mirrors tests/tui_gateway/test_entry_picker_prewarm.py: stubbed
    I/O collaborators, no subprocess, no real gateway; main() returns on the
    genuine stdin EOF.
    """
    import io

    from tui_gateway import entry

    writes: list[dict] = []

    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "_log_exit", lambda reason: None)
    # Genuine EOF — empty stdin drops main() straight out of its read loop.
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *a: False)
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)

    def _write_json(payload):
        writes.append(payload)
        return True

    monkeypatch.setattr(entry, "write_json", _write_json)

    # The prewarm helper is imported lazily from its own module inside main().
    import hermes_cli.model_switch as ms

    monkeypatch.setattr(ms, "prewarm_picker_cache_async", lambda: None)

    entry.main()

    return writes


def test_entry_ready_payload_carries_named_profile(monkeypatch, hermes_home):
    from tui_gateway.server import resolve_skin

    named = Path.home() / ".hermes" / "profiles" / "work"
    named.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(named))

    frames = _run_entry_main(monkeypatch)
    ready = next(f for f in frames if f.get("params", {}).get("type") == "gateway.ready")
    payload = ready["params"]["payload"]

    assert payload["profile"] == "work"
    # Additive: the pre-existing keys ride along untouched.
    assert payload["change_events"] is True
    assert payload["skin"] == resolve_skin()


def test_entry_ready_payload_omits_default_profile(monkeypatch, hermes_home):
    from tui_gateway import entry

    # Patch on entry itself — main() calls the name it imported at module load.
    monkeypatch.setattr(entry, "resolve_skin", lambda: {"name": "default"})
    frames = _run_entry_main(monkeypatch)

    ready = next(f for f in frames if f.get("params", {}).get("type") == "gateway.ready")
    payload = ready["params"]["payload"]

    assert payload["profile"] is None
    assert payload["change_events"] is True
    assert payload["skin"] == {"name": "default"}


# ── WebSocket transport: handle_ws's ready frame ─────────────────────────────


class _ReadyCaptureWS:
    """FakeWS that records every line written and disconnects immediately."""

    def __init__(self):
        self.lines: list[str] = []

    async def accept(self):
        pass

    async def send_text(self, line):
        self.lines.append(line)

    async def receive_text(self):
        import tui_gateway.ws as ws_mod

        raise ws_mod._WebSocketDisconnect()

    async def close(self):
        pass


def _run_handle_ws(monkeypatch) -> list[dict]:
    """Drive the REAL handle_ws through its ready write; return decoded frames."""
    import tui_gateway.ws as ws_mod

    monkeypatch.setattr(
        "tui_gateway.server._ensure_skin_watcher", lambda: None
    )

    fake = _ReadyCaptureWS()
    asyncio.run(ws_mod.handle_ws(fake))

    return [json.loads(line) for line in fake.lines]


def test_ws_ready_frame_carries_named_profile(monkeypatch, hermes_home):
    import tui_gateway.server as server_mod

    named = Path.home() / ".hermes" / "profiles" / "work"
    named.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(named))
    monkeypatch.setattr(server_mod, "resolve_skin", lambda: {"palette": "wired"})

    frames = _run_handle_ws(monkeypatch)
    ready = next(f for f in frames if f.get("params", {}).get("type") == "gateway.ready")
    payload = ready["params"]["payload"]

    assert payload["profile"] == "work"
    assert payload["change_events"] is True
    assert payload["skin"] == {"palette": "wired"}


def test_ws_ready_frame_reports_none_for_default_home(monkeypatch, hermes_home):
    import tui_gateway.server as server_mod

    monkeypatch.setattr(server_mod, "resolve_skin", lambda: {"name": "default"})

    frames = _run_handle_ws(monkeypatch)
    ready = next(f for f in frames if f.get("params", {}).get("type") == "gateway.ready")
    payload = ready["params"]["payload"]

    assert payload["profile"] is None
    assert payload["change_events"] is True

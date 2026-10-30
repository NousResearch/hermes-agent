"""``pdf.attach`` render-budget tests (issue #146639): the pdftoppm subprocess
timeout must scale with the requested page range, not sit at a flat 120 s."""

import base64
import subprocess
import threading

import pytest

from tui_gateway import server

_FAKE_PDF = base64.b64encode(b"%PDF-1.4 fake").decode()


@pytest.fixture()
def session_id(tmp_path, monkeypatch):
    """A building-state fake session, shaped like the real session record."""
    import sys
    import types

    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png", ".jpg"}
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    sid = "s1"
    record = {
        "profile_home": str(tmp_path),
        "cwd": str(tmp_path),
        "attached_images": [],
        "image_counter": 0,
        "running": False,
        "history_lock": threading.Lock(),
    }
    sessions = getattr(server, "_sessions", None)
    if sessions is not None:
        sessions[sid] = record
    monkeypatch.setattr(server, "_hermes_home", tmp_path, raising=False)
    yield sid
    if sessions is not None:
        sessions.pop(sid, None)


def _capture_run(monkeypatch):
    """Intercept the handler's ``subprocess.run`` so the real pdftoppm never
    runs and we can read the timeout kwarg it was given."""
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_single_page_keeps_the_120s_floor(session_id, monkeypatch):
    captured = _capture_run(monkeypatch)
    server._methods["pdf.attach"](1, {
        "session_id": session_id, "content_base64": _FAKE_PDF, "last_page": 1})
    assert captured["timeout"] == 120


def test_timeout_scales_with_requested_page_range(session_id, monkeypatch):
    captured = _capture_run(monkeypatch)
    # ``last_page=None`` means "render through the per-call cap", so the
    # budget must be sized for 25 pages, not one: 120 + 24 * 15 = 480.
    server._methods["pdf.attach"](1, {
        "session_id": session_id, "content_base64": _FAKE_PDF, "last_page": 25})
    assert captured["timeout"] == 480


def test_timeout_error_names_the_applied_budget(session_id, monkeypatch):
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    reply = server._methods["pdf.attach"](1, {
        "session_id": session_id, "content_base64": _FAKE_PDF, "last_page": 25})
    assert reply["error"]["message"] == "pdftoppm timed out (>480s)"

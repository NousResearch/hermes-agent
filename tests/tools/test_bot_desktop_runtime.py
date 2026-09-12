"""Bot Desktop runtime: thumbnail and display-number allocation invariants."""

from __future__ import annotations

import sys

from tools.bot_desktop import runtime, thumbnail


def test_no_running_screen_returns_none_without_grabbing(monkeypatch):
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":99"})
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", None)  # an import would now fail loudly
    assert thumbnail.thumbnail_data_url() is None


def test_recorded_display_held_by_a_live_server_is_not_reused(tmp_path, monkeypatch):
    """After profile A stops, B may take A's number; A restarting must pick another rather than
    unlink B's socket and lock."""
    import os

    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    (tmp_path / "display").write_text("37", encoding="utf-8")
    live = {37: os.getpid()}  # :37 is owned by a running server (this very process stands in for it)
    monkeypatch.setattr(runtime, "_display_in_use", lambda num: num in live)
    monkeypatch.setattr(runtime, "_ALLOC_LOCK", tmp_path / "alloc.lock")
    assert runtime._allocate_display() != 37
    live.clear()
    assert runtime._allocate_display() == 37, "a free recorded number is reclaimed"

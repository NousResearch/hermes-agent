"""Bot Desktop thumbnail: a stopped screen yields no frame and never touches X."""

from __future__ import annotations

import sys

from tools.bot_desktop import runtime, thumbnail


def test_no_running_screen_returns_none_without_grabbing(monkeypatch):
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":99"})
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", None)  # an import would now fail loudly
    assert thumbnail.thumbnail_data_url() is None

"""Inline transcript images over the gateway: ``image.render`` and ``tool.complete``'s ``image_path``."""

import json
from types import SimpleNamespace

import pytest

from tui_gateway import server

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.png"
    PIL.new("RGBA", (320, 160), (0, 200, 90, 255)).save(path)
    return str(path)


def _render(monkeypatch, params, inline_images="auto"):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"inline_images": inline_images}})
    response = server.handle_request({"id": "img", "method": "image.render", "params": params})
    assert "error" not in response, response
    return response["result"]


def test_render_without_a_terminal_returns_half_block_cells(monkeypatch, photo):
    result = _render(monkeypatch, {"path": photo, "cols": 20, "rows": 8})
    assert result["mode"] == "unicode"
    assert len(result["cells"]) == result["rows"] and len(result["cells"][0]) == result["cols"] <= 20
    assert result["cells"][0][0] == [0, 200, 90, 255, 0, 200, 90, 255]


def test_render_kitty_for_a_kitty_terminal(monkeypatch, photo):
    for key in ("TMUX", "TERM_PROGRAM", "WEZTERM_PANE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERM", "xterm-kitty")
    result = _render(monkeypatch, {"path": photo, "cols": 20, "rows": 8, "graphics": True})
    assert result["mode"] == "kitty"
    assert result["transmit"].startswith("\x1b_G") and len(result["placeholder"]) == result["rows"]


def test_render_honours_off_and_refuses_non_images(monkeypatch, photo, tmp_path):
    assert _render(monkeypatch, {"path": photo, "cols": 20, "rows": 8}, inline_images="off") == {"mode": "off"}
    notes = tmp_path / "notes.txt"
    notes.write_text("secret")
    assert _render(monkeypatch, {"path": str(notes), "cols": 20, "rows": 8}) == {"mode": "off"}
    assert _render(monkeypatch, {"path": str(tmp_path / "missing.png"), "cols": 20, "rows": 8}) == {"mode": "off"}


def test_tool_complete_names_the_image_the_tool_looked_at(monkeypatch, photo):
    events = []
    monkeypatch.setattr(server, "_emit", lambda event_type, sid, payload=None: events.append((event_type, payload)))
    monkeypatch.setitem(server._sessions, "img-sid", {
        "show_reasoning": True, "tool_progress_mode": "all", "tool_started_at": {}, "edit_snapshots": {},
        "agent": SimpleNamespace(reasoning_config={"enabled": True, "effort": "high"}),
    })

    server._on_tool_complete("img-sid", "t1", "vision_analyze", {"image_url": photo}, json.dumps({"analysis": "a cat"}))
    server._on_tool_complete("img-sid", "t2", "search_files", {"pattern": "*.png"}, json.dumps({"matches": [photo]}))

    completes = {payload["tool_id"]: payload for event, payload in events if event == "tool.complete"}
    assert completes["t1"]["image_path"] == photo
    assert "image_path" not in completes["t2"]


def test_canonical_local_viewer_frames_carry_the_image(photo):
    # The shared gateway's tool.complete for its local CLI/TUI viewer: same field as the sidecar's.
    from gateway.local_viewer_events import complete_fields

    turn = SimpleNamespace()
    assert complete_fields(turn, "c1", "vision_analyze", {"image_url": photo}, '{"analysis": "a cat"}')["image_path"] == photo
    assert complete_fields(turn, "c2", "terminal", {"command": "ls"}, '{"output": "' + photo.replace("\\", "/") + '"}')["image_path"] is None

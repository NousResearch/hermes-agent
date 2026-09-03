import json

from tools import screen_tutor_point_tool as st


def test_emits_normalized_pointer(monkeypatch):
    seen = {}

    def emit(event, payload):
        seen.update(event=event, payload=payload)
        return True

    monkeypatch.setattr(st.desktop_ui, "emit", emit)
    result = json.loads(st.screen_tutor_point_tool(" 7 ", 0.25, 0.75, " Save "))

    assert result == {"display_id": "7", "success": True, "x": 0.25, "y": 0.75}
    assert seen == {
        "event": "screen.tutor.point",
        "payload": {"display_id": "7", "label": "Save", "x": 0.25, "y": 0.75},
    }


def test_rejects_out_of_range_coordinates(monkeypatch):
    monkeypatch.setattr(st.desktop_ui, "emit", lambda *_: (_ for _ in ()).throw(AssertionError("must not emit")))

    assert "between 0 and 1" in json.loads(st.screen_tutor_point_tool("7", 1.1, 0.5))["error"]
    assert "display_id" in json.loads(st.screen_tutor_point_tool("", 0.5, 0.5))["error"]


def test_fails_when_no_desktop_renderer(monkeypatch):
    monkeypatch.setattr(st.desktop_ui, "emit", lambda *_: False)

    assert "desktop app" in json.loads(st.screen_tutor_point_tool("7", 0.5, 0.5))["error"]

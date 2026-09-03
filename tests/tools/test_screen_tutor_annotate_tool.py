import json

from tools import screen_tutor_annotate_tool as st


def test_emits_normalized_annotations(monkeypatch):
    seen = {}
    monkeypatch.setattr(st.desktop_ui, "emit", lambda event, payload: seen.update(event=event, payload=payload) or True)

    result = json.loads(
        st.screen_tutor_annotate_tool(
            "7",
            [{"kind": "arrow", "x": 0.1, "y": 0.8, "x2": 0.7, "y2": 0.2, "label": " Breakout ", "color": "amber"}],
            frozen=True,
            guide={
                "id": "excel-pivot",
                "title": "Build a pivot table",
                "instruction": "Open the Insert tab",
                "step": 1,
                "total": 4,
                "success_check": "The Insert ribbon is visible",
            },
        )
    )

    assert result == {"count": 1, "display_id": "7", "frozen": True, "success": True}
    assert seen["event"] == "screen.tutor.annotations"
    assert seen["payload"]["annotations"][0]["label"] == "Breakout"
    assert seen["payload"]["ttl_ms"] == 30_000
    assert seen["payload"]["guide"]["id"] == "excel-pivot"
    assert seen["payload"]["guide"]["success_check"] == "The Insert ribbon is visible"


def test_rejects_invalid_annotations(monkeypatch):
    monkeypatch.setattr(st.desktop_ui, "emit", lambda *_: (_ for _ in ()).throw(AssertionError("must not emit")))
    result = json.loads(st.screen_tutor_annotate_tool("7", [{"kind": "line", "x": 0, "y": 0}]))
    assert "No valid annotations" in result["error"]

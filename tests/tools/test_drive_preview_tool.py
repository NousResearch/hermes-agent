"""Tests for the GUI-surface ``drive_preview`` tool."""

import json

from tools import drive_preview_tool as ap




def test_targeted_inspection_preserves_targets_limit_and_renderer_result():
    """The existing bridge forwards targeted reads without interpreting page data."""
    result = {"success": True, "inspection": {"candidateCount": 0, "candidates": [], "truncated": False}}
    for target in ({"selector": "svg circle"}, {"ref": "btn-resize"}):
        calls = []

        def callback(payload):
            calls.append(payload)
            return json.dumps(result)

        answer = ap.registry.dispatch("drive_preview", {"action": "elements", **target, "max": 2}, callback=callback)
        assert calls == [{"action": "elements", **target, "max": 2}]
        assert json.loads(answer) == result


def test_inspection_cap_survives_final_tool_json_serialization():
    """JSON separator expansion and non-BMP characters count at the tool boundary."""
    for hint in ("\\" * 21 + "x" * 59, "😀" * 40, "\u0000" * 80):
        node = {"tag": "button", "nthOfType": 1, "id": hint, "class": hint, "testId": hint}
        candidate = {
            "node": node, "ancestors": [node] * 4,
            "rect": {"left": 0, "top": 0, "right": 20, "bottom": 20, "width": 20, "height": 20},
            "point": {"x": 10, "y": 10}, "centerInViewport": True,
            "style": {"pointerEvents": "auto", "visibility": "visible", "display": "block"},
            "hit": {"node": node, "ancestors": [node] * 4, "relationship": "self"},
        }
        inspection = {
            "coordinateSpace": "guest-viewport-css-pixels", "candidateCount": 5,
            "truncated": False, "viewport": {"width": 900, "height": 700, "scrollX": 0, "scrollY": 0, "devicePixelRatio": 1},
            "candidates": [candidate] * 5,
        }
        payload = {"success": True, "inspection": inspection}
        # Match the guest's compact JSON cap, including escaped control characters.
        while len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-16-le")) // 2 > 12000:
            inspection["candidates"].pop()
            inspection["truncated"] = True
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        answer = ap.registry.dispatch("drive_preview", {"action": "elements", "selector": "button"}, callback=lambda _: raw)
        assert len(answer.encode("utf-16-le")) // 2 <= 12000
        result = json.loads(answer)
        assert result["success"] is True
        assert result["inspection"]["candidateCount"] == 5
        assert len(result["inspection"]["candidates"]) <= len(inspection["candidates"])
        assert all(entry == candidate for entry in result["inspection"]["candidates"])
        assert result["inspection"]["truncated"] == (len(result["inspection"]["candidates"]) < 5)


def test_inspection_cap_drops_whole_entries_and_fails_closed_without_affecting_inventory():
    hint = "\\" * 80
    node = {"tag": "button", "nthOfType": 1, "id": hint, "class": hint, "testId": hint}
    candidate = {"node": node, "ancestors": [node] * 4, "hit": {"node": node, "ancestors": [node] * 4}}
    payload = {"success": True, "inspection": {"candidateCount": 9, "truncated": True, "candidates": [candidate] * 5}}
    raw = json.dumps(payload)
    for target in ({"ref": "btn-resize"}, {"selector": "button"}):
        answer = ap.registry.dispatch("drive_preview", {"action": "elements", **target}, callback=lambda _: raw)
        assert len(answer.encode("utf-16-le")) // 2 <= 12000
        inspection = json.loads(answer)["inspection"]
        assert 0 < len(inspection["candidates"]) < 5
        assert all(entry == candidate for entry in inspection["candidates"])
        assert inspection["candidateCount"] == 9
        assert inspection["truncated"] is True
        bad = json.dumps({"inspection": {"candidates": [], "unexpected": "private" * 12000}})
        failure = ap.registry.dispatch("drive_preview", {"action": "elements", **target}, callback=lambda _: bad)
        assert "response limit" in json.loads(failure)["error"]
        assert "private" not in failure
    ordinary = ap.registry.dispatch("drive_preview", {"action": "elements"}, callback=lambda _: raw)
    assert json.loads(ordinary) == payload
    assert len(ordinary.encode("utf-16-le")) // 2 > 12000


def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no crash."""
    result = json.loads(ap.drive_preview_tool(action="elements", callback=None))
    assert "desktop" in result["error"]


def test_rejects_an_unknown_action():
    result = json.loads(ap.drive_preview_tool(action="teleport", callback=lambda _p: "{}"))
    assert "action must be one of" in result["error"]


def test_interaction_verbs_need_a_target():
    """A click with nowhere to land is a mistake worth naming before the bridge."""
    for verb in ("click", "type", "press"):
        result = json.loads(ap.drive_preview_tool(action=verb, text="x", key="Enter", callback=lambda _p: "{}"))
        assert "ref" in result["error"], verb


def test_type_needs_text_and_press_needs_a_key():
    calls = []

    def cb(payload):
        calls.append(payload)
        return json.dumps({"success": True})

    assert "text" in json.loads(ap.drive_preview_tool(action="type", ref="@e1", callback=cb))["error"]
    assert "key" in json.loads(ap.drive_preview_tool(action="press", ref="@e1", callback=cb))["error"]
    assert calls == []


def test_typing_an_empty_string_is_allowed():
    """Clearing a field is a real intent — only a missing `text` is an error."""
    seen = {}
    ap.drive_preview_tool(action="type", ref="@e1", text="", callback=lambda p: seen.update(p) or "{}")

    assert seen["text"] == ""


def test_scroll_needs_no_target_and_validates_its_destination():
    seen = {}

    def cb(payload):
        seen.clear()
        seen.update(payload)
        return json.dumps({"success": True})

    ap.drive_preview_tool(action="scroll", callback=cb)
    assert seen == {"action": "scroll"}

    result = json.loads(ap.drive_preview_tool(action="scroll", to="sideways", callback=cb))
    assert "to must be one of" in result["error"]


def test_payload_forwards_only_what_was_given():
    seen = {}
    ap.drive_preview_tool(
        action="type",
        ref="inp-password",
        text="hunter2",
        submit=True,
        callback=lambda p: seen.update(p) or json.dumps({"success": True}),
    )

    assert seen == {"action": "type", "ref": "inp-password", "text": "hunter2", "submit": True}




def test_numeric_arguments_are_validated():
    result = json.loads(ap.drive_preview_tool(action="scroll", amount="lots", callback=lambda _p: "{}"))
    assert "integers" in result["error"]


def test_empty_answer_means_nothing_open():
    result = json.loads(ap.drive_preview_tool(action="elements", callback=lambda _p: ""))
    assert "open_preview" in result["error"]


def test_passes_the_renderer_answer_through():
    payload = {
        "success": True,
        "acted": 'clicked button "Sign in"',
        "url": "https://example.com/app",
        "elements": [{"ref": "@e1", "role": "button", "label": "Log out", "selector": "#out"}],
    }
    result = json.loads(ap.drive_preview_tool(action="click", ref="@e2", callback=lambda _p: json.dumps(payload)))

    assert result == payload


def test_wraps_non_json_text():
    result = json.loads(ap.drive_preview_tool(action="elements", callback=lambda _p: "plain words"))
    assert result == {"text": "plain words"}


def test_callback_failure_is_reported():
    def _boom(_payload):
        raise RuntimeError("renderer went away")

    result = json.loads(ap.drive_preview_tool(action="elements", callback=_boom))
    assert "renderer went away" in result["error"]


def test_empty_answer_distinguishes_no_tab_from_a_stale_app():
    """An empty bridge answer used to be one merged "timed out, or no window
    answered" string that blamed a closed tab even when the pane was open on an
    app older than this backend (#94272): the two cases need different next
    steps (open a tab vs update the app)."""
    result = json.loads(ap.drive_preview_tool(action="elements", callback=lambda _p: ""))

    assert "no preview tab is open" in result["error"]
    assert "older than this backend" in result["error"]

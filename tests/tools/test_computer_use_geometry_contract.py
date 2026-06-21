import json

import pytest


class _FakeCuaSession:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def call_tool(self, name, args, timeout=30.0):
        self.calls.append((name, args))
        queue = self.replies.get(name, [])
        if not queue:
            raise AssertionError(f"unexpected call {name}: {args}")
        if isinstance(queue, list):
            return queue.pop(0)
        return queue


def _cua_backend_with(fake_session):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._session = fake_session
    return backend


def _list_windows_reply(*windows):
    return {"data": {"windows": list(windows)}, "structuredContent": None, "images": [], "isError": False}


def _state_reply(tree='- AXApplication "TextEdit"\n  - [0] AXWindow "scratch.txt"\n    - [7] AXButton "OK"\n'):
    return {"data": {"tree_markdown": tree}, "structuredContent": None, "images": [], "isError": False}


def test_parse_element_accepts_future_aliases_and_bounds_image_pixels():
    from tools.computer_use.cua_backend import _parse_element

    element = _parse_element({
        "element_index": 42,
        "role": "AXButton",
        "title": "Continue",
        "bounds_image_pixels": {"x": 11, "y": 22, "width": 33, "height": 44},
    })

    assert element.index == 42
    assert element.label == "Continue"
    assert element.bounds == (11, 22, 33, 44)
    assert element.attributes["bounds_available"] is True
    assert element.attributes["geometry_status"] == "direct"
    assert element.attributes["geometry_source"] == "cua_driver.elements"
    assert element.attributes["bounds_coordinate_space"] == "window_logical"
    assert element.attributes["bounds_confidence"] == 1.0


def test_parse_element_marks_structured_nonzero_bounds_as_direct_geometry():
    from tools.computer_use.cua_backend import _parse_element

    element = _parse_element({
        "elementIndex": 3,
        "role": "AXTextField",
        "description": "Search",
        "screen_bounds": [100, 200, 300, 40],
    }, source="cua_driver.ui_elements")

    assert element.index == 3
    assert element.label == "Search"
    assert element.bounds == (100, 200, 300, 40)
    assert element.attributes["geometry_status"] == "direct"
    assert element.attributes["geometry_source"] == "cua_driver.ui_elements"
    assert element.attributes["bounds_coordinate_space"] == "screen_logical"


def test_tree_markdown_elements_are_explicit_missing_geometry():
    from tools.computer_use.cua_backend import _parse_elements_from_tree

    elements = _parse_elements_from_tree('- AXApplication "TextEdit"\n  - [7] AXButton "OK"\n')

    assert elements[0].index == 7
    assert elements[0].bounds == (0, 0, 0, 0)
    assert elements[0].attributes["bounds_available"] is False
    assert elements[0].attributes["geometry_status"] == "missing"
    assert elements[0].attributes["geometry_source"] == "tree_markdown"
    assert elements[0].attributes["bounds_coordinate_space"] == "unknown"
    assert elements[0].attributes["bounds_confidence"] == 0.0


def test_tree_markdown_parser_scopes_to_window_subtree_not_app_menubar():
    from tools.computer_use.cua_backend import _parse_elements_from_tree

    tree = '''- AXApplication "iTerm2"
  - [0] AXWindow "python3" actions=[AXRaise]
    - [1] AXScrollArea actions=[AXScrollDownByPage]
      - [2] AXButton
  - [9] AXMenuBar actions=[AXCancel]
    - [10] AXMenuBarItem "Apple" actions=[AXPick]
'''

    elements = _parse_elements_from_tree(tree)

    assert [e.index for e in elements] == [0, 1, 2]
    assert all("Menu" not in e.role for e in elements)


def test_choose_target_window_prefers_real_content_over_menu_and_thumbnail():
    from tools.computer_use.cua_backend import _choose_target_window

    target = _choose_target_window([
        {"window_id": 1, "bounds": (0, 0, 2560, 30), "off_screen": True, "z_index": 10},
        {"window_id": 2, "bounds": (0, 580, 64, 64), "off_screen": True, "z_index": 9},
        {"window_id": 3, "bounds": (0, 30, 1280, 478), "off_screen": False, "z_index": 8},
    ])

    assert target is not None
    assert target["window_id"] == 3


def test_fake_ax_helper_merges_geometry_without_renumbering():
    from tools.computer_use.backend import UIElement
    from tools.computer_use.cua_backend import _merge_ax_geometry

    elements = [
        UIElement(
            index=7,
            role="AXButton",
            label="OK",
            attributes={"bounds_available": False, "geometry_status": "missing"},
        )
    ]
    payload = {
        "ok": True,
        "roots": [{
            "role": "AXWindow",
            "position": {"x": 100, "y": 200},
            "size": {"width": 800, "height": 600},
            "children": [{
                "role": "AXButton",
                "text": "OK",
                "position": {"x": 120, "y": 230},
                "size": {"width": 50, "height": 20},
                "actions": ["AXPress"],
            }],
        }],
    }

    merged = _merge_ax_geometry(
        elements,
        payload,
        window_bounds=(100, 200, 800, 600),
        capture_size=(800, 600),
    )

    assert merged == 1
    assert elements[0].index == 7
    assert elements[0].bounds == (20, 30, 50, 20)
    assert elements[0].attributes["geometry_status"] == "derived"
    assert elements[0].attributes["geometry_source"] == "hermes_ax_map"


def test_ambiguous_helper_matches_leave_geometry_missing():
    from tools.computer_use.backend import UIElement
    from tools.computer_use.cua_backend import _merge_ax_geometry

    elements = [
        UIElement(
            index=7,
            role="AXButton",
            label="OK",
            attributes={"bounds_available": False, "geometry_status": "missing"},
        )
    ]
    payload = {
        "ok": True,
        "roots": [{
            "role": "AXWindow",
            "children": [
                {"role": "AXButton", "text": "OK", "position": {"x": 10, "y": 10}, "size": {"width": 20, "height": 20}},
                {"role": "AXButton", "text": "OK", "position": {"x": 40, "y": 10}, "size": {"width": 20, "height": 20}},
            ],
        }],
    }

    merged = _merge_ax_geometry(
        elements,
        payload,
        window_bounds=(0, 0, 100, 100),
        capture_size=(100, 100),
    )

    assert merged == 0
    assert elements[0].bounds == (0, 0, 0, 0)
    assert elements[0].attributes["geometry_match_error"] == "ambiguous"
    assert elements[0].attributes["geometry_status"] == "missing"


def test_structured_cua_menu_bar_elements_are_filtered_from_window_content():
    from tools.computer_use.backend import UIElement
    from tools.computer_use.cua_backend import _is_window_content_element

    assert _is_window_content_element(UIElement(index=1, role="AXMenuButton", label="document actions")) is True
    assert _is_window_content_element(UIElement(index=2, role="AXMenuBar", label="")) is False
    assert _is_window_content_element(UIElement(index=3, role="AXMenuBarItem", label="File")) is False
    assert _is_window_content_element(UIElement(index=4, role="AXMenu", label="")) is False


def test_structured_cua_screen_bounds_are_normalized_to_window_local():
    from tools.computer_use.backend import UIElement
    from tools.computer_use.cua_backend import _normalize_structured_screen_bounds

    elements = [UIElement(
        index=0,
        role="AXWindow",
        label="node",
        bounds=(100, 200, 800, 600),
        attributes={"geometry_source": "cua_driver.elements", "bounds_source": "AXPosition+AXSize"},
    )]

    _normalize_structured_screen_bounds(
        elements,
        window_bounds=(100, 200, 800, 600),
        capture_size=(800, 600),
    )

    assert elements[0].bounds == (0, 0, 800, 600)
    assert elements[0].attributes["geometry_status"] == "direct"
    assert elements[0].attributes["geometry_source"] == "cua_driver.elements"
    assert elements[0].attributes["bounds_coordinate_space"] == "window_logical"


def test_ax_payload_can_synthesize_coordinate_backed_elements():
    from tools.computer_use.cua_backend import _elements_from_ax_payload

    payload = {
        "ok": True,
        "roots": [{
            "role": "AXWindow",
            "text": "scratch.txt",
            "position": {"x": 100, "y": 200},
            "size": {"width": 800, "height": 600},
            "children": [
                {"role": "AXMenuBar", "text": "Apple", "position": {"x": 0, "y": 0}, "size": {"width": 100, "height": 24}},
                {"role": "AXTextArea", "text": "body", "position": {"x": 120, "y": 250}, "size": {"width": 700, "height": 500}},
            ],
        }],
    }

    elements = _elements_from_ax_payload(
        payload,
        window_bounds=(100, 200, 800, 600),
        capture_size=(800, 600),
    )

    assert [e.index for e in elements] == [1, 2]
    assert elements[0].role == "AXWindow"
    assert elements[1].role == "AXTextArea"
    assert elements[1].bounds == (20, 50, 700, 500)
    assert elements[1].attributes["action_backend"] == "coordinate"
    assert elements[1].attributes["synthetic_element"] is True


def test_empty_label_role_matches_use_stable_order():
    from tools.computer_use.backend import UIElement
    from tools.computer_use.cua_backend import _merge_ax_geometry

    elements = [
        UIElement(index=2, role="AXButton", label="", attributes={"bounds_available": False}),
        UIElement(index=3, role="AXButton", label="", attributes={"bounds_available": False}),
    ]
    payload = {
        "ok": True,
        "roots": [{
            "role": "AXWindow",
            "children": [
                {"role": "AXButton", "text": "", "position": {"x": 10, "y": 10}, "size": {"width": 20, "height": 20}},
                {"role": "AXButton", "text": "", "position": {"x": 40, "y": 10}, "size": {"width": 20, "height": 20}},
            ],
        }],
    }

    merged = _merge_ax_geometry(
        elements,
        payload,
        window_bounds=(0, 0, 100, 100),
        capture_size=(100, 100),
    )

    assert merged == 2
    assert [element.index for element in elements] == [2, 3]
    assert [element.bounds for element in elements] == [(10, 10, 20, 20), (40, 10, 20, 20)]


def test_ax_helper_failure_does_not_break_capture(monkeypatch):
    import tools.computer_use.cua_backend as cua_backend

    def fail_helper(**kwargs):
        raise RuntimeError("helper exploded")

    monkeypatch.setattr(cua_backend, "_run_ax_geometry_helper", fail_helper)
    fake = _FakeCuaSession({
        "list_windows": [_list_windows_reply({
            "app_name": "TextEdit",
            "pid": 123,
            "window_id": 456,
            "title": "scratch.txt",
            "is_on_screen": True,
            "z_index": 7,
            "bounds": {"x": 10, "y": 20, "width": 640, "height": 480},
        })],
        "get_window_state": [_state_reply()],
    })
    backend = _cua_backend_with(fake)

    cap = backend.capture(mode="ax", app="TextEdit")

    assert cap.elements[0].index == 0
    assert cap.elements[1].index == 7
    assert cap.elements[1].bounds == (0, 0, 0, 0)
    assert cap.elements[1].attributes["geometry_status"] == "missing"
    assert cap.elements[1].attributes["geometry_helper_error"] == "helper exploded"


def test_capture_summary_prints_real_bounds_and_geometry_source():
    from tools.computer_use import tool as cu_tool
    from tools.computer_use.backend import CaptureResult, UIElement

    cap = CaptureResult(
        mode="ax",
        width=100,
        height=50,
        elements=[
            UIElement(
                index=7,
                role="AXButton",
                label="OK",
                bounds=(10, 20, 30, 40),
                attributes={
                    "bounds_available": True,
                    "geometry_status": "derived",
                    "geometry_source": "hermes_ax_map",
                },
            )
        ],
    )

    out = json.loads(cu_tool._capture_response(cap))

    assert "#7 AXButton 'OK' @ (10, 20, 30, 40) source=hermes_ax_map" in out["summary"]
    assert "bounds unavailable" not in out["summary"]

"""The browser tools and arguments the extension controller already supports, on the built-in browser.

One schema serves both lanes. Where the built-in browser (agent-browser) can do an action it does, through the
same registry handler the agent calls; where it cannot, the call returns a clear "not available in this
browser" error naming the cause and the remedy. Nothing is dropped or ignored silently.
"""

import base64
import json

import pytest

from gateway.browser_control_broker import BROWSER_CONTROL_CAPABILITIES
from tools import browser_tool
from tools import browser_tool_eval_policy as bt_eval_policy
from tools import browser_tool_install as bt_install
from tools import browser_tool_session as bt_session
from tools.registry import registry
from toolsets import _HERMES_CORE_TOOLS

NEW_TOOLS = (
    "browser_batch", "browser_fill", "browser_select", "browser_hover", "browser_drag", "browser_scroll_to",
    "browser_tabs", "browser_tab_activate", "browser_tab_create", "browser_tab_close", "browser_tab_group",
    "browser_tab_ungroup", "browser_screenshot",
)
# Tools the built-in browser cannot do: offered only when a bound controller advertises them.
EXTENSION_ONLY = (
    "browser_tabs", "browser_tab_activate", "browser_tab_create", "browser_tab_close", "browser_tab_group",
    "browser_tab_ungroup",
)
# A 1x1 PNG.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==")


@pytest.fixture
def browser(monkeypatch):
    """The built-in browser with a fake command runner; returns the list of ``(command, args)`` it ran."""
    commands = []

    def run(task_id, command, args, **_kwargs):
        commands.append((command, list(args)))
        return {"success": True, "data": {}}

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: False)
    monkeypatch.setattr(bt_session, "_run_browser_command", run)
    return commands


def _call(tool, args):
    """Through the registry handler, as the agent calls it (extension control off: the built-in browser runs)."""
    return json.loads(registry.get_entry(tool).handler(dict(args), task_id="task-1"))


def test_every_new_tool_is_registered_allowlisted_and_has_a_legacy_function():
    schemas = {s["name"] for s in browser_tool.BROWSER_TOOL_SCHEMAS}
    for name in NEW_TOOLS:
        assert name in schemas, name
        assert callable(getattr(browser_tool, name)), f"{name} needs a module-level legacy function"
        assert registry.get_entry(name).toolset == "browser"
        assert name in _HERMES_CORE_TOOLS
        assert name in BROWSER_CONTROL_CAPABILITIES


def test_every_declared_argument_reaches_the_legacy_handler():
    """_fallback_call hands the legacy function only the arguments its table row declares."""
    schemas = {s["name"]: s for s in browser_tool.BROWSER_TOOL_SCHEMAS}
    for name, _emoji, _check, defaults, *_extra in browser_tool._BROWSER_TOOL_TABLE:
        declared = set(schemas[name]["parameters"]["properties"])
        assert declared <= set(defaults), f"{name} would silently drop {sorted(declared - set(defaults))}"


def test_schema_ranges_match_the_extension():
    schemas = {s["name"]: s["parameters"]["properties"] for s in browser_tool.BROWSER_TOOL_SCHEMAS}
    assert schemas["browser_scroll"]["direction"]["enum"] == ["up", "down", "left", "right"]
    assert (schemas["browser_scroll"]["amount"]["minimum"], schemas["browser_scroll"]["amount"]["maximum"]) == (100, 5000)
    assert schemas["browser_press"]["modifiers"]["items"]["enum"] == ["Alt", "Control", "Meta", "Shift"]
    assert schemas["browser_type"]["clear"]["default"] is True
    assert schemas["browser_batch"]["actions"]["maxItems"] == 20
    assert schemas["browser_click"]["x"]["type"] == schemas["browser_click"]["y"]["type"] == "number"
    click = next(s for s in browser_tool.BROWSER_TOOL_SCHEMAS if s["name"] == "browser_click")
    assert click["parameters"]["required"] == []  # a ref or both coordinates: checked in code, not by the schema


@pytest.mark.parametrize(
    "tool, args, command, command_args, reported",
    [
        ("browser_fill", {"ref": "e1", "value": "hello"}, "fill", ["@e1", "hello"], {"filled": "hello"}),
        ("browser_select", {"ref": "@e2", "value": "Two"}, "select", ["@e2", "Two"], {"selected": "@e2", "value": "Two"}),
        ("browser_hover", {"ref": "@e3"}, "hover", ["@e3"], {"hovered": "@e3"}),
        ("browser_drag", {"ref": "@e3", "to_ref": "@e4"}, "drag", ["@e3", "@e4"], {"dragged": "@e3", "dropped_on": "@e4"}),
        ("browser_scroll_to", {"ref": "@e9"}, "scrollintoview", ["@e9"], {"scrolled_to": "@e9"}),
        ("browser_scroll", {"direction": "left", "amount": 250}, "scroll", ["left", "250"], {"scrolled": "left"}),
        ("browser_scroll", {"direction": "down"}, "scroll", ["down", "500"], {"scrolled": "down"}),
        ("browser_press", {"key": "a", "modifiers": ["control", "Shift"]}, "press", ["Control+Shift+a"],
         {"pressed": "Control+Shift+a"}),
        ("browser_type", {"ref": "@e1", "text": "more", "clear": False}, "type", ["@e1", "more"], {"typed": "more"}),
        ("browser_type", {"ref": "@e1", "text": "all new"}, "fill", ["@e1", "all new"], {"typed": "all new"}),
    ],
)
def test_actions_the_built_in_browser_can_do_run_with_their_arguments(browser, tool, args, command, command_args, reported):
    out = _call(tool, args)

    assert browser == [(command, command_args)]
    assert out["success"] is True
    assert reported.items() <= out.items()


def test_a_coordinate_click_moves_presses_and_releases_at_the_point(browser):
    out = _call("browser_click", {"x": 12, "y": 34.5})

    assert browser == [("mouse", ["move", "12", "34.5"]), ("mouse", ["down"]), ("mouse", ["up"])]
    assert out == {"success": True, "clicked": "(12, 34.5)"}


def test_a_ref_wins_over_coordinates_and_clicks_the_element(browser):
    _call("browser_click", {"ref": "@e5", "x": 1, "y": 2})

    assert browser == [("click", ["@e5"])]


@pytest.mark.parametrize(
    "tool, args",
    [
        ("browser_tabs", {}),
        ("browser_tab_activate", {"tab_id": 3}),
        ("browser_tab_create", {"url": "https://example.test"}),
        ("browser_tab_close", {"tab_id": 3}),
        ("browser_tab_group", {"tab_ids": [3, 4]}),
        ("browser_tab_ungroup", {"tab_ids": [3, 4]}),
        ("browser_screenshot", {"ref": "@e1"}),
    ],
)
def test_what_the_built_in_browser_cannot_do_is_a_clear_error_never_a_silent_no_op(browser, tool, args):
    out = _call(tool, args)

    assert out["success"] is False and out["code"] == "unsupported_in_this_browser"
    assert out["error"].startswith(f"{tool} is not available in this browser: ")
    assert "Extension" in out["error"]  # the remedy
    assert browser == []


def test_the_active_backend_that_cannot_do_something_says_so_instead_of_dropping_the_argument(browser, monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(browser_tool, "_camofox", lambda *a: pytest.fail("Camofox must not receive an argument it cannot honour"))

    for tool, args in [
        ("browser_click", {"x": 1, "y": 2}),
        ("browser_type", {"ref": "@e1", "text": "x", "clear": False}),
        ("browser_scroll", {"direction": "left"}),
        ("browser_scroll", {"direction": "down", "amount": 300}),
        ("browser_press", {"key": "a", "modifiers": ["Control"]}),
        ("browser_fill", {"ref": "@e1", "value": "x"}),
        ("browser_select", {"ref": "@e1", "value": "x"}),
        ("browser_hover", {"ref": "@e1"}),
        ("browser_drag", {"ref": "@e1", "to_ref": "@e2"}),
        ("browser_scroll_to", {"ref": "@e1"}),
        ("browser_screenshot", {}),
    ]:
        out = _call(tool, args)
        assert out["success"] is False and "Camofox" in out["error"], (tool, out)
    assert browser == []


@pytest.mark.parametrize(
    "tool, args, fragment",
    [
        ("browser_click", {}, "needs 'ref'"),
        ("browser_click", {"x": 5}, "needs both 'x' and 'y'"),
        ("browser_click", {"ref": "@e1", "y": "high"}, "'y' must be a number"),
        ("browser_scroll", {"direction": "diagonal"}, "'direction' must be one of up, down, left, right"),
        ("browser_scroll", {"direction": "down", "amount": 99}, "from 100 to 5000"),
        ("browser_scroll", {"direction": "down", "amount": 5001}, "from 100 to 5000"),
        ("browser_press", {"key": "a", "modifiers": ["Hyper"]}, "unknown modifier 'Hyper'"),
        ("browser_press", {"key": ""}, "needs 'key'"),
        ("browser_type", {"ref": "@e1", "text": "x", "clear": "no"}, "'clear' must be true or false"),
        ("browser_fill", {"ref": "@e1"}, "needs 'value'"),
        ("browser_drag", {"ref": "@e1"}, "needs 'to_ref'"),
        ("browser_tab_activate", {"tab_id": 0}, "'tab_id' must be a tab id"),
        ("browser_tab_close", {}, "needs 'tab_id'"),
        ("browser_tab_create", {"url": ""}, "needs 'url'"),
        ("browser_tab_group", {}, "needs 'tab_ids'"),
        ("browser_tab_group", {"tab_ids": []}, "'tab_ids' must be a non-empty list of tab ids"),
        ("browser_tab_group", {"tab_ids": 3}, "'tab_ids' must be a non-empty list of tab ids"),
        ("browser_tab_group", {"tab_ids": [3, 0]}, "'tab_ids' must be a non-empty list of tab ids"),
        ("browser_tab_group", {"tab_ids": [3, "4"]}, "'tab_ids' must be a non-empty list of tab ids"),
        ("browser_tab_group", {"tab_ids": [3, 3]}, "'tab_ids' must list each tab once"),
        ("browser_tab_group", {"tab_ids": [3], "tab_id": 3}, "does not take tab_id"),
        ("browser_tab_ungroup", {}, "needs 'tab_ids'"),
        ("browser_tab_ungroup", {"tab_ids": [3, 3]}, "'tab_ids' must list each tab once"),
        ("browser_tab_ungroup", {"tab_ids": [3], "tab_id": 3}, "does not take tab_id"),
    ],
)
def test_a_bad_argument_is_refused_before_anything_runs(browser, tool, args, fragment):
    out = _call(tool, args)

    assert out["success"] is False and out["error"].startswith(f"{tool}: ") and fragment in out["error"]
    assert browser == []


@pytest.mark.parametrize("tool", ("browser_tab_group", "browser_tab_ungroup"))
def test_the_tab_group_tools_name_their_tabs_in_tab_ids_not_tab_id(tool):
    schema = next(s for s in browser_tool.BROWSER_TOOL_SCHEMAS if s["name"] == tool)
    properties = schema["parameters"]["properties"]

    assert set(properties) == {"tab_ids"} and schema["parameters"]["required"] == ["tab_ids"]
    assert properties["tab_ids"]["type"] == "array" and properties["tab_ids"]["minItems"] == 1
    assert properties["tab_ids"]["uniqueItems"] is True
    assert properties["tab_ids"]["items"] == {"type": "integer", "minimum": 1}
    assert "lease_not_owned" in schema["description"] and "never switches the tab" in schema["description"]


def test_the_built_in_browser_keeps_its_default_scroll_and_clear_behaviour(browser):
    _call("browser_scroll", {"direction": "up"})
    _call("browser_type", {"ref": "@e1", "text": "x"})

    assert browser == [("scroll", ["up", "500"]), ("fill", ["@e1", "x"])]


# --- who is offered which tool ------------------------------------------------------------------------------


@pytest.mark.parametrize("tool", EXTENSION_ONLY)
def test_extension_only_tools_need_a_controller_that_advertises_them(monkeypatch, tool):
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: True)
    advertised = set()
    monkeypatch.setattr(browser_tool, "extension_controller_available", lambda action: action in advertised)
    check = registry.get_entry(tool).check_fn

    assert check() is False  # a built-in-only install never sees a tool that can only fail
    advertised.add(tool)
    assert check() is True


@pytest.mark.parametrize("tool", sorted(set(NEW_TOOLS) - set(EXTENSION_ONLY)))
def test_tools_the_built_in_browser_can_run_follow_the_usual_availability(monkeypatch, tool):
    monkeypatch.setattr(browser_tool, "extension_controller_available", lambda action: False)
    check = registry.get_entry(tool).check_fn

    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: True)
    assert check() is True
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: False)
    assert check() is False
    monkeypatch.setattr(browser_tool, "extension_controller_available", lambda action: action == tool)
    assert check() is True


# --- screenshots --------------------------------------------------------------------------------------------


def _data_url(image=PNG):
    return "data:image/png;base64," + base64.b64encode(image).decode()


def test_an_extension_screenshot_becomes_the_native_image_envelope_not_text(monkeypatch):
    monkeypatch.setattr("tools.vision_tools._should_use_native_vision_fast_path", lambda: True)
    raw = json.dumps({"status": "captured", "dataUrl": _data_url()})

    result = browser_tool._extension_screenshot_result(raw)

    assert isinstance(result, dict) and result["_multimodal"] is True
    assert [part["type"] for part in result["content"]] == ["text", "image_url"]
    assert "base64" not in json.dumps(result["content"][0]) and "base64" not in result["text_summary"]
    assert (result["meta"]["image_url"]).endswith(".png")


def test_an_extension_screenshot_for_a_model_without_images_is_saved_and_named(monkeypatch):
    monkeypatch.setattr("tools.vision_tools._should_use_native_vision_fast_path", lambda: False)

    out = json.loads(browser_tool._extension_screenshot_result(json.dumps({"status": "captured", "dataUrl": _data_url()})))

    assert out["success"] is True and "base64" not in json.dumps(out)
    assert open(out["screenshot_path"], "rb").read() == PNG
    assert "cannot view images" in out["note"]


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({"error": "screenshot_too_large"}),  # an extension error
        json.dumps({"status": "captured"}),
        "not json at all",
        {"_multimodal": True, "content": []},  # the built-in browser's own envelope
    ],
)
def test_anything_that_is_not_an_extension_screenshot_passes_through(raw):
    assert browser_tool._extension_screenshot_result(raw) == raw


@pytest.mark.parametrize("data_url", ["data:image/jpeg;base64,AAAA", "data:image/png;base64,***not-base64***", "http://x/y.png"])
def test_an_unreadable_extension_screenshot_is_an_error_with_a_remedy(data_url):
    out = json.loads(browser_tool._extension_screenshot_result(json.dumps({"dataUrl": data_url})))

    assert out["success"] is False and "Update the Hermes Browser Extension" in out["error"]


def test_an_oversized_extension_screenshot_is_refused_and_writes_no_file(tmp_path, monkeypatch):
    shots = tmp_path / "shots"
    monkeypatch.setattr(browser_tool, "_screenshots_dir", lambda: shots)
    huge = "data:image/png;base64," + "A" * 1_500_000

    out = json.loads(browser_tool._extension_screenshot_result(json.dumps({"dataUrl": huge})))

    assert out["success"] is False and "Update the Hermes Browser Extension" in out["error"]
    assert not shots.exists()


def test_the_built_in_browser_screenshot_goes_through_the_same_native_path(browser, monkeypatch, tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG)
    monkeypatch.setattr("tools.vision_tools._should_use_native_vision_fast_path", lambda: True)
    monkeypatch.setattr(browser_tool, "_blocked_private_page_content", lambda task_id: None)
    monkeypatch.setattr(browser_tool, "_capture_vision_screenshot", lambda task_id, annotate, path, lp: ({}, shot, None))

    result = registry.get_entry("browser_screenshot").handler({}, task_id="task-1")

    assert isinstance(result, dict) and result["_multimodal"] is True
    assert result["meta"]["screenshot_path"] == str(shot)

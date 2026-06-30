from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import browser_tool
from tools.registry import registry
from tui_gateway import browser_desktop_bridge
import model_tools
import toolsets
from agent.tool_guardrails import IDEMPOTENT_TOOL_NAMES


@pytest.fixture(autouse=True)
def _reset_desktop_bridge():
    browser_desktop_bridge.clear_desktop_browser_command_runner()
    yield
    browser_desktop_bridge.clear_desktop_browser_command_runner()


def _disable_non_desktop_browser_paths(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda _session_key: None)
    monkeypatch.setattr(browser_tool, "_get_command_timeout", lambda: 3)


def test_browser_screenshot_is_registered_as_plain_visible_artifact_tool():
    entry = registry.get_entry("browser_screenshot")

    assert entry is not None
    assert entry.toolset == "browser"
    assert entry.schema["name"] == "browser_screenshot"
    assert entry.schema["parameters"]["properties"]["annotate"]["type"] == "boolean"
    assert "without running vision analysis" in entry.schema["description"]
    assert "browser_screenshot" in registry.get_tool_names_for_toolset("browser")
    assert "browser_screenshot" in model_tools._LEGACY_TOOLSET_MAP["browser_tools"]
    assert "browser_screenshot" in toolsets.resolve_toolset("browser")
    assert "browser_screenshot" in IDEMPOTENT_TOOL_NAMES
    acp_source = (browser_tool.Path(__file__).resolve().parents[2] / "acp_adapter" / "tools.py").read_text(encoding="utf-8")
    assert '"browser_screenshot": "read"' in acp_source
    assert 'tool_name == "browser_screenshot"' in acp_source


def test_browser_navigate_uses_visible_desktop_bridge_before_hidden_session_setup(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    calls: list[tuple[str, str, dict]] = []

    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda *_args, **_kwargs: pytest.fail("visible Desktop navigation must not create a hidden browser session first"),
    )


    def runner(session_key: str, command: str, params: dict, *, tab_id=None, timeout=None):
        calls.append((session_key, command, dict(params)))
        if command == "navigate":
            return {"ok": True, "result": {"url": params["url"], "title": "Visible Page"}}
        if command == "snapshot":
            return {
                "ok": True,
                "result": {
                    "title": "Visible Page",
                    "url": "https://example.com/",
                    "text": "Visible body",
                    "elements": [{"ref": "@e0", "label": "button", "text": "Submit"}],
                },
            }
        raise AssertionError(f"unexpected command {command}")

    browser_desktop_bridge.set_desktop_browser_command_runner(runner)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("headless browser path should not be used when desktop bridge is active"),
    )

    result = json.loads(browser_tool.browser_navigate("example.com", task_id="session-1"))

    assert result["success"] is True
    assert result["url"] == "https://example.com/"
    assert result["title"] == "Visible Page"
    assert "Visible body" in result["snapshot"]
    assert "[@e0] button: Submit" in result["snapshot"]
    assert result["element_count"] == 1
    assert calls == [
        ("session-1", "navigate", {"url": "https://example.com/"}),
        ("session-1", "snapshot", {"full": False}),
    ]


def test_browser_snapshot_click_type_scroll_back_press_route_to_visible_desktop(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    calls: list[tuple[str, str, dict]] = []

    def runner(session_key: str, command: str, params: dict, *, tab_id=None, timeout=None):
        calls.append((session_key, command, dict(params)))
        if command == "snapshot":
            return {
                "ok": True,
                "result": {
                    "title": "Visible",
                    "url": "https://app.test/",
                    "text": "App text",
                    "elements": [{"ref": "@e2", "stableRef": "@sabc", "label": "input", "text": "Email"}],
                },
            }
        return {"ok": True, "result": {"ok": True}}

    browser_desktop_bridge.set_desktop_browser_command_runner(runner)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("headless browser path should not be used when desktop bridge is active"),
    )

    snapshot = json.loads(browser_tool.browser_snapshot(task_id="session-2"))
    click = json.loads(browser_tool.browser_click("@e2", task_id="session-2"))
    hover = json.loads(browser_tool.browser_hover("@sabc", task_id="session-2"))
    double_click = json.loads(browser_tool.browser_double_click("@sabc", task_id="session-2"))
    right_click = json.loads(browser_tool.browser_right_click("@sabc", task_id="session-2"))
    typed = json.loads(browser_tool.browser_type("@e2", "cloud@example.com", task_id="session-2"))
    scroll = json.loads(browser_tool.browser_scroll("right", amount=750, task_id="session-2"))
    back = json.loads(browser_tool.browser_back(task_id="session-2"))
    press = json.loads(browser_tool.browser_press("Enter", task_id="session-2"))

    assert snapshot["success"] is True
    assert snapshot["element_count"] == 1
    assert "@sabc" in snapshot["snapshot"]
    assert click["success"] is True
    assert click["clicked"] == "@e2"
    assert hover["success"] is True
    assert hover["hovered"] == "@sabc"
    assert double_click["success"] is True
    assert double_click["double_clicked"] == "@sabc"
    assert right_click["success"] is True
    assert right_click["right_clicked"] == "@sabc"
    assert typed["success"] is True
    assert typed["typed"] == "cloud@example.com"
    assert typed["element"] == "@e2"
    assert scroll["success"] is True
    assert scroll["scrolled"] == "right"
    assert back["success"] is True
    assert back["navigated_back"] is True
    assert press["success"] is True
    assert press["pressed"] == "Enter"
    assert calls == [
        ("session-2", "snapshot", {"full": False}),
        ("session-2", "clickRef", {"ref": "@e2"}),
        ("session-2", "hoverRef", {"ref": "@sabc"}),
        ("session-2", "doubleClickRef", {"ref": "@sabc"}),
        ("session-2", "rightClickRef", {"ref": "@sabc"}),
        ("session-2", "fillRef", {"ref": "@e2", "text": "cloud@example.com"}),
        ("session-2", "scroll", {"direction": "right", "amount": 750}),
        ("session-2", "goBack", {}),
        ("session-2", "press", {"key": "Enter"}),
    ]


def test_visible_desktop_bridge_error_does_not_fall_back_to_second_browser_reality(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    browser_desktop_bridge.set_desktop_browser_command_runner(
        lambda *_args, **_kwargs: {"ok": False, "error": "No visible browser tab is bound for this session"}
    )
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("must not fall back to a second browser reality after desktop bridge error"),
    )

    result = json.loads(browser_tool.browser_snapshot(task_id="session-3"))

    assert result["success"] is False
    assert "No visible browser tab" in result["error"]


def test_browser_console_expression_and_images_use_visible_desktop(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    calls: list[tuple[str, str, dict]] = []

    def runner(session_key: str, command: str, params: dict, *, tab_id=None, timeout=None):
        calls.append((session_key, command, dict(params)))
        if command == "evaluate":
            return {"ok": True, "result": "Visible Title"}
        if command == "getImages":
            return {"ok": True, "result": [{"src": "https://example.com/a.png", "alt": "A"}]}
        raise AssertionError(command)

    browser_desktop_bridge.set_desktop_browser_command_runner(runner)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("headless eval/images path should not be used when desktop bridge is active"),
    )

    console = json.loads(browser_tool.browser_console(expression="document.title", task_id="session-visible"))
    images = json.loads(browser_tool.browser_get_images(task_id="session-visible"))

    assert console["success"] is True
    assert console["result"] == "Visible Title"
    assert console["method"] == "desktop_visible"
    assert images["success"] is True
    assert images["count"] == 1
    assert images["images"][0]["alt"] == "A"
    assert calls == [
        ("session-visible", "evaluate", {"expression": "document.title"}),
        ("session-visible", "getImages", {}),
    ]


def test_browser_console_and_network_buffers_use_visible_desktop(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    calls: list[tuple[str, str, dict]] = []

    def runner(session_key: str, command: str, params: dict, *, tab_id=None, timeout=None):
        calls.append((session_key, command, dict(params)))
        if command == "getConsole":
            return {"ok": True, "result": {"messages": [
                {"level": "warn", "message": "careful", "source": "console", "url": "https://example.com"},
                {"level": "error", "message": "boom", "source": "exception", "url": "https://example.com/app.js"},
            ]}}
        if command == "clearConsole":
            return {"ok": True, "result": {"ok": True}}
        if command == "getNetwork":
            return {"ok": True, "result": {"events": [{"type": "response", "url": "https://example.com/api", "status": 500}]}}
        if command == "clearNetwork":
            return {"ok": True, "result": {"ok": True}}
        raise AssertionError(command)

    browser_desktop_bridge.set_desktop_browser_command_runner(runner)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("headless console/network path should not be used when desktop bridge is active"),
    )

    console = json.loads(browser_tool.browser_console(clear=True, task_id="session-visible"))
    network = json.loads(browser_tool.browser_network(clear=True, task_id="session-visible"))

    assert console["success"] is True
    assert console["total_messages"] == 1
    assert console["total_errors"] == 1
    assert console["console_messages"][0]["text"] == "careful"
    assert console["js_errors"][0]["message"] == "boom"
    assert network["success"] is True
    assert network["total_events"] == 1
    assert network["events"][0]["status"] == 500
    assert calls == [
        ("session-visible", "getConsole", {}),
        ("session-visible", "clearConsole", {}),
        ("session-visible", "getNetwork", {}),
        ("session-visible", "clearNetwork", {}),
    ]


def test_browser_vision_saves_visible_desktop_screenshot_file(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    png_data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )

    def runner(session_key: str, command: str, params: dict, *, tab_id=None, timeout=None):
        assert session_key == "session-vision"
        assert command == "screenshot"
        return {
            "ok": True,
            "result": {
                "dataUrl": png_data_url,
                "url": "https://example.com/",
                "title": "Visible Screenshot",
            },
        }

    browser_desktop_bridge.set_desktop_browser_command_runner(runner)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("headless screenshot path should not be used when desktop bridge is active"),
    )
    monkeypatch.setattr(
        browser_tool,
        "call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="visible screenshot analyzed"))]
        ),
    )

    result = json.loads(browser_tool.browser_vision("what is visible?", task_id="session-vision"))

    assert result["success"] is True
    assert result["analysis"] == "visible screenshot analyzed"
    assert result["browser_surface"] == "desktop_visible"
    assert result["screenshot_path"].endswith(".png")
    assert browser_tool.Path(result["screenshot_path"]).exists()
    history_path = browser_tool.Path(result["screenshot_path"]).parent / "browser_screenshot_history.json"
    assert history_path.exists()
    assert "Visible Screenshot" in history_path.read_text(encoding="utf-8")


def test_browser_screenshot_saves_visible_desktop_artifact_without_vision_analysis(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    calls: list[tuple[str, str, dict]] = []
    png_data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )

    def runner(session_key: str, command: str, params: dict, *, tab_id=None, timeout=None):
        calls.append((session_key, command, dict(params)))
        return {
            "ok": True,
            "result": {
                "dataUrl": png_data_url,
                "url": "https://example.com/plain-shot",
                "title": "Plain Screenshot",
            },
        }

    browser_desktop_bridge.set_desktop_browser_command_runner(runner)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("browser_screenshot must not spawn a hidden browser fallback"),
    )
    monkeypatch.setattr(
        browser_tool,
        "call_llm",
        lambda **_kwargs: pytest.fail("browser_screenshot must not invoke vision analysis"),
    )

    result = json.loads(browser_tool.browser_screenshot(annotate=True, task_id="session-shot"))

    assert result == {
        "success": True,
        "screenshot_path": result["screenshot_path"],
        "url": "https://example.com/plain-shot",
        "title": "Plain Screenshot",
        "browser_surface": "desktop_visible",
    }
    assert result["screenshot_path"].endswith(".png")
    assert browser_tool.Path(result["screenshot_path"]).exists()
    history_path = browser_tool.Path(result["screenshot_path"]).parent / "browser_screenshot_history.json"
    assert "Plain Screenshot" in history_path.read_text(encoding="utf-8")
    assert calls == [("session-shot", "screenshot", {"annotate": True, "full": True})]


def test_browser_screenshot_fails_closed_without_visible_browserpane(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("browser_screenshot must not fall back to a hidden browser when BrowserPane is disabled"),
    )

    result = json.loads(browser_tool.browser_screenshot(task_id="session-missing"))

    assert result["success"] is False
    assert result["browser_surface"] == "desktop_visible"
    assert "No visible Desktop BrowserPane" in result["error"]


def test_browser_screenshot_fails_closed_on_visible_bridge_errors_and_capture_failures(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("visible BrowserPane errors must not fall back to a hidden browser"),
    )

    browser_desktop_bridge.set_desktop_browser_command_runner(
        lambda *_args, **_kwargs: {"ok": False, "error": "agent control is paused"}
    )
    paused = json.loads(browser_tool.browser_screenshot(task_id="session-paused"))
    assert paused["success"] is False
    assert "agent control is paused" in paused["error"]

    browser_desktop_bridge.set_desktop_browser_command_runner(
        lambda *_args, **_kwargs: {"ok": True, "result": {"dataUrl": "not-an-image"}}
    )
    invalid = json.loads(browser_tool.browser_screenshot(task_id="session-invalid"))
    assert invalid["success"] is False
    assert "supported image data URL" in invalid["error"]


def test_browser_design_and_accessibility_tools_use_visible_desktop(monkeypatch):
    _disable_non_desktop_browser_paths(monkeypatch)
    calls: list[tuple[str, str, dict]] = []

    def runner(session_key: str, command: str, params: dict, *, tab_id=None, timeout=None):
        calls.append((session_key, command, dict(params)))
        if command == "inspectElement":
            return {"ok": True, "result": {"element": {"tag": "button", "text": "Buy now", "styles": {"color": "red"}}}}
        if command == "designHandoff":
            return {"ok": True, "result": {
                "mode": "agent-mediated",
                "unsafeDirectDomMutation": False,
                "prompt": "Make CTA stronger",
                "selected": [{"tag": "button", "text": "Buy now"}],
            }}
        if command == "accessibilityAudit":
            return {"ok": True, "result": {
                "summary": {"error": 1, "warning": 0},
                "findings": [{"rule": "image-alt", "severity": "error", "message": "Image missing alt"}],
            }}
        raise AssertionError(command)

    browser_desktop_bridge.set_desktop_browser_command_runner(runner)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: pytest.fail("headless browser path should not be used for design/a11y visible tools"),
    )

    inspection = json.loads(browser_tool.browser_inspect_element("@scta", task_id="session-design"))
    handoff = json.loads(browser_tool.browser_design_handoff("Make CTA stronger", refs=["@scta"], task_id="session-design"))
    audit = json.loads(browser_tool.browser_accessibility_audit(task_id="session-design"))

    assert inspection["success"] is True
    assert inspection["element"]["tag"] == "button"
    assert handoff["success"] is True
    assert handoff["mode"] == "agent-mediated"
    assert handoff["unsafeDirectDomMutation"] is False
    assert audit["success"] is True
    assert audit["summary"]["error"] == 1
    assert calls == [
        ("session-design", "inspectElement", {"ref": "@scta"}),
        ("session-design", "designHandoff", {"goal": "Make CTA stronger", "refs": ["@scta"]}),
        ("session-design", "accessibilityAudit", {}),
    ]

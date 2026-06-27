import json
from unittest.mock import patch

from tools.browser_autonomy import assess_browser_state, browser_autonomy_check


def test_assess_human_verification_requires_handoff():
    result = assess_browser_state(
        title="Security check",
        snapshot='text: Please verify you are human\nbutton "I am not a robot" [ref=e2]',
    )

    assert result["status"] == "needs_human"
    assert result["kind"] == "human_verification"
    assert result["policy"] == "do_not_solve_or_bypass_captcha"
    assert result["matches"]
    assert "official API" in " ".join(result["safe_next_steps"])


def test_assess_access_wall_blocks_autonomy():
    result = assess_browser_state(
        title="Access Denied",
        snapshot="text: Request blocked due to automated traffic",
    )

    assert result["status"] == "blocked"
    assert result["kind"] == "bot_or_access_wall"
    assert result["policy"] == "do_not_evade_access_controls"


def test_assess_normal_page_ready():
    result = assess_browser_state(
        title="Example Domain",
        snapshot='heading "Example Domain" [ref=e1]\nlink "More information" [ref=e2]',
    )

    assert result["status"] == "ready"
    assert result["matches"] == []


def test_browser_autonomy_check_tool_shape():
    result = json.loads(
        browser_autonomy_check(
            title="Just a moment",
            snapshot="text: Checking your browser before accessing the site",
        )
    )

    assert result["success"] is True
    assert result["autonomy"]["status"] == "needs_human"


def test_browser_navigate_attaches_autonomy_metadata(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda task_id: {
            "session_name": "test",
            "_first_nav": False,
            "features": {"local": True, "proxies": True},
        },
    )

    with patch(
        "tools.browser_tool._run_browser_command",
        side_effect=[
            {
                "success": True,
                "data": {"title": "Security Check", "url": "https://example.test/"},
            },
            {
                "success": True,
                "data": {
                    "snapshot": 'text: Verify you are human\nbutton "I am not a robot" [ref=e1]',
                    "refs": {"e1": {}},
                },
            },
        ],
    ):
        response = json.loads(
            browser_tool.browser_navigate("https://example.test", task_id="autonomy-nav")
        )

    assert response["success"] is True
    assert response["autonomy"]["status"] == "needs_human"
    assert response["human_handoff_required"] is True
    assert "bypass" not in response["bot_detection_warning"].lower()
    browser_tool._last_active_session_key.pop("autonomy-nav", None)


def test_browser_snapshot_attaches_autonomy_metadata(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)

    with patch(
        "tools.browser_tool._run_browser_command",
        return_value={
            "success": True,
            "data": {
                "snapshot": 'heading "Access denied"\ntext: automated traffic detected',
                "refs": {},
            },
        },
    ):
        response = json.loads(browser_tool.browser_snapshot(task_id="autonomy-snap"))

    assert response["success"] is True
    assert response["autonomy"]["status"] == "blocked"
    assert response["autonomy"]["kind"] == "bot_or_access_wall"


def test_browser_autonomy_tool_is_in_browser_toolset():
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
    from tools.registry import registry
    import tools.browser_autonomy  # noqa: F401

    assert "browser_autonomy_check" in TOOLSETS["browser"]["tools"]
    assert "browser_autonomy_check" in _HERMES_CORE_TOOLS
    assert "browser_autonomy_check" in registry._tools


def test_browser_vision_schema_uses_handoff_language():
    from tools.browser_tool import BROWSER_TOOL_SCHEMAS

    schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_vision")
    description = schema["description"].lower()

    assert "do not solve or bypass" in description
    assert "human handoff" in description

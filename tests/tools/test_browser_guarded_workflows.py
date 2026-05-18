"""Tests for guarded browser inspection workflows."""

import json
from pathlib import Path


def test_guarded_allowlist_empty_blocks():
    import tools.browser_tool as bt

    allowed, reason = bt._url_matches_browser_allowlist("https://example.com", [])

    assert allowed is False
    assert "allowlisted_domains is empty" in reason


def test_guarded_allowlist_exact_and_wildcard_hosts():
    import tools.browser_tool as bt

    assert bt._url_matches_browser_allowlist("https://example.com/path", ["example.com"])[0] is True
    assert bt._url_matches_browser_allowlist("https://app.example.com", ["example.com"])[0] is True
    assert bt._url_matches_browser_allowlist("https://app.example.com", ["*.example.com"])[0] is True
    assert bt._url_matches_browser_allowlist("https://example.com", ["*.example.com"])[0] is False
    assert bt._url_matches_browser_allowlist("https://evil.com", ["example.com"])[0] is False


def test_browser_inspect_page_writes_screenshot_and_action_log(monkeypatch, tmp_path):
    import tools.browser_tool as bt

    cfg = {
        "enabled": True,
        "allowlisted_domains": ["example.com"],
        "action_log_dir": str(tmp_path),
        "dry_run": False,
        "approval_required": True,
        "sensitive_action_keywords": list(bt._SENSITIVE_BROWSER_ACTION_DEFAULTS),
    }
    monkeypatch.setattr(bt, "_get_guarded_workflow_config", lambda: cfg)
    monkeypatch.setattr(
        bt,
        "browser_navigate",
        lambda url, task_id=None: json.dumps({"success": True, "url": url, "title": "Example"}),
    )
    monkeypatch.setattr(
        bt,
        "browser_snapshot",
        lambda full=False, task_id=None, user_task=None: json.dumps({
            "success": True,
            "snapshot": '- link "Docs" [ref=e1]\n- button "Refresh" [ref=e2]\n- text "Status: green"',
            "element_count": 2,
        }),
    )
    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(bt, "_get_command_timeout", lambda: 5)

    def fake_run_browser_command(task_id, command, args=None, timeout=None, _engine_override=None):
        assert command == "screenshot"
        screenshot = Path(args[-1])
        screenshot.write_bytes(b"fake png")
        return {"success": True, "data": {"path": str(screenshot)}}

    monkeypatch.setattr(bt, "_run_browser_command", fake_run_browser_command)

    result = json.loads(bt.browser_inspect_page(
        "https://example.com/dashboard",
        extract_fields=["Status"],
        task_id="test-task",
    ))

    assert result["success"] is True
    assert Path(result["screenshot_path"]).exists()
    assert Path(result["action_log_path"]).exists()
    actions = [json.loads(line)["action"] for line in Path(result["action_log_path"]).read_text().splitlines()]
    assert actions == ["start", "navigate", "snapshot", "screenshot", "complete"]
    assert result["extracted_fields"]["Status"] == '- text "Status: green"'
    assert any("Docs" in item for item in result["actionable_items"])


def test_sensitive_click_requires_live_approval_when_noninteractive(monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(
        bt,
        "_get_guarded_workflow_config",
        lambda: {
            "approval_required": True,
            "sensitive_action_keywords": ["delete"],
        },
    )
    monkeypatch.setattr(
        bt,
        "_run_browser_command",
        lambda task_id, command, args=None, timeout=None, _engine_override=None: {
            "success": True,
            "data": {"snapshot": '- button "Delete account" [ref=e9]'},
        },
    )

    blocked = bt._guard_sensitive_browser_action("click @e9", "task", ref="@e9")

    assert blocked is not None
    assert blocked["success"] is False
    assert "requires explicit live approval" in blocked["error"]
    assert "Delete account" in blocked["element_label"]


def test_browser_inspect_page_schema_registered():
    from tools.browser_tool import BROWSER_TOOL_SCHEMAS

    names = {schema["name"] for schema in BROWSER_TOOL_SCHEMAS}
    assert "browser_inspect_page" in names

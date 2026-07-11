import json

import pytest

from tools.desktop_browser_tool import (
    DesktopBrowserProtocolError,
    build_desktop_browser_action,
    desktop_browser_approval_reason,
    execute_desktop_browser_tool,
    normalize_desktop_browser_snapshot,
    validate_desktop_browser_navigation,
)


def _snapshot():
    return {
        "capturedAt": 123,
        "elements": [
            {
                "ariaLabel": "发布评论",
                "hermesRef": "snapshot-1:47",
                "index": 47,
                "role": "button",
                "tag": "button",
                "text": "发布",
                "value": "must-not-reach-the-model",
                "visible": True,
            },
            {
                "index": 48,
                "placeholder": "密码",
                "tag": "input",
                "type": "password",
                "value": "hunter2",
                "visible": True,
            },
        ],
        "headings": [{"level": 1, "text": "商家后台"}],
        "ok": True,
        "tables": [{"caption": "线索", "headers": ["姓名", "状态"], "rows": [["张三", "待回复"]]}],
        "text": "订单\n客户消息",
        "title": "抖音商家后台",
        "url": "https://fxg.jinritemai.com/im",
    }


def test_snapshot_uses_stable_refs_and_drops_form_values():
    result, state = normalize_desktop_browser_snapshot(_snapshot())

    assert result["success"] is True
    assert result["url"] == "https://fxg.jinritemai.com/im"
    assert "@e0 button" in result["snapshot"]
    assert "@e1 input" in result["snapshot"]
    assert "H1: 商家后台" in result["snapshot"]
    assert "Table: 线索" in result["snapshot"]
    assert "姓名 | 状态" in result["snapshot"]
    assert "订单\n客户消息" in result["snapshot"]
    assert "hunter2" not in str(result)
    assert "must-not-reach-the-model" not in str(result)
    assert state["refs"]["@e0"] == {
        "ariaLabel": "发布评论",
        "hermesRef": "snapshot-1:47",
        "role": "button",
        "tag": "button",
        "text": "发布",
    }


def test_action_resolves_ref_to_fingerprint_and_snapshot_url():
    _, state = normalize_desktop_browser_snapshot(_snapshot())

    action = build_desktop_browser_action("browser_click", {"ref": "@e0"}, state)

    assert action == {
        "expectedUrl": "https://fxg.jinritemai.com/im",
        "kind": "click",
        "target": {
            "ariaLabel": "发布评论",
            "hermesRef": "snapshot-1:47",
            "role": "button",
            "tag": "button",
            "text": "发布",
        },
    }


def test_action_rejects_unknown_or_missing_snapshot_ref():
    _, state = normalize_desktop_browser_snapshot(_snapshot())

    with pytest.raises(DesktopBrowserProtocolError, match="not present in the latest Desktop snapshot"):
        build_desktop_browser_action("browser_click", {"ref": "@e99"}, state)

    with pytest.raises(DesktopBrowserProtocolError, match="Take a new browser_snapshot"):
        build_desktop_browser_action("browser_click", {"ref": "@e0"}, None)


def test_douyin_mutations_and_enter_require_approval_but_navigation_links_do_not():
    _, state = normalize_desktop_browser_snapshot(_snapshot())
    publish = build_desktop_browser_action("browser_click", {"ref": "@e0"}, state)

    assert "发布" in desktop_browser_approval_reason(publish)
    assert "Enter" in desktop_browser_approval_reason(
        build_desktop_browser_action("browser_press", {"key": "Enter"}, state)
    )
    assert (
        desktop_browser_approval_reason(
            {
                "expectedUrl": "https://www.douyin.com/user/example",
                "kind": "click",
                "target": {"href": "https://www.douyin.com/video/1", "tag": "a", "text": "查看视频"},
            }
        )
        is None
    )
    assert (
        desktop_browser_approval_reason(
            {
                "expectedUrl": "https://www.douyin.com/jingxuan",
                "kind": "click",
                "target": {"tag": "div", "text": "印度第一大城市孟买"},
            }
        )
        is None
    )


def test_navigation_preflight_normalizes_public_urls_and_blocks_secret_or_metadata_urls():
    assert validate_desktop_browser_navigation("example.com") == ("https://example.com/", None)

    _, secret_error = validate_desktop_browser_navigation("https://example.com/?token=sk-ant-api03-secret")
    assert "API key or token" in secret_error

    _, metadata_error = validate_desktop_browser_navigation("http://169.254.169.254/latest/meta-data/")
    assert "cloud metadata" in metadata_error


def test_string_callback_results_still_cross_the_browser_redaction_boundary():
    secret = "sk-proj-ABCD1234567890EFGH"

    result = execute_desktop_browser_tool(
        "browser_snapshot",
        {},
        lambda _name, _args: json.dumps({"success": True, "snapshot": secret}),
    )

    assert secret not in result
    assert "sk-pro" in result

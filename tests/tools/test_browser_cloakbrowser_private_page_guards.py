"""Regression tests for CloakBrowser private-page guards on snapshot/action paths."""

import json

import pytest

from tools import browser_tool


BLOCKED_URLS = [
    "http://127.0.0.1:8080/internal",
    "http://10.0.0.8/admin",
    "http://169.254.169.254/latest/meta-data/",
]


@pytest.fixture(autouse=True)
def _cloakbrowser_mode(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_cloakbrowser_mode", lambda: True)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_cloakbrowser_snapshot_blocks_private_pages(monkeypatch, url):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "cloakbrowser_current_url", lambda task_id: url)
    monkeypatch.setattr(
        browser_tool,
        "cloakbrowser_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("snapshot backend should not run")),
    )

    out = json.loads(browser_tool.browser_snapshot(task_id="task-1"))

    assert out["success"] is False
    assert "private or internal address" in out["error"]
    assert url in out["error"]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_cloakbrowser_get_images_blocks_private_pages(monkeypatch, url):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "cloakbrowser_current_url", lambda task_id: url)
    monkeypatch.setattr(
        browser_tool,
        "cloakbrowser_get_images",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("get_images backend should not run")),
    )

    out = json.loads(browser_tool.browser_get_images(task_id="task-1"))

    assert out["success"] is False
    assert "private or internal address" in out["error"]
    assert url in out["error"]


@pytest.mark.parametrize(
    ("tool_call", "args"),
    [
        (browser_tool.browser_click, ("@e1",)),
        (browser_tool.browser_type, ("@e1", "secret-text")),
        (browser_tool.browser_press, ("Enter",)),
    ],
)
@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_cloakbrowser_actions_block_private_pages(monkeypatch, tool_call, args, url):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "cloakbrowser_current_url", lambda task_id: url)

    backend_name = {
        browser_tool.browser_click: "cloakbrowser_click",
        browser_tool.browser_type: "cloakbrowser_type",
        browser_tool.browser_press: "cloakbrowser_press",
    }[tool_call]
    monkeypatch.setattr(
        browser_tool,
        backend_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("action backend should not run")),
    )

    out = json.loads(tool_call(*args, task_id="task-1"))

    assert out["success"] is False
    assert "private or internal address" in out["error"]
    assert url in out["error"]
    assert "secret-text" not in json.dumps(out)


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_cloakbrowser_back_blocks_when_history_lands_on_private_page(monkeypatch, url):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "cloakbrowser_current_url", lambda task_id: url)
    monkeypatch.setattr(
        browser_tool,
        "cloakbrowser_back",
        lambda task_id=None: json.dumps({"success": True, "url": url}),
    )

    out = json.loads(browser_tool.browser_back(task_id="task-1"))

    assert out["success"] is False
    assert "private or internal address" in out["error"]
    assert url in out["error"]
    assert "url" not in out


def test_cloakbrowser_snapshot_allows_public_page(monkeypatch):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "cloakbrowser_current_url", lambda task_id: "https://example.com/")
    monkeypatch.setattr(
        browser_tool,
        "cloakbrowser_snapshot",
        lambda full=False, task_id=None, user_task=None: json.dumps({"success": True, "snapshot": "ok", "element_count": 1}),
    )

    out = json.loads(browser_tool.browser_snapshot(task_id="task-1"))

    assert out == {"success": True, "snapshot": "ok", "element_count": 1}


def test_cloakbrowser_guard_inactive_does_not_probe(monkeypatch):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: False)

    def fail_probe(task_id):
        raise AssertionError("cloakbrowser_current_url must not be probed when guard inactive")

    monkeypatch.setattr(browser_tool, "cloakbrowser_current_url", fail_probe)
    monkeypatch.setattr(
        browser_tool,
        "cloakbrowser_click",
        lambda ref, task_id=None: json.dumps({"success": True, "clicked": ref}),
    )

    out = json.loads(browser_tool.browser_click("@e1", task_id="task-1"))

    assert out == {"success": True, "clicked": "@e1"}

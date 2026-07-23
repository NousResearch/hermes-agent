"""Regression tests for private-page browser interaction guards."""

import json

import pytest

from tools import browser_tool


PRIVATE_URL = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture(autouse=True)
def _browser_mode(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)


@pytest.mark.parametrize(
    ("tool_call", "args"),
    [
        (browser_tool.browser_click, ("@e1",)),
        (browser_tool.browser_type, ("@e1", "do-not-send-this")),
        (browser_tool.browser_press, ("Enter",)),
    ],
)
def test_private_page_blocks_state_changing_actions(monkeypatch, tool_call, args):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", lambda task_id: PRIVATE_URL)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("browser command should not run on a private page")

    monkeypatch.setattr(browser_tool, "_run_browser_command", fail_run)

    out = json.loads(tool_call(*args, task_id="task-1"))

    assert out["success"] is False
    assert PRIVATE_URL in out["error"]
    assert "private or internal address" in out["error"]
    assert "do-not-send-this" not in json.dumps(out)


def test_click_still_runs_when_current_page_is_public(monkeypatch):
    calls = []

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", lambda task_id: None)

    def fake_run(task_id, command, args):
        calls.append((task_id, command, args))
        return {"success": True}

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    out = json.loads(browser_tool.browser_click("e1", task_id="task-1"))

    assert out == {"success": True, "clicked": "@e1"}
    assert calls == [("task-1", "click", ["@e1"])]


@pytest.mark.parametrize(
    ("tool_call", "args", "command"),
    [
        (browser_tool.browser_click, ("@e1",), "click"),
        (browser_tool.browser_type, ("@e1", "do-not-send-this"), "fill"),
        (browser_tool.browser_press, ("Enter",), "press"),
    ],
)
def test_private_page_probe_failure_blocks_state_changing_actions(
    monkeypatch, tool_call, args, command
):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(
        browser_tool,
        "_current_page_private_url",
        lambda task_id: browser_tool._URL_PROBE_FAILED,
    )

    def fail_run(task_id, actual_command, actual_args):
        if actual_command == command:
            raise AssertionError("browser command should not run when URL probe fails")
        return {"success": True}

    monkeypatch.setattr(browser_tool, "_run_browser_command", fail_run)

    out = json.loads(tool_call(*args, task_id="task-1"))

    assert out["success"] is False
    assert "could not verify the current page URL" in out["error"]
    assert "do-not-send-this" not in json.dumps(out)


def test_guard_inactive_does_not_block_or_probe(monkeypatch):
    """When the SSRF guard is inactive (local backend / allow_private_urls),
    the action must proceed WITHOUT even probing the page URL — a private-looking
    current URL is irrelevant. This is the branch most likely to silently regress
    if the guard condition is ever inverted, so it is exercised explicitly."""
    calls = []

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: False)

    def fail_probe(task_id):
        raise AssertionError("_current_page_private_url must not be probed when guard inactive")

    monkeypatch.setattr(browser_tool, "_current_page_private_url", fail_probe)

    def fake_run(task_id, command, args):
        calls.append((task_id, command, args))
        return {"success": True}

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    out = json.loads(browser_tool.browser_click("@e1", task_id="task-1"))

    assert out == {"success": True, "clicked": "@e1"}
    assert calls == [("task-1", "click", ["@e1"])]


def test_camofox_short_circuits_before_guard(monkeypatch):
    """Camofox mode returns from the dedicated camofox_* path BEFORE reaching the
    private-page guard, so the guard's helpers must never be consulted. Guards the
    ordering invariant (camofox early-return precedes _last_session_key + guard)."""
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)

    def fail_guard(task_id):
        raise AssertionError("guard must not run in camofox mode")

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", fail_guard)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", fail_guard)

    import tools.browser_camofox as camofox

    monkeypatch.setattr(camofox, "camofox_click", lambda ref, task_id: '{"success": true, "camofox": true}')

    out = json.loads(browser_tool.browser_click("@e1", task_id="task-1"))

    assert out == {"success": True, "camofox": True}


# ---------------------------------------------------------------------------
# browser_back — unlike click/type/press (check current page BEFORE acting),
# going back IS the navigation: the guard must fire AFTER _run_browser_command
# reports success, checking the page it just landed on, not the page it left.
# ---------------------------------------------------------------------------


def test_browser_back_blocks_when_landed_page_is_private(monkeypatch):
    """Browser history can land on a private/internal address the initial
    browser_navigate preflight never saw — the same class of gap already
    closed for browser_snapshot/vision/console/eval and click/type/press."""
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", lambda task_id: PRIVATE_URL)
    monkeypatch.setattr(
        browser_tool, "_run_browser_command",
        lambda task_id, command, args: {"success": True, "data": {"url": PRIVATE_URL}},
    )

    out = json.loads(browser_tool.browser_back(task_id="task-1"))

    assert out["success"] is False
    assert PRIVATE_URL in out["error"]
    assert "private or internal address" in out["error"]
    # The blocked payload must not itself leak the raw URL as a "url" field
    # the way the success payload does.
    assert "url" not in out


def test_browser_back_returns_url_when_landed_page_is_public(monkeypatch):
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", lambda task_id: None)
    monkeypatch.setattr(
        browser_tool, "_run_browser_command",
        lambda task_id, command, args: {"success": True, "data": {"url": "https://example.com/"}},
    )

    out = json.loads(browser_tool.browser_back(task_id="task-1"))

    assert out == {"success": True, "url": "https://example.com/"}


def test_browser_back_guard_inactive_does_not_probe(monkeypatch):
    """When the SSRF guard is inactive (local backend), back navigation must
    proceed without even probing the landed page URL."""
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: False)

    def fail_probe(task_id):
        raise AssertionError("_current_page_private_url must not be probed when guard inactive")

    monkeypatch.setattr(browser_tool, "_current_page_private_url", fail_probe)
    monkeypatch.setattr(
        browser_tool, "_run_browser_command",
        lambda task_id, command, args: {"success": True, "data": {"url": "https://example.com/"}},
    )

    out = json.loads(browser_tool.browser_back(task_id="task-1"))

    assert out == {"success": True, "url": "https://example.com/"}


def test_browser_back_failed_navigation_does_not_probe(monkeypatch):
    """No page change happened, so there is nothing new to check — the guard
    must not fire (or probe) on a failed back navigation."""
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)

    def fail_probe(task_id):
        raise AssertionError("must not probe when the back navigation itself failed")

    monkeypatch.setattr(browser_tool, "_current_page_private_url", fail_probe)
    monkeypatch.setattr(
        browser_tool, "_run_browser_command",
        lambda task_id, command, args: {"success": False, "error": "no history"},
    )

    out = json.loads(browser_tool.browser_back(task_id="task-1"))

    assert out == {"success": False, "error": "no history"}


def test_browser_back_camofox_short_circuits_before_guard(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)

    def fail_guard(task_id):
        raise AssertionError("guard must not run in camofox mode")

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", fail_guard)
    monkeypatch.setattr(browser_tool, "_current_page_private_url", fail_guard)

    import tools.browser_camofox as camofox

    monkeypatch.setattr(camofox, "camofox_back", lambda task_id: '{"success": true, "camofox": true}')

    out = json.loads(browser_tool.browser_back(task_id="task-1"))

    assert out == {"success": True, "camofox": True}

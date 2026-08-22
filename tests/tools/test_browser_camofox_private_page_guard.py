"""Regression tests for the Camofox private-page read guards.

Companion to ``tests/tools/test_browser_private_page_action_guard.py`` (which
covers the agent-browser path) and ``test_browser_eval_ssrf.py`` (which covers
the Camofox *eval* path added in #56874).  These cover the remaining Camofox
content-read tools — snapshot / vision / image-extraction — which read current
page state and, on a non-local backend, could otherwise leak the content of a
private/internal page the terminal itself can't reach.
"""

import json

import pytest

from tools import browser_camofox


PRIVATE_URL = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture
def _session(monkeypatch):
    session = {"tab_id": "tab-1", "user_id": "user-1"}
    monkeypatch.setattr(browser_camofox, "_get_session", lambda task_id: session)
    return session


def _block_active(monkeypatch):
    """Make the SSRF guard active and the current page resolve to a private URL."""
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(
        browser_tool, "_camofox_current_page_private_url", lambda tab_id, user_id: PRIVATE_URL
    )


def _block_inactive_guard(monkeypatch):
    """SSRF guard inactive (local backend / allow_private_urls)."""
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: False)

    def fail_probe(tab_id, user_id):
        raise AssertionError("must not probe page URL when the SSRF guard is inactive")

    monkeypatch.setattr(browser_tool, "_camofox_current_page_private_url", fail_probe)


def _public_page(monkeypatch):
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(
        browser_tool, "_camofox_current_page_private_url", lambda tab_id, user_id: None
    )


@pytest.mark.parametrize(
    ("tool_call", "action_phrase"),
    [
        (lambda: browser_camofox.camofox_snapshot(task_id="t1"), "read a page snapshot"),
        (lambda: browser_camofox.camofox_get_images(task_id="t1"), "extract page images"),
        (lambda: browser_camofox.camofox_vision("what is here?", task_id="t1"), "capture a screenshot"),
    ],
)
def test_private_page_blocks_camofox_reads(monkeypatch, _session, tool_call, action_phrase):
    _block_active(monkeypatch)

    # Any HTTP call would mean the guard failed to short-circuit before the read.
    def fail_http(*_args, **_kwargs):
        raise AssertionError("Camofox HTTP call should not run on a private page")

    monkeypatch.setattr(browser_camofox, "_get", fail_http)
    monkeypatch.setattr(browser_camofox, "_get_raw", fail_http)
    monkeypatch.setattr(browser_camofox, "_post", fail_http)

    out = json.loads(tool_call())

    assert out["success"] is False
    assert PRIVATE_URL in out["error"]
    assert "private or internal address" in out["error"]
    assert action_phrase in out["error"]


@pytest.mark.parametrize(
    ("tool_call", "action_phrase"),
    [
        (lambda: browser_camofox.camofox_click("@e1", task_id="t1"), "click"),
        (
            lambda: browser_camofox.camofox_type("@e1", "do-not-send-this", task_id="t1"),
            "type",
        ),
        (lambda: browser_camofox.camofox_press("Enter", task_id="t1"), "press"),
    ],
)
def test_private_page_blocks_camofox_input_actions(monkeypatch, _session, tool_call, action_phrase):
    _block_active(monkeypatch)

    def fail_post(*_args, **_kwargs):
        raise AssertionError("Camofox action HTTP call should not run on a private page")

    monkeypatch.setattr(browser_camofox, "_post", fail_post)

    out = json.loads(tool_call())

    assert out["success"] is False
    assert PRIVATE_URL in out["error"]
    assert "private or internal address" in out["error"]
    assert action_phrase in out["error"]
    assert "do-not-send-this" not in json.dumps(out)


def test_snapshot_still_runs_when_page_is_public(monkeypatch, _session):
    _public_page(monkeypatch)

    monkeypatch.setattr(
        browser_camofox,
        "_get",
        lambda path, params=None: {"snapshot": "- heading \"Hi\" [e1]", "refsCount": 1},
    )

    out = json.loads(browser_camofox.camofox_snapshot(task_id="t1"))

    assert out["success"] is True
    assert out["element_count"] == 1


def test_camofox_click_still_runs_when_page_is_public(monkeypatch, _session):
    _public_page(monkeypatch)
    calls = []

    def fake_post(path, body=None, timeout=None):
        calls.append((path, body, timeout))
        return {"url": "https://example.test/"}

    monkeypatch.setattr(browser_camofox, "_post", fake_post)

    out = json.loads(browser_camofox.camofox_click("@e1", task_id="t1"))

    assert out["success"] is True
    assert out["clicked"] == "e1"
    assert calls == [
        (
            "/tabs/tab-1/click",
            {"userId": "user-1", "ref": "e1"},
            None,
        )
    ]


def test_guard_inactive_does_not_probe(monkeypatch, _session):
    """When the SSRF guard is inactive the read proceeds WITHOUT probing the URL.

    This is the branch most likely to silently regress if the guard condition is
    ever inverted, so it is exercised explicitly (mirrors the agent-browser
    guard test).
    """
    _block_inactive_guard(monkeypatch)

    monkeypatch.setattr(
        browser_camofox,
        "_get",
        lambda path, params=None: {"snapshot": "- heading \"Hi\" [e1]", "refsCount": 1},
    )

    out = json.loads(browser_camofox.camofox_snapshot(task_id="t1"))

    assert out["success"] is True
    assert out["element_count"] == 1


# ---------------------------------------------------------------------------
# Post-redirect SSRF guard on camofox_navigate (P1: redirect chain to
# private/metadata leaks the auto-snapshot) and camofox_back guard (P2).
# ---------------------------------------------------------------------------


def _mock_navigate_http(monkeypatch, final_url, snapshot_body):
    """Route navigate through a fake Camofox server that follows a redirect and
    returns *final_url*, and whose /snapshot returns *snapshot_body*."""
    monkeypatch.setattr(browser_camofox, "_get_session", lambda task_id: {"tab_id": "tab-1", "user_id": "user-1"})
    monkeypatch.setattr(browser_camofox, "_ensure_tab", lambda task_id, url: {"tab_id": "tab-1", "user_id": "user-1"})

    nav_calls = []

    def fake_post(path, body=None, timeout=None):
        if "navigate" in path:
            nav_calls.append(body)
            return {"ok": True, "url": final_url, "title": "redirected"}
        return {"ok": True, "url": final_url}

    monkeypatch.setattr(browser_camofox, "_post", fake_post)
    monkeypatch.setattr(
        browser_camofox,
        "_get",
        lambda path, params=None: {"snapshot": snapshot_body, "refsCount": 1},
    )
    return nav_calls


def test_camofox_navigate_blocks_redirect_to_metadata(monkeypatch, _session):
    """Public URL 302->IMDS: camofox_navigate must fail closed BEFORE the auto-snapshot.

    Regression for the Camofox redirect-chain SSRF leak: browser_navigate routed
    through camofox_navigate before any post-redirect check, and the auto-snapshot
    returned the landing page content.
    """
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    nav_calls = _mock_navigate_http(
        monkeypatch,
        final_url="http://169.254.169.254/latest/meta-data/",
        snapshot_body="<pre>arn:aws:iam::123456789012:role/leaked</pre>",
    )

    out = json.loads(browser_camofox.camofox_navigate("https://example.com/r", task_id="t1"))

    assert out["success"] is False
    assert "metadata endpoint" in out["error"]
    assert "snapshot" not in out, "leaked snapshot must not be returned"
    # The private page must be cleared (about:blank) to prevent later snapshot leaks.
    assert nav_calls[-1]["url"] == "about:blank"


def test_camofox_navigate_blocks_redirect_to_private(monkeypatch, _session):
    """Public URL 302->RFC1918 with the SSRF guard active: blocked, no snapshot."""
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    nav_calls = _mock_navigate_http(
        monkeypatch,
        final_url="http://10.0.0.5/admin",
        snapshot_body="<h1>internal admin</h1>",
    )

    out = json.loads(browser_camofox.camofox_navigate("https://example.com/r", task_id="t1"))

    assert out["success"] is False
    assert "private/internal address" in out["error"]
    assert "snapshot" not in out
    assert nav_calls[-1]["url"] == "about:blank"


def test_camofox_navigate_allows_public_redirect_when_guard_inactive(monkeypatch, _session):
    """SSRF guard inactive (local terminal): redirect to private is allowed, matching
    the local browser behavior; the auto-snapshot is returned."""
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: False)
    _mock_navigate_http(
        monkeypatch,
        final_url="http://10.0.0.5/admin",
        snapshot_body="<h1>internal admin</h1>",
    )

    out = json.loads(browser_camofox.camofox_navigate("https://example.com/r", task_id="t1"))

    assert out["success"] is True
    assert out["url"] == "http://10.0.0.5/admin"
    assert "internal admin" in out["snapshot"]


def test_camofox_back_blocks_private_post_back_page(monkeypatch, _session):
    """camofox_back must apply the private-page guard like every other Camofox
    content-returning action (snapshot/click/type/press)."""
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(
        browser_tool,
        "_camofox_current_page_private_url",
        lambda tab_id, user_id: "http://169.254.169.254/latest/meta-data/",
    )

    # The back HTTP call itself succeeds (matching browser_back: history nav runs,
    # then the post-back page is re-checked), and the guard must block the result.
    monkeypatch.setattr(
        browser_camofox,
        "_post",
        lambda path, body=None, timeout=None: {"ok": True, "url": "http://169.254.169.254/latest/meta-data/"},
    )

    out = json.loads(browser_camofox.camofox_back(task_id="t1"))

    assert out["success"] is False
    assert "private or internal address" in out["error"]


def test_camofox_back_still_runs_when_page_public(monkeypatch, _session):
    """Post-back page public: camofox_back returns the URL normally."""
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
    monkeypatch.setattr(browser_tool, "_camofox_current_page_private_url", lambda tab_id, user_id: None)

    def fake_post(path, body=None, timeout=None):
        return {"ok": True, "url": "https://example.com/prev"}

    monkeypatch.setattr(browser_camofox, "_post", fake_post)

    out = json.loads(browser_camofox.camofox_back(task_id="t1"))

    assert out["success"] is True
    assert out["url"] == "https://example.com/prev"

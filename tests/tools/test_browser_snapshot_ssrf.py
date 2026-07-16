"""Tests that browser_snapshot blocks content from eval-navigated private pages.

When browser_console() changes location.href to a private/internal address,
browser_snapshot() must detect this and return an error instead of exposing
the private page content.

This is the fix for the SSRF bypass described in issue #44731.
"""

import json

import pytest

from tools import browser_tool


def _make_snapshot_result(snapshot="Public page content", refs=None):
    """Return a mock successful snapshot result."""
    return {
        "success": True,
        "data": {
            "snapshot": snapshot,
            "refs": refs or {"@e1": {"role": "heading", "name": "Public"}},
        },
    }


def _make_eval_result(result):
    """Return a mock successful eval result."""
    return {"success": True, "data": {"result": result}}


# ---------------------------------------------------------------------------
# browser_snapshot: private-network guard after eval navigation
# ---------------------------------------------------------------------------


class TestBrowserSnapshotPrivateNetworkGuard:
    """browser_snapshot must block content from private pages navigated via eval."""

    PRIVATE_URL = "http://127.0.0.1:8080/secret"
    PUBLIC_URL = "https://example.com/page"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        """Common patches for snapshot SSRF tests."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
                "_first_nav": False,
            },
        )

    def test_blocks_private_url_after_eval_navigation(self, monkeypatch):
        """Snapshot must block when current page URL is private."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        commands = []

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            commands.append((command, kwargs))
            if command == "snapshot":
                return _make_snapshot_result()
            if command == "eval":
                return _make_eval_result(self.PRIVATE_URL)
            if command == "open":
                return {"success": True, "data": {"url": "about:blank"}}
            return {"success": False, "error": "unknown command"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is False
        assert "private or internal address" in result["error"]
        assert self.PRIVATE_URL in result["error"]
        assert [command for command, _ in commands] == ["snapshot", "eval", "open"]
        assert commands[1][1]["_allow_fallback"] is False
        assert commands[2][1]["_allow_fallback"] is False

    def test_allows_public_url_after_eval_navigation(self, monkeypatch):
        """Snapshot must succeed when current page URL is public."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            elif command == "eval":
                return _make_eval_result(self.PUBLIC_URL)
            return {"success": False, "error": "unknown command"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is True
        assert "snapshot" in result

    def test_skips_check_in_local_backend_mode(self, monkeypatch):
        """Local backend mode skips SSRF check entirely."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            return {"success": False, "error": "should not be called"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is True
        assert "snapshot" in result


    def test_skips_check_for_local_sidecar_session(self, monkeypatch):
        """Local sidecar sessions can legitimately access private URLs."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        # Simulate the effective_task_id being a local sidecar key
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            return {"success": False, "error": "should not be called"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is True
        assert "snapshot" in result

    def test_skips_check_when_private_urls_allowed(self, monkeypatch):
        """When allow_private_urls is enabled, SSRF check is skipped."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            return {"success": False, "error": "should not be called"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is True
        assert "snapshot" in result

    def test_handles_eval_failure_fail_closed(self, monkeypatch):
        """If URL eval fails, snapshot content is withheld."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        secret = "METADATA_SECRET_CONTENT"

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result(snapshot=secret)
            if command == "eval":
                return {"success": False, "error": "eval failed"}
            if command == "open":
                return {"success": True, "data": {"url": "about:blank"}}
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is False
        assert "unable to verify" in result["error"]
        assert secret not in json.dumps(result)

    def test_handles_empty_url_result_fail_closed(self, monkeypatch):
        """An empty URL probe cannot release snapshot content."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        secret = "METADATA_SECRET_CONTENT"

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result(snapshot=secret)
            if command == "eval":
                return _make_eval_result("")
            if command == "open":
                return {"success": True, "data": {"url": "about:blank"}}
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is False
        assert "unable to verify" in result["error"]
        assert secret not in json.dumps(result)

    def test_handles_eval_exception_fail_closed(self, monkeypatch):
        """An eval exception withholds snapshot content."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        secret = "METADATA_SECRET_CONTENT"

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result(snapshot=secret)
            if command == "eval":
                raise RuntimeError("CDP connection lost")
            if command == "open":
                return {"success": True, "data": {"url": "about:blank"}}
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is False
        assert "unable to verify" in result["error"]
        assert secret not in json.dumps(result)

    def test_blocks_loopback_url(self, monkeypatch):
        """Loopback URLs (localhost) must be blocked."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            elif command == "eval":
                return _make_eval_result("http://localhost:3000/admin")
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_blocks_private_ip_range(self, monkeypatch):
        """Private IP ranges (10.x, 172.16.x, 192.168.x) must be blocked."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        for private_ip in ["http://10.0.0.1/api", "http://172.16.0.1/admin", "http://192.168.1.1/config"]:
            def mock_run_browser_command(task_id, command, args=None, **kwargs):
                if command == "snapshot":
                    return _make_snapshot_result()
                elif command == "eval":
                    return _make_eval_result(private_ip)
                return {"success": False, "error": "unknown"}

            monkeypatch.setattr(
                browser_tool, "_run_browser_command", mock_run_browser_command
            )

            result = json.loads(browser_browser_snapshot(task_id="test"))
            assert result["success"] is False, f"Expected block for {private_ip}"
            assert "private or internal address" in result["error"]


# Helper to avoid name collision with the actual function
def browser_browser_snapshot(**kwargs):
    from tools.browser_tool import browser_snapshot
    return browser_snapshot(**kwargs)


def browser_browser_vision(**kwargs):
    from tools.browser_tool import browser_vision
    return browser_vision(**kwargs)


def _make_screenshot_result(path="/tmp/test_screenshot.png"):
    """Return a mock successful screenshot result."""
    return {"success": True, "data": {"path": path}}


# ---------------------------------------------------------------------------
# browser_vision: private-network guard after eval navigation
# ---------------------------------------------------------------------------


class TestBrowserVisionPrivateNetworkGuard:
    """browser_vision must block screenshots from private pages navigated via eval."""

    PRIVATE_URL = "http://127.0.0.1:8080/secret"
    PUBLIC_URL = "https://example.com/page"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        """Common patches for vision SSRF tests."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)

    def test_blocks_private_url_after_eval_navigation(self, monkeypatch):
        """Vision must block when current page URL is private."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "eval":
                return _make_eval_result(self.PRIVATE_URL)
            return {"success": False, "error": "should not reach screenshot"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_vision(question="what do you see", task_id="test"))
        assert result["success"] is False
        assert "private or internal address" in result["error"]
        assert self.PRIVATE_URL in result["error"]

    def test_allows_public_url_after_eval_navigation(self, monkeypatch):
        """Vision must proceed when current page URL is public."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "eval":
                return _make_eval_result(self.PUBLIC_URL)
            elif command == "screenshot":
                return _make_screenshot_result()
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )
        # Screenshot file won't exist — that's fine, function returns error
        # but the important thing is the guard didn't block it.

        result_raw = browser_browser_vision(question="what do you see", task_id="test")
        result = json.loads(result_raw)
        # Guard passed; function continues to screenshot path.
        # Since screenshot file doesn't exist, it returns a file-not-found error,
        # NOT the "private or internal address" error.
        assert "private or internal address" not in result.get("error", "")

    def test_skips_check_in_local_backend_mode(self, monkeypatch):
        """Local backend mode skips SSRF check entirely."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "screenshot":
                return _make_screenshot_result()
            return {"success": False, "error": "should not be called"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result_raw = browser_browser_vision(question="what", task_id="test")
        result = json.loads(result_raw)
        assert "private or internal address" not in result.get("error", "")


    def test_skips_check_for_local_sidecar_session(self, monkeypatch):
        """Local sidecar sessions can legitimately access private URLs."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        # Simulate the effective_task_id being a local sidecar key
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "screenshot":
                return _make_screenshot_result()
            return {"success": False, "error": "should not be called"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result_raw = browser_browser_vision(question="what", task_id="test")
        result = json.loads(result_raw)
        assert "private or internal address" not in result.get("error", "")

    def test_skips_check_when_private_urls_allowed(self, monkeypatch):
        """When allow_private_urls is enabled, SSRF check is skipped."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "screenshot":
                return _make_screenshot_result()
            return {"success": False, "error": "should not be called"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result_raw = browser_browser_vision(question="what", task_id="test")
        result = json.loads(result_raw)
        assert "private or internal address" not in result.get("error", "")

    def test_handles_eval_failure_gracefully(self, monkeypatch):
        """If URL eval fails, vision should still proceed (fail-open)."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "eval":
                return {"success": False, "error": "eval failed"}
            elif command == "screenshot":
                return _make_screenshot_result()
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result_raw = browser_browser_vision(question="what", task_id="test")
        result = json.loads(result_raw)
        assert "private or internal address" not in result.get("error", "")

    def test_handles_eval_exception(self, monkeypatch):
        """If URL eval raises an exception, vision should still proceed."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "eval":
                raise RuntimeError("CDP connection lost")
            elif command == "screenshot":
                return _make_screenshot_result()
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result_raw = browser_browser_vision(question="what", task_id="test")
        result = json.loads(result_raw)
        assert "private or internal address" not in result.get("error", "")


@pytest.mark.parametrize(
    "tool_name",
    ["snapshot", "eval-cli", "images", "vision", "console"],
)
def test_actual_served_fallback_session_is_guarded_on_every_content_path(
    monkeypatch, tmp_path, tool_name
):
    """Content is discarded when the exact fallback serving session is unsafe."""
    task_id = f"served-guard-{tool_name}"
    local_session_key = f"{task_id}::local"
    secret = "METADATA_SECRET_CONTENT"
    screenshot_path = tmp_path / f"{tool_name}.png"
    events = []
    href_probes = 0

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda session_key: {
            "session_key": session_key,
            "owner_task_id": task_id,
            "session_name": f"s_{session_key}",
            "features": {},
            "_first_nav": False,
        },
    )

    fallback_meta = {
        "browser_session_key": local_session_key,
        "browser_backend_fallback": {"from": "cdp", "to": "local"},
    }

    def run_command(session_key, command, args=None, timeout=None, **kwargs):
        nonlocal href_probes
        args = args or []
        events.append((session_key, command, args, kwargs))
        if command == "eval" and args == ["window.location.href"]:
            href_probes += 1
            url = (
                "https://example.com/"
                if session_key == task_id
                else "http://169.254.169.254/latest/meta-data"
            )
            return {"success": True, "data": {"result": url}}
        if command == "snapshot":
            return {
                "success": True,
                "data": {"snapshot": secret, "refs": {"e1": {}}},
                **fallback_meta,
            }
        if command == "eval":
            payload = (
                json.dumps([{"src": secret, "alt": secret}])
                if tool_name == "images"
                else secret
            )
            return {"success": True, "data": {"result": payload}, **fallback_meta}
        if command == "console":
            return {
                "success": True,
                "data": {"messages": [{"type": "log", "text": secret}]},
                **fallback_meta,
            }
        if command == "errors":
            return {"success": False, "error": "no error buffer"}
        if command == "screenshot":
            screenshot_path.write_bytes(secret.encode())
            return {
                "success": True,
                "data": {"path": str(screenshot_path)},
                **fallback_meta,
            }
        if command == "open" and args == ["about:blank"]:
            return {"success": True, "data": {"url": "about:blank"}}
        raise AssertionError((session_key, command, args, timeout, kwargs))

    monkeypatch.setattr(browser_tool, "_run_browser_command", run_command)

    if tool_name == "snapshot":
        raw_response = browser_tool.browser_snapshot(task_id=task_id)
    elif tool_name == "eval-cli":
        raw_response = browser_tool.browser_console(
            expression="document.body.innerText",
            task_id=task_id,
        )
    elif tool_name == "images":
        raw_response = browser_tool.browser_get_images(task_id=task_id)
    elif tool_name == "vision":
        raw_response = browser_tool.browser_vision("what is visible?", task_id=task_id)
    else:
        raw_response = browser_tool.browser_console(task_id=task_id)

    response = raw_response if isinstance(raw_response, dict) else json.loads(raw_response)
    assert response["success"] is False
    assert "metadata endpoint" in response["error"]
    assert secret not in json.dumps(response)
    assert href_probes >= 1
    if tool_name == "vision":
        assert not screenshot_path.exists()


def test_navigate_auto_snapshot_guards_the_actual_serving_session(monkeypatch):
    """A safe open proof cannot authorize snapshot bytes from a later unsafe page."""
    task_id = "served-guard-navigate"
    local_session_key = f"{task_id}::local"
    secret = "METADATA_SECRET_CONTENT"
    probe_count = 0

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(
        browser_tool,
        "_get_session_info",
        lambda session_key: {
            "session_key": session_key,
            "owner_task_id": task_id,
            "session_name": f"s_{session_key}",
            "features": {"cdp_override": True},
            "_first_nav": False,
        },
    )

    def run_command(session_key, command, args=None, **kwargs):
        nonlocal probe_count
        args = args or []
        if command == "open" and args == ["https://example.com"]:
            return {
                "success": True,
                "data": {"title": "safe", "url": "https://example.com/"},
                "browser_session_key": local_session_key,
                "browser_backend_fallback": {"from": "cdp", "to": "local"},
            }
        if command == "eval":
            assert session_key == local_session_key
            probe_count += 1
            url = (
                "https://example.com/"
                if probe_count == 1
                else "http://169.254.169.254/latest/meta-data"
            )
            return {"success": True, "data": {"result": url}}
        if command == "snapshot":
            assert session_key == local_session_key
            return {
                "success": True,
                "data": {"snapshot": secret, "refs": {"e1": {}}},
            }
        if command == "open" and args == ["about:blank"]:
            return {"success": True, "data": {"url": "about:blank"}}
        raise AssertionError((session_key, command, args, kwargs))

    monkeypatch.setattr(browser_tool, "_run_browser_command", run_command)
    response = json.loads(
        browser_tool.browser_navigate("https://example.com", task_id=task_id)
    )

    assert response["success"] is False
    assert "metadata endpoint" in response["error"]
    assert secret not in json.dumps(response)
    assert probe_count == 2


def test_supervisor_eval_result_uses_actual_served_url_guard(monkeypatch):
    """The CDP-supervisor eval branch cannot bypass the common post-result guard."""
    task_id = "served-guard-supervisor"
    secret = "METADATA_SECRET_CONTENT"

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

    from tools import browser_supervisor

    class FakeSupervisor:
        def evaluate_runtime(self, _expression):
            return {"ok": True, "result": secret}

    monkeypatch.setattr(
        browser_supervisor,
        "SUPERVISOR_REGISTRY",
        {task_id: FakeSupervisor()},
    )

    def run_command(session_key, command, args=None, **kwargs):
        assert session_key == task_id
        if command == "eval":
            return {
                "success": True,
                "data": {"result": "http://169.254.169.254/latest/meta-data"},
            }
        if command == "open":
            return {"success": True, "data": {"url": "about:blank"}}
        raise AssertionError((session_key, command, args, kwargs))

    monkeypatch.setattr(browser_tool, "_run_browser_command", run_command)
    response = json.loads(
        browser_tool.browser_console(expression="document.body.innerText", task_id=task_id)
    )

    assert response["success"] is False
    assert "metadata endpoint" in response["error"]
    assert secret not in json.dumps(response)
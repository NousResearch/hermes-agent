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

        call_count = {"n": 0}

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            call_count["n"] += 1
            if command == "snapshot":
                return _make_snapshot_result()
            elif command == "eval":
                return _make_eval_result(self.PRIVATE_URL)
            return {"success": False, "error": "unknown command"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is False
        assert "private or internal address" in result["error"]
        assert self.PRIVATE_URL in result["error"]
        # Must have called eval to check URL
        assert call_count["n"] == 2  # snapshot + eval

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

    def test_handles_eval_failure_gracefully(self, monkeypatch):
        """If URL eval fails, snapshot should still succeed (fail-open)."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            elif command == "eval":
                return {"success": False, "error": "eval failed"}
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        # Should succeed — eval failure means we can't determine URL, fail-open
        assert result["success"] is True

    def test_handles_empty_url_result(self, monkeypatch):
        """If URL eval returns empty string, snapshot should succeed."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            elif command == "eval":
                return _make_eval_result("")
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is True

    def test_handles_eval_exception(self, monkeypatch):
        """If URL eval raises an exception, snapshot should succeed."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "snapshot":
                return _make_snapshot_result()
            elif command == "eval":
                raise RuntimeError("CDP connection lost")
            return {"success": False, "error": "unknown"}

        monkeypatch.setattr(
            browser_tool, "_run_browser_command", mock_run_browser_command
        )

        result = json.loads(browser_browser_snapshot(task_id="test"))
        assert result["success"] is True

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
            if command == "eval":
                return _make_eval_result(self.PUBLIC_URL)
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

    def test_peer_violation_removes_requested_and_backend_screenshot_paths(
        self, monkeypatch, tmp_path
    ):
        """Blocked screenshots cannot remain at a backend-returned alternate path."""
        from types import SimpleNamespace

        import hermes_constants
        from tools import browser_supervisor

        records = []
        requested_dir = tmp_path / "requested"
        actual_path = tmp_path / "backend" / "actual.png"
        requested_paths = []
        unsafe_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/internal",
            remote_ip="192.168.1.10",
        )
        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
        )

        def mock_run_browser_command(task_id, command, args=None, **kwargs):
            if command == "screenshot":
                requested_path = type(actual_path)(args[-1])
                requested_paths.append(requested_path)
                requested_path.parent.mkdir(parents=True, exist_ok=True)
                requested_path.write_bytes(b"private requested screenshot")
                actual_path.parent.mkdir(parents=True, exist_ok=True)
                actual_path.write_bytes(b"private backend screenshot")
                records.append(unsafe_record)
                return _make_screenshot_result(str(actual_path))
            if command == "open":
                return {"success": True}
            return {"success": False, "error": "unexpected command"}

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "chrome")
        monkeypatch.setattr(browser_tool, "_should_inject_engine", lambda engine: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", mock_run_browser_command)
        monkeypatch.setattr(hermes_constants, "get_hermes_dir", lambda *args: requested_dir)
        monkeypatch.setattr(
            browser_supervisor.SUPERVISOR_REGISTRY,
            "get",
            lambda task_id: fake_supervisor,
        )

        result = json.loads(browser_browser_vision(question="what", task_id="test"))

        assert result["success"] is False
        assert "browser connected to a private/internal address" in result["error"]
        assert requested_paths and not requested_paths[0].exists()
        assert not actual_path.exists()

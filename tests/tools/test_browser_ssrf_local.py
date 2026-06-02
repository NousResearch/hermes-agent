"""Tests that browser_navigate SSRF checks respect the allow_private_urls setting.

Local and cloud backends both enforce SSRF checks by default because browser
snapshots can expose local files or internal service responses in reduced-tool
configurations. Users can opt out via ``browser.allow_private_urls: true``.
"""

import json

import pytest

from tools import browser_tool


def _make_browser_result(url="https://example.com"):
    """Return a mock successful browser command result."""
    return {"success": True, "data": {"title": "OK", "url": url}}


# ---------------------------------------------------------------------------
# Pre-navigation SSRF check
# ---------------------------------------------------------------------------


class TestPreNavigationSsrf:
    PRIVATE_URL = "http://127.0.0.1:8080/dashboard"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for pre-navigation tests that pass the SSRF check."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
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
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(),
        )

    # -- Cloud mode: SSRF active -----------------------------------------------

    def test_cloud_blocks_private_url_by_default(self, monkeypatch, _common_patches):
        """SSRF protection blocks private URLs in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_cloud_allows_private_url_when_setting_true(self, monkeypatch, _common_patches):
        """Private URLs pass in cloud mode when allow_private_urls is True."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    def test_cloud_allows_public_url(self, monkeypatch, _common_patches):
        """Public URLs always pass in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True

    # -- Local mode: SSRF active by default -------------------------------------

    def test_local_blocks_private_url_by_default(self, monkeypatch, _common_patches):
        """Local backends block private URLs unless the user opts out."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_local_allows_private_url_when_setting_true(self, monkeypatch, _common_patches):
        """Local backends allow private URLs when allow_private_urls is True."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

    def test_local_blocks_file_url_by_default(self, monkeypatch, _common_patches):
        """Local backends block non-network schemes unless the user opts out."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate("file:///etc/passwd"))

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_local_allows_public_url(self, monkeypatch, _common_patches):
        """Local backends pass public URLs too (sanity check)."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate("https://example.com"))

        assert result["success"] is True

    # -- Always-blocked floor: hybrid routing bypass regression (#16234) -------

    # Hybrid-routing feature flips auto_local_this_nav=True for private URLs,
    # which previously short-circuited _is_safe_url() entirely. An agent
    # running on EC2/GCP/Azure could navigate to 169.254.169.254 via the
    # spawned local Chromium sidecar and read IAM credentials via
    # browser_snapshot. The always-blocked floor must fire regardless of
    # routing.
    IMDS_URLS = [
        "http://169.254.169.254/latest/meta-data/",      # AWS / GCP / Azure / DO / Oracle
        "http://169.254.169.253/metadata/instance",        # Azure IMDS wire server
        "http://169.254.170.2/v2/credentials",             # AWS ECS task metadata
        "http://100.100.100.200/latest/meta-data/",        # Alibaba Cloud
        "http://metadata.google.internal/computeMetadata/v1/",  # GCP hostname
    ]

    @pytest.mark.parametrize("imds_url", IMDS_URLS)
    def test_cloud_blocks_imds_even_when_routing_to_local_sidecar(
        self, monkeypatch, _common_patches, imds_url
    ):
        """Hybrid routing must not let cloud metadata endpoints through."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        # Simulate hybrid routing kicking in for this URL (what happens on
        # main pre-fix — cloud provider configured, _url_is_private → True,
        # so the session key routes to a local Chromium sidecar).
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)
        # _is_safe_url would catch IMDS, but pre-fix it never ran. Force
        # it to return True here so the test is specifically pinning the
        # always-blocked floor as an independent gate.
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)

        result = json.loads(browser_tool.browser_navigate(imds_url))

        assert result["success"] is False
        assert "cloud metadata endpoint" in result["error"]

    def test_cloud_allows_ordinary_private_url_via_sidecar(
        self, monkeypatch, _common_patches
    ):
        """Hybrid routing still works for ordinary private URLs — floor
        must be narrow enough to not break the PR #16136 feature."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        for private in (
            "http://127.0.0.1:8080/dashboard",
            "http://192.168.1.1/admin",
            "http://10.0.0.5/",
            "http://myservice.local/",
        ):
            result = json.loads(browser_tool.browser_navigate(private))
            assert result["success"] is True, f"Unexpected block for {private}: {result}"


# ---------------------------------------------------------------------------
# _is_local_backend() unit tests
# ---------------------------------------------------------------------------


class TestIsLocalBackend:
    def test_camofox_is_local(self, monkeypatch):
        """Camofox mode counts as a local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: "anything")

        assert browser_tool._is_local_backend() is True

    def test_no_cloud_provider_is_local(self, monkeypatch):
        """No cloud provider configured → local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)

        assert browser_tool._is_local_backend() is True

    def test_cloud_provider_is_not_local(self, monkeypatch):
        """Cloud provider configured and not Camofox → NOT local."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: "bb")

        assert browser_tool._is_local_backend() is False


# ---------------------------------------------------------------------------
# Post-redirect SSRF check
# ---------------------------------------------------------------------------


class TestPostRedirectSsrf:
    PUBLIC_URL = "https://example.com/redirect"
    PRIVATE_FINAL_URL = "http://192.168.1.1/internal"

    @pytest.fixture()
    def _common_patches(self, monkeypatch):
        """Shared patches for redirect tests."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
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

    # -- Cloud mode: redirect SSRF active --------------------------------------

    def test_cloud_blocks_redirect_to_private(self, monkeypatch, _common_patches):
        """Redirects to private addresses are blocked in cloud mode."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is False
        assert "redirect landed on a private/internal address" in result["error"]

    def test_cloud_allows_redirect_to_private_when_setting_true(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in cloud mode with allow_private_urls."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == self.PRIVATE_FINAL_URL

    # -- Local mode: redirect SSRF active by default ---------------------------

    def test_local_blocks_redirect_to_private_by_default(self, monkeypatch, _common_patches):
        """Redirects to private addresses are blocked in local mode by default."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is False
        assert "redirect landed on a private/internal address" in result["error"]

    def test_local_allows_redirect_to_private_when_setting_true(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in local mode with allow_private_urls."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(
            browser_tool, "_is_safe_url", lambda url: "192.168" not in url,
        )
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=self.PRIVATE_FINAL_URL),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == self.PRIVATE_FINAL_URL

    def test_cloud_allows_redirect_to_public(self, monkeypatch, _common_patches):
        """Redirects to public addresses always pass (cloud mode)."""
        final = "https://example.com/final"
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=final),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is True
        assert result["url"] == final

    # -- Always-blocked floor: redirect to IMDS via hybrid sidecar (#16234) ----

    def test_cloud_blocks_redirect_to_imds_even_via_sidecar(
        self, monkeypatch, _common_patches
    ):
        """Redirect to a cloud metadata endpoint is blocked regardless of
        routing — even the hybrid local sidecar path can't return IMDS
        content to the agent."""
        imds_final = "http://169.254.169.254/latest/meta-data/"
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: True)
        # _is_safe_url would catch it on main; force True to pin the
        # always-blocked floor as an independent gate.
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *a, **kw: _make_browser_result(url=imds_final),
        )

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL))

        assert result["success"] is False
        assert "cloud metadata endpoint" in result["error"]


class TestAllowPrivateUrlsConfig:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        browser_tool._allow_private_urls_resolved = False
        browser_tool._cached_allow_private_urls = None
        yield
        browser_tool._allow_private_urls_resolved = False
        browser_tool._cached_allow_private_urls = None

    def test_browser_config_string_false_stays_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"allow_private_urls": "false"}},
        )

        assert browser_tool._allow_private_urls() is False


class TestPostActionSsrf:
    """Navigation-capable browser actions must not leave cloud sessions on private URLs."""

    METADATA_URL = "http://169.254.169.254/latest/meta-data/"
    INTERNAL_SNAPSHOT = "INTERNAL_METADATA"

    @pytest.fixture()
    def _cloud_patches(self, monkeypatch):
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: url == self.METADATA_URL)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: url != self.METADATA_URL)
        monkeypatch.setattr(
            browser_tool,
            "_get_session_info",
            lambda task_id: {
                "session_name": f"s_{task_id}",
                "bb_session_id": "remote-session",
                "cdp_url": "wss://firecrawl.example/devtools/browser/1",
                "features": {"firecrawl": True},
                "_first_nav": False,
            },
        )

    def test_click_resets_cloud_browser_after_metadata_navigation(self, monkeypatch, _cloud_patches):
        calls = []

        def fake_run(task_id, command, args=None, **kwargs):
            calls.append((command, args or []))
            if command == "click":
                return {"success": True, "data": {}}
            if command == "eval":
                return {"success": True, "data": {"result": self.METADATA_URL}}
            if command == "open" and args == ["about:blank"]:
                return {"success": True, "data": {"url": "about:blank"}}
            raise AssertionError(f"unexpected browser command: {command} {args}")

        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

        result = json.loads(browser_tool.browser_click("@e1", task_id="task"))

        assert result["success"] is False
        assert "cloud metadata endpoint" in result["error"]
        assert ("open", ["about:blank"]) in calls

    def test_snapshot_refuses_to_read_cloud_browser_on_metadata_url(self, monkeypatch, _cloud_patches):
        calls = []

        def fake_run(task_id, command, args=None, **kwargs):
            calls.append((command, args or []))
            if command == "eval":
                return {"success": True, "data": {"result": self.METADATA_URL}}
            if command == "open" and args == ["about:blank"]:
                return {"success": True, "data": {"url": "about:blank"}}
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": self.INTERNAL_SNAPSHOT, "refs": {}}}
            raise AssertionError(f"unexpected browser command: {command} {args}")

        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

        result = json.loads(browser_tool.browser_snapshot(task_id="task"))

        assert result["success"] is False
        assert "cloud metadata endpoint" in result["error"]
        assert ("snapshot", ["-c"]) not in calls
        assert ("open", ["about:blank"]) in calls

    def test_console_eval_resets_cloud_browser_after_metadata_navigation(self, monkeypatch, _cloud_patches):
        calls = []

        def fake_run(task_id, command, args=None, **kwargs):
            calls.append((command, args or []))
            if command == "eval" and args == ["location.href = 'http://169.254.169.254/latest/meta-data/'"]:
                return {"success": True, "data": {"result": "undefined"}}
            if command == "eval" and args == ["window.location.href"]:
                return {"success": True, "data": {"result": self.METADATA_URL}}
            if command == "open" and args == ["about:blank"]:
                return {"success": True, "data": {"url": "about:blank"}}
            raise AssertionError(f"unexpected browser command: {command} {args}")

        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

        result = json.loads(
            browser_tool.browser_console(
                expression="location.href = 'http://169.254.169.254/latest/meta-data/'",
                task_id="task",
            )
        )

        assert result["success"] is False
        assert "cloud metadata endpoint" in result["error"]
        assert ("open", ["about:blank"]) in calls

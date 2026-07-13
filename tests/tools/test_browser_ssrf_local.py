"""Tests that browser_navigate SSRF checks respect local-backend mode and
the allow_private_urls setting.

Local backends (Camofox, headless Chromium without a cloud provider) skip
SSRF checks entirely — the agent already has full local-network access via
the terminal tool.

Cloud backends (Browserbase, BrowserUse) enforce SSRF by default.  Users
can opt out for cloud mode via ``browser.allow_private_urls: true``.
"""

import json
from types import SimpleNamespace

import pytest

from tools import browser_supervisor, browser_tool


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

    # -- Local mode: SSRF skipped ----------------------------------------------

    def test_local_allows_private_url(self, monkeypatch, _common_patches):
        """Local backends skip SSRF — private URLs are always allowed."""
        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False)

        result = json.loads(browser_tool.browser_navigate(self.PRIVATE_URL))

        assert result["success"] is True

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

    @pytest.mark.parametrize("backend", ["docker", "modal", "daytona", "ssh", "singularity"])
    def test_container_terminal_backend_is_not_local(self, monkeypatch, backend):
        """Terminal running in a container → NOT local (browser on host can access internal networks)."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("TERMINAL_ENV", backend)

        assert browser_tool._is_local_backend() is False

    def test_empty_terminal_env_is_local(self, monkeypatch):
        """Empty TERMINAL_ENV → local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("TERMINAL_ENV", "")

        assert browser_tool._is_local_backend() is True

    def test_local_terminal_env_is_local(self, monkeypatch):
        """Explicit 'local' TERMINAL_ENV → local backend."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert browser_tool._is_local_backend() is True

    def test_camofox_overrides_container_backend(self, monkeypatch):
        """Camofox mode always counts as local, even with container terminal."""
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        assert browser_tool._is_local_backend() is True


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

    # -- Local mode: redirect SSRF skipped -------------------------------------

    def test_local_allows_redirect_to_private(self, monkeypatch, _common_patches):
        """Redirects to private addresses pass in local mode."""
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

    # -- CDP-observed remote IP: DNS rebinding / TOCTOU guard ------------------

    def test_cloud_blocks_browser_observed_private_peer(
        self, monkeypatch, _common_patches
    ):
        """CDP-reported peer IP blocks DNS rebinding with unchanged final URL."""
        calls = []
        cleared = []

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            return _make_browser_result(url=self.PUBLIC_URL)

        record = SimpleNamespace(
            ts=101.0,
            url=self.PUBLIC_URL,
            remote_ip="192.168.1.10",
        )
        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=(record,)),
            clear_network_responses=lambda: cleared.append(True),
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda task_id: fake_supervisor)

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL, task_id="rebind"))

        assert result["success"] is False
        assert "browser connected to a private/internal address" in result["error"]
        assert cleared == [True]
        assert ("rebind", "open", ["about:blank"]) in calls

    def test_cloud_allows_browser_observed_public_peer(
        self, monkeypatch, _common_patches
    ):
        """CDP-reported public peer IP preserves normal navigation success."""
        cleared = []
        record = SimpleNamespace(
            ts=101.0,
            url=self.PUBLIC_URL,
            remote_ip="93.184.216.34",
        )
        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=(record,)),
            clear_network_responses=lambda: cleared.append(True),
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", lambda *a, **kw: _make_browser_result(url=self.PUBLIC_URL))
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda task_id: fake_supervisor)

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL, task_id="public"))

        assert result["success"] is True
        assert result["url"] == self.PUBLIC_URL
        assert cleared == [True]

    def test_cloud_blocks_peer_observed_during_auto_snapshot(
        self, monkeypatch, _common_patches
    ):
        """The post-snapshot guard blocks peers observed after navigation."""
        calls = []
        cleared = []
        records = []

        unsafe_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/snapshot?token=secret",
            remote_ip="192.168.1.10",
        )

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            if command == "snapshot":
                records.append(unsafe_record)
                return {
                    "success": True,
                    "data": {
                        "snapshot": "unsafe private content",
                        "refs": {"e1": "link"},
                    },
                }
            return _make_browser_result(url=self.PUBLIC_URL)

        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
            clear_network_responses=lambda: cleared.append(True),
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda task_id: fake_supervisor)

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL, task_id="snapshot-rebind"))

        assert result["success"] is False
        assert "browser connected to a private/internal address" in result["error"]
        assert "snapshot" not in result
        assert cleared == [True]
        assert calls == [
            ("snapshot-rebind", "open", [self.PUBLIC_URL]),
            ("snapshot-rebind", "snapshot", ["-c"]),
            ("snapshot-rebind", "open", ["about:blank"]),
        ]

    def test_cloud_blocks_browser_observed_metadata_peer_even_with_private_allowed(
        self, monkeypatch, _common_patches
    ):
        """The always-blocked floor still applies to observed peer IPs."""
        calls = []
        cleared = []

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            return _make_browser_result(url=self.PUBLIC_URL)

        record = SimpleNamespace(
            ts=101.0,
            url=self.PUBLIC_URL,
            remote_ip="169.254.169.254",
        )
        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=(record,)),
            clear_network_responses=lambda: cleared.append(True),
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda task_id: fake_supervisor)

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL, task_id="imds"))

        assert result["success"] is False
        assert "browser connected to a cloud metadata endpoint" in result["error"]
        assert cleared == [True]
        assert ("imds", "open", ["about:blank"]) in calls

    def test_cloud_blocks_malformed_observed_peer_even_with_private_allowed(
        self, monkeypatch, _common_patches
    ):
        """Malformed CDP peer IPs fail closed even when private URLs are allowed."""
        calls = []
        cleared = []

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            return _make_browser_result(url=self.PUBLIC_URL)

        record = SimpleNamespace(
            ts=101.0,
            url=self.PUBLIC_URL,
            remote_ip="not an ip",
        )
        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=(record,)),
            clear_network_responses=lambda: cleared.append(True),
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda task_id: fake_supervisor)

        result = json.loads(browser_tool.browser_navigate(self.PUBLIC_URL, task_id="bad-ip"))

        assert result["success"] is False
        assert "browser connected to a malformed remote IP address" in result["error"]
        assert cleared == [True]
        assert ("bad-ip", "open", ["about:blank"]) in calls

    def test_failed_navigation_cannot_discard_prior_peer_violation(
        self, monkeypatch, _common_patches
    ):
        """A failed recovery open must keep the old unsafe page unreadable."""
        calls = []
        prior_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/internal",
            remote_ip="192.168.1.10",
        )
        records = [prior_record]

        def start_window():
            prior_records = tuple(records)
            records.clear()
            return prior_records

        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
            start_network_response_window=start_window,
        )

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            if command == "open" and args != ["about:blank"]:
                return {"success": False, "error": "Navigation failed"}
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(
            browser_tool,
            "_navigation_session_key",
            lambda task_id, url: task_id,
        )
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(
            browser_supervisor.SUPERVISOR_REGISTRY,
            "get",
            lambda task_id: fake_supervisor,
        )

        result = json.loads(
            browser_tool.browser_navigate(self.PUBLIC_URL, task_id="rebind")
        )

        assert result["success"] is False
        assert "browser connected to a private/internal address" in result["error"]
        assert ("rebind", "open", ["about:blank"]) in calls

    @pytest.mark.parametrize("action", ["click", "back", "press"])
    def test_cloud_blocks_peer_observed_during_navigation_action(
        self, monkeypatch, _common_patches, action
    ):
        """Navigation-capable actions cannot succeed after reaching a private peer."""
        calls = []
        cleared = []
        records = []
        unsafe_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/internal",
            remote_ip="192.168.1.10",
        )

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            if command == action:
                records.append(unsafe_record)
                return {"success": True, "data": {"url": self.PUBLIC_URL}}
            if command == "eval":
                return {"success": True, "data": {"result": self.PUBLIC_URL}}
            return _make_browser_result(url=self.PUBLIC_URL)

        def clear_responses():
            cleared.append(True)
            records.clear()

        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
            clear_network_responses=clear_responses,
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        monkeypatch.setattr(
            browser_supervisor.SUPERVISOR_REGISTRY,
            "get",
            lambda task_id: fake_supervisor,
        )

        if action == "click":
            result = json.loads(browser_tool.browser_click("@e1", task_id="rebind"))
        elif action == "back":
            result = json.loads(browser_tool.browser_back(task_id="rebind"))
        else:
            result = json.loads(browser_tool.browser_press("Enter", task_id="rebind"))

        assert result["success"] is False
        assert "browser connected to a private/internal address" in result["error"]
        assert cleared == [True]
        assert ("rebind", "open", ["about:blank"]) in calls

    @pytest.mark.parametrize("action", ["click", "back", "press"])
    def test_cloud_blocks_peer_observed_before_followup_snapshot_returns_content(
        self, monkeypatch, _common_patches, action
    ):
        """A late action response cannot leak through a later snapshot."""
        calls = []
        cleared = []
        records = []
        unsafe_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/internal",
            remote_ip="192.168.1.10",
        )

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            if command == "snapshot":
                records.append(unsafe_record)
                return {
                    "success": True,
                    "data": {
                        "snapshot": "unsafe private content",
                        "refs": {},
                    },
                }
            if command == "eval":
                return {"success": True, "data": {"result": self.PUBLIC_URL}}
            return {"success": True, "data": {"url": self.PUBLIC_URL}}

        def clear_responses():
            cleared.append(True)
            records.clear()

        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
            clear_network_responses=clear_responses,
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        monkeypatch.setattr(
            browser_supervisor.SUPERVISOR_REGISTRY,
            "get",
            lambda task_id: fake_supervisor,
        )

        if action == "click":
            action_result = json.loads(
                browser_tool.browser_click("@e1", task_id="late-rebind")
            )
        elif action == "back":
            action_result = json.loads(browser_tool.browser_back(task_id="late-rebind"))
        else:
            action_result = json.loads(
                browser_tool.browser_press("Enter", task_id="late-rebind")
            )
        snapshot_result = json.loads(browser_tool.browser_snapshot(task_id="late-rebind"))

        assert action_result["success"] is True
        assert snapshot_result["success"] is False
        assert "browser connected to a private/internal address" in snapshot_result["error"]
        assert "unsafe private content" not in json.dumps(snapshot_result)
        assert cleared == [True]
        assert ("late-rebind", "open", ["about:blank"]) in calls

    @pytest.mark.parametrize("content_tool", ["snapshot", "images"])
    def test_cloud_blocks_peer_before_content_tool_failure_is_returned(
        self, monkeypatch, _common_patches, content_tool
    ):
        """Browser-controlled failure text is content and must also be gated."""
        calls = []
        records = []
        unsafe_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/internal",
            remote_ip="192.168.1.10",
        )

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            if command in {"snapshot", "eval"}:
                records.append(unsafe_record)
                return {"success": False, "error": "PRIVATE_RESPONSE_BODY"}
            return {"success": True}

        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
        )

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(
            browser_supervisor.SUPERVISOR_REGISTRY,
            "get",
            lambda task_id: fake_supervisor,
        )

        if content_tool == "snapshot":
            result = json.loads(browser_tool.browser_snapshot(task_id="rebind"))
        else:
            result = json.loads(browser_tool.browser_get_images(task_id="rebind"))

        assert result["success"] is False
        assert "browser connected to a private/internal address" in result["error"]
        assert "PRIVATE_RESPONSE_BODY" not in json.dumps(result)
        assert ("rebind", "open", ["about:blank"]) in calls

    def test_content_sink_flushes_delayed_peer_event_before_return(
        self, monkeypatch, _common_patches
    ):
        """The supervisor barrier publishes queued peer events before policy reads."""
        calls = []
        records = []
        peer_event_pending = False
        unsafe_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/internal",
            remote_ip="192.168.1.10",
        )

        def flush_network_events():
            nonlocal peer_event_pending
            if peer_event_pending:
                records.append(unsafe_record)
                peer_event_pending = False
            return True

        fake_supervisor = SimpleNamespace(
            flush_network_events=flush_network_events,
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
        )

        def fake_run(task_id, command, args, **kwargs):
            nonlocal peer_event_pending
            calls.append((task_id, command, list(args)))
            if command == "snapshot":
                peer_event_pending = True
                return {
                    "success": True,
                    "data": {"snapshot": "PRIVATE_RESPONSE_BODY", "refs": {}},
                }
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(
            browser_supervisor.SUPERVISOR_REGISTRY,
            "get",
            lambda task_id: fake_supervisor,
        )

        result = json.loads(browser_tool.browser_snapshot(task_id="rebind"))

        assert result["success"] is False
        assert "browser connected to a private/internal address" in result["error"]
        assert "PRIVATE_RESPONSE_BODY" not in json.dumps(result)
        assert ("rebind", "open", ["about:blank"]) in calls

    def test_failed_blank_navigation_retains_peer_violation(
        self, monkeypatch, _common_patches
    ):
        """A failed about:blank open cannot make the unsafe page readable."""
        calls = []
        records = []
        unsafe_record = SimpleNamespace(
            ts=101.0,
            url="https://rebind.example/internal",
            remote_ip="192.168.1.10",
        )

        def start_window():
            prior_records = tuple(records)
            records.clear()
            return prior_records

        def retain_violation(remote_ip, url):
            records.append(SimpleNamespace(ts=102.0, url=url, remote_ip=remote_ip))

        fake_supervisor = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(network_responses=tuple(records)),
            start_network_response_window=start_window,
            retain_network_violation=retain_violation,
        )

        def fake_run(task_id, command, args, **kwargs):
            calls.append((task_id, command, list(args)))
            if command == "click":
                records.append(unsafe_record)
                return {"success": True}
            if command == "open":
                return {"success": False, "error": "blank failed"}
            if command == "snapshot":
                return {
                    "success": True,
                    "data": {"snapshot": "PRIVATE_RESPONSE_BODY", "refs": {}},
                }
            return {"success": True}

        monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
        monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
        monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
        monkeypatch.setattr(
            browser_supervisor.SUPERVISOR_REGISTRY,
            "get",
            lambda task_id: fake_supervisor,
        )

        click_result = json.loads(browser_tool.browser_click("@e1", task_id="rebind"))
        snapshot_result = json.loads(browser_tool.browser_snapshot(task_id="rebind"))

        assert click_result["success"] is False
        assert "browser connected to a private/internal address" in click_result["error"]
        assert snapshot_result["success"] is False
        assert "browser connected to a private/internal address" in snapshot_result["error"]
        assert "PRIVATE_RESPONSE_BODY" not in json.dumps(snapshot_result)
        assert calls.count(("rebind", "open", ["about:blank"])) == 2


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

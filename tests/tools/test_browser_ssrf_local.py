"""SSRF checks respect local-backend mode and the allow_private_urls setting.

Local backends (headless Chromium without a cloud provider, local terminal) skip the private-address
floor — the agent already has full local-network access via the terminal tool.

Cloud backends (Browserbase, Browser Use) enforce it by default. Users can opt out for cloud mode via
``browser.allow_private_urls: true``. Cloud metadata endpoints are refused unconditionally.
"""

import pytest

from tools import browser_session_cdp, browser_session_cloud, browser_url_policy, browser_use_cli
from tools.browser_url_policy import evaluate_url_safety


@pytest.fixture(autouse=True)
def _no_website_policy(monkeypatch):
    monkeypatch.setattr(browser_url_policy, "check_website_access", lambda url: None)


# ---------------------------------------------------------------------------
# Pre-navigation SSRF check
# ---------------------------------------------------------------------------


class TestPreNavigationSsrf:
    PRIVATE_URL = "http://127.0.0.1:8080/dashboard"

    # -- Cloud mode: SSRF active -----------------------------------------------

    def test_cloud_blocks_private_url_by_default(self, monkeypatch):
        """SSRF protection blocks private URLs in cloud mode."""
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: False)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: False)

        result = evaluate_url_safety(self.PRIVATE_URL)

        assert result["success"] is False
        assert "private or internal address" in result["error"]

    def test_cloud_allows_private_url_when_setting_true(self, monkeypatch):
        """Private URLs pass in cloud mode when allow_private_urls is True."""
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: False)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: True)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: False)

        assert evaluate_url_safety(self.PRIVATE_URL) is None

    # -- Local mode: SSRF skipped ----------------------------------------------

    def test_local_allows_private_url(self, monkeypatch):
        """Local backends skip SSRF — private URLs are always allowed."""
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: True)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: False)

        assert evaluate_url_safety(self.PRIVATE_URL) is None

    def test_local_allows_public_url(self, monkeypatch):
        """Local backends pass public URLs too (sanity check)."""
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: True)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: True)

        assert evaluate_url_safety("https://example.com") is None

    # -- Always-blocked floor: hybrid routing bypass regression (#16234) -------

    # Hybrid routing relaxes the private-address floor for URLs the local sidecar serves. An agent
    # running on EC2/GCP/Azure could otherwise reach 169.254.169.254 via the spawned local Chromium
    # sidecar and read IAM credentials. The always-blocked floor must fire regardless of routing.
    IMDS_URLS = [
        "http://169.254.169.254/latest/meta-data/",      # AWS / GCP / Azure / DO / Oracle
        "http://169.254.169.253/metadata/instance",        # Azure IMDS wire server
        "http://169.254.170.2/v2/credentials",             # AWS ECS task metadata
        "http://100.100.100.200/latest/meta-data/",        # Alibaba Cloud
        "http://metadata.google.internal/computeMetadata/v1/",  # GCP hostname
    ]

    @pytest.mark.parametrize("imds_url", IMDS_URLS)
    def test_cloud_blocks_imds_even_when_routing_to_local_sidecar(self, monkeypatch, imds_url):
        """Hybrid routing must not let cloud metadata endpoints through."""
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: False)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
        # Hybrid routing kicks in for this URL (cloud provider configured, private URL → local sidecar).
        monkeypatch.setattr(browser_url_policy, "hybrid_routes_locally", lambda url: True)
        # is_safe_url would catch IMDS, but the sidecar relaxes it. Force it True so the test pins the
        # always-blocked floor as an independent gate.
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: True)

        verdict = browser_use_cli._code_url_verdict(f'new_tab("{imds_url}")')

        assert verdict.blocked is not None
        assert "cloud metadata endpoint" in verdict.blocked

    def test_cloud_allows_ordinary_private_url_via_sidecar(self, monkeypatch):
        """Hybrid routing still works for ordinary private URLs — floor
        must be narrow enough to not break the PR #16136 feature."""
        monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: False)
        monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
        monkeypatch.setattr(browser_url_policy, "hybrid_routes_locally", lambda url: True)
        monkeypatch.setattr(browser_url_policy, "is_safe_url", lambda url: False)

        for private in (
            "http://127.0.0.1:8080/dashboard",
            "http://192.168.1.1/admin",
            "http://10.0.0.5/",
            "http://myservice.local/",
        ):
            verdict = browser_use_cli._code_url_verdict(f'new_tab("{private}")')
            assert verdict.blocked is None, f"Unexpected block for {private}: {verdict.blocked}"
            assert verdict.local_sidecar is True


# ---------------------------------------------------------------------------
# is_local_backend() unit tests
# ---------------------------------------------------------------------------


class TestIsLocalBackend:
    def test_no_cloud_provider_is_local(self, monkeypatch):
        """No cloud provider configured → local backend."""
        monkeypatch.setattr(browser_session_cdp, "get_cdp_override_raw", lambda: "")
        monkeypatch.setattr(browser_session_cloud, "get_cloud_provider", lambda: None)

        assert browser_session_cloud.is_local_backend() is True


class TestAllowPrivateUrlsConfig:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        browser_session_cloud.reset_caches()
        yield
        browser_session_cloud.reset_caches()

    def test_browser_config_string_false_stays_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config_effective.load_user_config_effective",
            lambda: {"browser": {"allow_private_urls": "false"}},
        )

        assert browser_session_cloud.allow_private_urls() is False

    @pytest.mark.parametrize(
        "profile_order",
        [("allowed", "blocked"), ("blocked", "allowed")],
        ids=["allowed-then-blocked", "blocked-then-allowed"],
    )
    def test_profile_scoped_config_does_not_reuse_another_profiles_opt_out(
        self, tmp_path, profile_order
    ):
        """The browser's independent guard must follow the active profile."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        allowed_home = tmp_path / "allowed"
        blocked_home = tmp_path / "blocked"
        allowed_home.mkdir()
        blocked_home.mkdir()
        (allowed_home / "config.yaml").write_text(
            "browser:\n  allow_private_urls: true\n", encoding="utf-8"
        )
        (blocked_home / "config.yaml").write_text(
            "browser:\n  allow_private_urls: false\n", encoding="utf-8"
        )

        def under_profile(home):
            token = set_hermes_home_override(home)
            try:
                return browser_session_cloud.allow_private_urls()
            finally:
                reset_hermes_home_override(token)

        homes = {"allowed": allowed_home, "blocked": blocked_home}
        expected = {"allowed": True, "blocked": False}
        for profile in profile_order:
            assert under_profile(homes[profile]) is expected[profile]

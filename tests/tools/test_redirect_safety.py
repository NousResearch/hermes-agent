"""Tests for tools/redirect_safety.py — safe redirect validation."""

import pytest

from tools.redirect_safety import (
    safe_redirect_url,
    _normalize_origin,
    _matches_wildcard_domain,
    _is_allowed_origin,
)


class TestNormalizeOrigin:
    """Test origin extraction and normalization."""
    
    def test_extracts_https_origin(self):
        assert _normalize_origin("https://example.com/path") == "https://example.com"
    
    def test_extracts_http_origin(self):
        assert _normalize_origin("http://example.com/path") == "http://example.com"
    
    def test_preserves_non_standard_port(self):
        assert _normalize_origin("https://example.com:8443/path") == "https://example.com:8443"
    
    def test_handles_subdomain(self):
        assert _normalize_origin("https://api.example.com/path") == "https://api.example.com"
    
    def test_returns_empty_for_relative_url(self):
        assert _normalize_origin("/path/to/resource") == ""
    
    def test_returns_empty_for_missing_scheme(self):
        assert _normalize_origin("example.com/path") == ""
    
    def test_handles_url_with_query_and_fragment(self):
        assert _normalize_origin("https://example.com/path?q=1#hash") == "https://example.com"


class TestMatchesWildcardDomain:
    """Test wildcard domain matching logic."""
    
    def test_exact_match_without_wildcard(self):
        assert _matches_wildcard_domain("example.com", "example.com")
    
    def test_no_match_without_wildcard(self):
        assert not _matches_wildcard_domain("api.example.com", "example.com")
    
    def test_wildcard_matches_subdomain(self):
        assert _matches_wildcard_domain("api.example.com", "*.example.com")
    
    def test_wildcard_matches_nested_subdomain(self):
        assert _matches_wildcard_domain("v1.api.example.com", "*.example.com")
    
    def test_wildcard_does_not_match_root_domain(self):
        # *.example.com should not match example.com itself
        assert not _matches_wildcard_domain("example.com", "*.example.com")
    
    def test_wildcard_does_not_match_different_domain(self):
        assert not _matches_wildcard_domain("evil.com", "*.example.com")
    
    def test_wildcard_does_not_match_partial_suffix(self):
        # Should not match if the suffix is just a substring
        assert not _matches_wildcard_domain("fakeexample.com", "*.example.com")


class TestIsAllowedOrigin:
    """Test origin allowlist matching."""
    
    def test_exact_origin_match(self):
        assert _is_allowed_origin(
            "https://example.com/path",
            ["https://example.com"],
        )
    
    def test_no_match_different_origin(self):
        assert not _is_allowed_origin(
            "https://evil.com/path",
            ["https://example.com"],
        )
    
    def test_wildcard_subdomain_match(self):
        assert _is_allowed_origin(
            "https://api.example.com/path",
            ["https://*.example.com"],
        )
    
    def test_wildcard_does_not_match_root(self):
        assert not _is_allowed_origin(
            "https://example.com/path",
            ["https://*.example.com"],
        )
    
    def test_scheme_must_match(self):
        assert not _is_allowed_origin(
            "http://example.com/path",
            ["https://example.com"],
        )
    
    def test_port_must_match_when_specified(self):
        assert not _is_allowed_origin(
            "https://example.com:8443/path",
            ["https://example.com:9000"],
        )
    
    def test_matches_when_port_matches(self):
        assert _is_allowed_origin(
            "https://example.com:8443/path",
            ["https://example.com:8443"],
        )
    
    def test_multiple_allowlist_entries(self):
        allowed = [
            "https://app.example.com",
            "https://api.example.com",
            "https://*.trusted.com",
        ]
        
        assert _is_allowed_origin("https://app.example.com/path", allowed)
        assert _is_allowed_origin("https://api.example.com/path", allowed)
        assert _is_allowed_origin("https://v1.trusted.com/path", allowed)
        assert not _is_allowed_origin("https://evil.com/path", allowed)


class TestSafeRedirectUrl:
    """Test the main safe_redirect_url function."""
    
    BASE_URL = "https://app.example.com"
    
    # Relative URLs
    
    def test_allows_relative_path(self):
        result = safe_redirect_url("/dashboard", self.BASE_URL)
        assert result == "https://app.example.com/dashboard"
    
    def test_allows_relative_path_with_query(self):
        result = safe_redirect_url("/search?q=test", self.BASE_URL)
        assert result == "https://app.example.com/search?q=test"
    
    def test_allows_relative_path_with_fragment(self):
        result = safe_redirect_url("/page#section", self.BASE_URL)
        assert result == "https://app.example.com/page#section"
    
    def test_allows_query_only_relative_url(self):
        result = safe_redirect_url("?next=value", self.BASE_URL)
        assert result == "https://app.example.com?next=value"
    
    # Same-origin absolute URLs
    
    def test_allows_same_origin_absolute_url(self):
        result = safe_redirect_url("https://app.example.com/dashboard", self.BASE_URL)
        assert result == "https://app.example.com/dashboard"
    
    def test_allows_same_origin_with_port(self):
        base_with_port = "https://app.example.com:8443"
        result = safe_redirect_url(
            "https://app.example.com:8443/dashboard",
            base_with_port,
        )
        assert result == "https://app.example.com:8443/dashboard"
    
    # Cross-origin blocking (default, no allowlist)
    
    def test_blocks_cross_origin_without_allowlist(self):
        result = safe_redirect_url("https://evil.com/phishing", self.BASE_URL)
        assert result == "/"
    
    def test_blocks_subdomain_without_allowlist(self):
        result = safe_redirect_url("https://api.example.com/endpoint", self.BASE_URL)
        assert result == "/"
    
    def test_blocks_different_scheme_without_allowlist(self):
        result = safe_redirect_url("http://app.example.com/page", self.BASE_URL)
        assert result == "/"
    
    def test_blocks_different_port_without_allowlist(self):
        result = safe_redirect_url("https://app.example.com:8443/page", self.BASE_URL)
        assert result == "/"
    
    # Cross-origin with allowlist
    
    def test_allows_cross_origin_in_allowlist(self):
        result = safe_redirect_url(
            "https://docs.example.com/guide",
            self.BASE_URL,
            allowed_origins=["https://docs.example.com"],
        )
        assert result == "https://docs.example.com/guide"
    
    def test_allows_wildcard_subdomain_in_allowlist(self):
        result = safe_redirect_url(
            "https://api.example.com/endpoint",
            self.BASE_URL,
            allowed_origins=["https://*.example.com"],
        )
        assert result == "https://api.example.com/endpoint"
    
    def test_blocks_cross_origin_not_in_allowlist(self):
        result = safe_redirect_url(
            "https://evil.com/phishing",
            self.BASE_URL,
            allowed_origins=["https://docs.example.com"],
        )
        assert result == "/"
    
    # Scheme validation
    
    def test_blocks_javascript_scheme(self):
        result = safe_redirect_url("javascript:alert(1)", self.BASE_URL)
        assert result == "/"
    
    def test_blocks_data_scheme(self):
        result = safe_redirect_url("data:text/html,<script>alert(1)</script>", self.BASE_URL)
        assert result == "/"
    
    def test_blocks_file_scheme(self):
        result = safe_redirect_url("file:///etc/passwd", self.BASE_URL)
        assert result == "/"
    
    def test_allows_custom_scheme_when_specified(self):
        result = safe_redirect_url(
            "custom://app/path",
            self.BASE_URL,
            allowed_schemes=frozenset({"custom"}),
            allowed_origins=["custom://app"],
        )
        # Note: this will fail same-origin check, but should pass scheme check
        # and be allowed via the allowlist
        assert result == "custom://app/path"
    
    # Custom fallback URL
    
    def test_uses_custom_fallback_on_block(self):
        result = safe_redirect_url(
            "https://evil.com/phishing",
            self.BASE_URL,
            fallback_url="/home",
        )
        assert result == "/home"
    
    # Edge cases and error handling
    
    def test_handles_empty_redirect_url(self):
        result = safe_redirect_url("", self.BASE_URL)
        assert result == "/"
    
    def test_handles_none_redirect_url(self):
        result = safe_redirect_url(None, self.BASE_URL)  # type: ignore
        assert result == "/"
    
    def test_handles_whitespace_only_redirect_url(self):
        result = safe_redirect_url("   ", self.BASE_URL)
        assert result == "/"
    
    def test_strips_whitespace_from_valid_url(self):
        result = safe_redirect_url("  /dashboard  ", self.BASE_URL)
        assert result == "https://app.example.com/dashboard"
    
    def test_handles_malformed_url(self):
        # ://malformed parses as a relative path (no scheme/netloc)
        # This is treated as a weird but harmless relative URL
        result = safe_redirect_url("://malformed", self.BASE_URL)
        assert result == "https://app.example.com/:/malformed"
    
    def test_handles_invalid_base_url(self):
        result = safe_redirect_url("/dashboard", "not-a-url")
        assert result == "/"
    
    # Protocol-relative URLs (edge case)
    
    def test_handles_protocol_relative_url(self):
        # //evil.com is a protocol-relative URL that inherits the scheme
        # After parsing, it becomes an absolute URL with netloc="evil.com"
        result = safe_redirect_url("//evil.com/phishing", self.BASE_URL)
        # This should be treated as cross-origin and blocked
        assert result == "/"
    
    # URL encoding and special characters
    
    def test_allows_url_encoded_path(self):
        result = safe_redirect_url("/path%20with%20spaces", self.BASE_URL)
        assert result == "https://app.example.com/path%20with%20spaces"
    
    def test_allows_special_chars_in_query(self):
        result = safe_redirect_url("/search?q=test&filter=1", self.BASE_URL)
        assert result == "https://app.example.com/search?q=test&filter=1"
    
    # Unicode and internationalized domains
    
    def test_handles_unicode_path(self):
        result = safe_redirect_url("/路径", self.BASE_URL)
        # urljoin should handle this correctly
        assert "/路径" in result or "/%E8" in result  # Either raw or percent-encoded
    
    # Multiple validation passes
    
    def test_allowlist_does_not_bypass_scheme_check(self):
        # Even if origin is in allowlist, scheme must be allowed
        result = safe_redirect_url(
            "javascript://docs.example.com/xss",
            self.BASE_URL,
            allowed_origins=["javascript://docs.example.com"],
        )
        assert result == "/"
    
    # Real-world attack scenarios
    
    def test_blocks_open_redirect_to_attacker_site(self):
        # Typical open redirect attack
        result = safe_redirect_url("https://attacker.com/phishing", self.BASE_URL)
        assert result == "/"
    
    def test_blocks_open_redirect_with_legitimate_looking_path(self):
        # Attacker tries to make URL look legitimate with path
        result = safe_redirect_url(
            "https://evil.com/app.example.com/login",
            self.BASE_URL,
        )
        assert result == "/"
    
    def test_blocks_homograph_attack_attempt(self):
        # Using similar-looking characters (though URL encoding handles this)
        result = safe_redirect_url("https://examp1e.com/dashboard", self.BASE_URL)
        assert result == "/"


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_oauth_callback_redirect(self):
        # OAuth flow: redirect user back to where they came from
        base = "https://auth.example.com"
        
        # Same-origin callback
        result = safe_redirect_url("/oauth/callback?code=xyz", base)
        assert result == "https://auth.example.com/oauth/callback?code=xyz"
        
        # Cross-origin callback to main app (in allowlist)
        result = safe_redirect_url(
            "https://app.example.com/dashboard",
            base,
            allowed_origins=["https://app.example.com"],
        )
        assert result == "https://app.example.com/dashboard"
        
        # Malicious redirect attempt
        result = safe_redirect_url("https://evil.com/steal-token", base)
        assert result == "/"
    
    def test_multi_tenant_saas_with_subdomains(self):
        # SaaS app where each tenant has a subdomain
        base = "https://app.example.com"
        
        # Allow redirects to any tenant subdomain
        result = safe_redirect_url(
            "https://tenant1.example.com/dashboard",
            base,
            allowed_origins=["https://*.example.com"],
        )
        assert result == "https://tenant1.example.com/dashboard"
        
        result = safe_redirect_url(
            "https://tenant2.example.com/settings",
            base,
            allowed_origins=["https://*.example.com"],
        )
        assert result == "https://tenant2.example.com/settings"
        
        # Block external domains
        result = safe_redirect_url(
            "https://evil.com",
            base,
            allowed_origins=["https://*.example.com"],
        )
        assert result == "/"
    
    def test_microservices_architecture(self):
        # Multiple services on different subdomains
        base = "https://api.example.com"
        allowed = [
            "https://api.example.com",
            "https://auth.example.com",
            "https://billing.example.com",
        ]
        
        # Allow redirects between services
        for service_url in allowed:
            result = safe_redirect_url(
                f"{service_url}/endpoint",
                base,
                allowed_origins=allowed,
            )
            assert result == f"{service_url}/endpoint"
        
        # Block external domains
        result = safe_redirect_url(
            "https://evil.com",
            base,
            allowed_origins=allowed,
        )
        assert result == "/"
    
    def test_logout_redirect_with_return_url(self):
        # Common pattern: logout and redirect back to public page
        base = "https://app.example.com"
        
        # Safe return URL
        result = safe_redirect_url(
            "/landing-page",
            base,
            fallback_url="/",
        )
        assert result == "https://app.example.com/landing-page"
        
        # Attacker-controlled return URL
        result = safe_redirect_url(
            "https://evil.com/fake-login",
            base,
            fallback_url="/",
        )
        assert result == "/"


class TestAliases:
    """Test function aliases."""
    
    BASE_URL = "https://app.example.com"
    
    def test_getSafeRedirect_alias(self):
        from tools.redirect_safety import getSafeRedirect
        
        # Verify the alias works
        result = getSafeRedirect("/dashboard", self.BASE_URL)
        assert result == "https://app.example.com/dashboard"
        
        # Verify it blocks javascript: URLs
        result = getSafeRedirect("javascript:alert(1)", self.BASE_URL)
        assert result == "/"

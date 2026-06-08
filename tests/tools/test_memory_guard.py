"""Tests for memory guard — URL, sensitive paths, structural content blocking.

These tests cover the new patterns added to tools/threat_patterns.py
for memory write protection (strict scope).
"""

import pytest

from tools.threat_patterns import (
    _check_structural_content,
    first_threat_message,
    scan_for_threats,
)


# =========================================================================
# URL blocking (strict scope only)
# =========================================================================


class TestUrlBlocking:
    """URLs are not allowed in memory writes (strict scope)."""

    def test_http_url_blocked(self):
        msg = first_threat_message("Visit http://example.com for details", scope="strict")
        assert msg is not None
        assert "URL" in msg or "url" in msg.lower()

    def test_https_url_blocked(self):
        msg = first_threat_message("See https://github.com/user/repo", scope="strict")
        assert msg is not None
        assert "URL" in msg or "url" in msg.lower()

    def test_url_not_blocked_at_context_scope(self):
        # URLs are only blocked at strict scope (memory/skill writes)
        assert first_threat_message("Visit http://example.com", scope="context") is None

    def test_no_url_passes(self):
        assert first_threat_message("User prefers dark mode", scope="strict") is None


# =========================================================================
# Sensitive paths (strict scope)
# =========================================================================


class TestSensitivePaths:
    """Sensitive file paths are blocked in memory writes."""

    def test_env_file_blocked(self):
        msg = first_threat_message("Config in /home/user/.env", scope="strict")
        assert msg is not None
        assert ".env" in msg or "sensitive" in msg.lower()

    def test_env_local_blocked(self):
        msg = first_threat_message("See .env.local for secrets", scope="strict")
        assert msg is not None

    def test_ssh_dir_blocked(self):
        msg = first_threat_message("Keys in ~/.ssh/authorized_keys", scope="strict")
        assert msg is not None
        # Caught by either ssh_backdoor or sensitive_path_ssh
        assert ".ssh" in msg or "sensitive" in msg.lower() or "ssh_backdoor" in msg

    def test_npmrc_blocked(self):
        msg = first_threat_message("npm config at ~/.npmrc", scope="strict")
        assert msg is not None

    def test_credentials_file_blocked(self):
        msg = first_threat_message("AWS creds in /path/credentials.json", scope="strict")
        assert msg is not None

    def test_appdata_blocked(self):
        msg = first_threat_message("Config in C:\\Users\\user\\AppData\\Local", scope="strict")
        assert msg is not None
        assert "AppData" in msg or "sensitive" in msg.lower()

    def test_sensitive_keyword_blocked(self):
        msg = first_threat_message("File at /path/secret/config.json", scope="strict")
        assert msg is not None

    # ── False positive regression: reviewer-reported normal paths ──

    def test_config_path_not_blocked(self):
        """config as a path segment should NOT trigger sensitive_path_keyword."""
        msg = first_threat_message("File at /home/user/config/app.py", scope="strict")
        assert msg is None

    def test_secret_santa_not_blocked(self):
        """secret-santa is a compound word, not a sensitive path."""
        msg = first_threat_message("Data at /opt/secret-santa/data", scope="strict")
        assert msg is None

    def test_token_spec_not_blocked(self):
        """token-spec is a compound word, not a sensitive path."""
        msg = first_threat_message("Doc at /usr/share/doc/token-spec", scope="strict")
        assert msg is None

    def test_apikey_management_not_blocked(self):
        """apikey-management is a compound word, not a sensitive path."""
        msg = first_threat_message("App at /var/app/apikey-management/", scope="strict")
        assert msg is None

    # ── Positive: must still block ──

    def test_secrets_dir_blocked(self):
        msg = first_threat_message("Config at /app/secrets/db.yaml", scope="strict")
        assert msg is not None

    def test_credentials_dir_blocked(self):
        msg = first_threat_message("Creds at /app/credentials/aws", scope="strict")
        assert msg is not None

    def test_private_key_file_blocked(self):
        msg = first_threat_message("Key at /home/user/.ssh/id_rsa", scope="strict")
        assert msg is not None

    def test_env_file_blocked(self):
        msg = first_threat_message("Env at /project/.env", scope="strict")
        assert msg is not None


# =========================================================================
# Structural content (code blocks, log dumps)
# =========================================================================


class TestStructuralContent:
    """Large code blocks and log dumps are blocked."""

    def test_small_code_block_passes(self):
        content = "Some text\n```python\nprint('hello')\n```\nMore text"
        assert _check_structural_content(content) is None

    def test_large_code_block_blocked(self):
        lines = ["Some text", "```python"]
        for i in range(15):
            lines.append(f"x = {i}")
        lines.append("```")
        content = "\n".join(lines)
        msg = _check_structural_content(content)
        assert msg is not None
        assert "code block" in msg.lower()

    def test_traceback_dump_blocked(self):
        lines = ["Error occurred:"]
        for i in range(20):
            lines.append(f'  File "module_{i}.py", line {i}')
            lines.append("    some_code()")
        lines.append("Traceback (most recent call last):")
        lines.append("Traceback (most recent call last):")
        content = "\n".join(lines)
        msg = _check_structural_content(content)
        assert msg is not None
        assert "log" in msg.lower() or "traceback" in msg.lower()

    def test_short_text_passes(self):
        assert _check_structural_content("Short note about settings") is None

    def test_empty_content_passes(self):
        assert _check_structural_content("") is None


# =========================================================================
# Integration — first_threat_message catches all categories
# =========================================================================


class TestMemoryGuardIntegration:
    """Verify first_threat_message catches all memory guard categories."""

    def test_url_category_message(self):
        msg = first_threat_message("Link: https://example.com", scope="strict")
        assert msg is not None
        assert "URL" in msg or "url" in msg.lower()

    def test_sensitive_path_category_message(self):
        msg = first_threat_message("See /path/.env file", scope="strict")
        assert msg is not None
        assert "sensitive" in msg.lower() or ".env" in msg

    def test_code_block_category_message(self):
        lines = ["```python"]
        for i in range(15):
            lines.append(f"x = {i}")
        lines.append("```")
        msg = first_threat_message("\n".join(lines), scope="strict")
        assert msg is not None
        assert "code block" in msg.lower()

    def test_normal_memory_passes(self):
        """Normal memory content should pass all guards."""
        content = """User preferences:
- Dark mode enabled
- Python 3.12 with FastAPI
- Uses pytest for testing
- Prefers concise responses"""
        assert first_threat_message(content, scope="strict") is None

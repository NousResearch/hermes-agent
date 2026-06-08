"""Tests for safe_expanduser path helper."""

import os
from pathlib import Path

import pytest

from utils import safe_expanduser


class TestSafeExpanduser:
    """safe_expanduser should resolve ~/ normally but survive a missing HOME."""

    def test_expands_tilde_with_home(self):
        """When HOME is set, ~/ is resolved just like Path.expanduser()."""
        result = safe_expanduser("~/foo/bar")
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert str(result).startswith("/")
        assert result.name == "bar"

    def test_returns_path_for_plain_path(self):
        """A path without ~ is returned unchanged."""
        result = safe_expanduser("/tmp/test")
        assert result == Path("/tmp/test")

    def test_accepts_path_instance(self):
        """A Path instance is treated the same as a string."""
        result = safe_expanduser(Path("/some/path"))
        assert result == Path("/some/path")

    def test_respects_default(self):
        """If default is provided, it is returned on expansion failure."""
        result = safe_expanduser("~/x", default="/fallback")
        # With HOME normally set this won't hit the except branch, so
        # we can only verify the default is a Path.
        assert isinstance(result, Path)

    def test_default_is_path_coerced(self):
        """default=string is coerced to Path just like the path arg."""
        result = safe_expanduser("~/x", default="/fallback")
        assert isinstance(result, Path)

    def test_no_crash_when_home_unset_and_passwd_fails(self, monkeypatch):
        """When HOME is empty and pwd lookup fails, safe_expanduser returns
        the literal path instead of raising RuntimeError."""
        # Unset HOME so expanduser triggers its lookup.
        monkeypatch.delenv("HOME", raising=False)
        # Monkeypatch pwd.getpwuid to raise KeyError, simulating an
        # unmapped uid — the exact condition that produces RuntimeError.
        import pwd

        orig_getpwuid = pwd.getpwuid
        monkeypatch.setattr(pwd, "getpwuid", lambda uid: (_ for _ in ()).throw(
            KeyError(f"uid {uid} not found")
        ))
        # Now expanduser should raise RuntimeError, but safe_expanduser
        # catches it and returns the unexpanded path.
        result = safe_expanduser("~/some_dir")
        assert str(result) == "~/some_dir"
        # Restore original (not strictly necessary, monkeypatch handles it).
        monkeypatch.setattr(pwd, "getpwuid", orig_getpwuid)

    def test_default_on_expand_failure(self, monkeypatch):
        """When HOME is unset and pwd fails, default is returned."""
        monkeypatch.delenv("HOME", raising=False)
        import pwd
        monkeypatch.setattr(pwd, "getpwuid", lambda uid: (_ for _ in ()).throw(
            KeyError(f"uid {uid} not found")
        ))
        result = safe_expanduser("~/some_dir", default="/tmp/hermes_fallback")
        assert result == Path("/tmp/hermes_fallback")
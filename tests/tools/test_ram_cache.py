"""Tests for tools.ram_cache — RAM-backed tool result cache."""

import json
import os
import platform
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

# Ensure the project root is on the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.ram_cache import (
    CACHE_ROOT,
    cache_clear,
    cache_get,
    cache_invalidate,
    cache_set,
    cache_stats,
    check_cache_before_dispatch,
    file_cache_get,
    file_cache_set,
    is_read_only_command,
    store_cache_after_dispatch,
    _default_cache_root,
    _make_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """Give each test its own cache directory to avoid parallel worker conflicts."""
    import tools.ram_cache as rc
    monkeypatch.setattr(rc, "CACHE_ROOT", tmp_path / "hermes-cache")
    monkeypatch.setattr(rc, "ENABLED", True)
    yield


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

class TestCacheGetSet:
    """Tests for basic cache get/set operations."""

    def test_set_and_get(self):
        assert cache_set("test", ("key1",), "hello world")
        assert cache_get("test", ("key1",)) == "hello world"

    def test_get_miss(self):
        assert cache_get("test", ("nonexistent",)) is None

    def test_set_complex_value(self):
        value = {"data": [1, 2, 3], "nested": {"key": "value"}}
        cache_set("test", ("complex",), value)
        assert cache_get("test", ("complex",)) == value

    def test_set_unicode(self):
        cache_set("test", ("unicode",), "héllo wörld 日本語")
        assert cache_get("test", ("unicode",)) == "héllo wörld 日本語"

    def test_ttl_expiry(self):
        cache_set("test", ("expire",), "short-lived", ttl=1)
        assert cache_get("test", ("expire",), ttl=1) == "short-lived"
        time.sleep(1.1)
        assert cache_get("test", ("expire",), ttl=1) is None

    def test_different_keys_independent(self):
        cache_set("test", ("a",), "value_a")
        cache_set("test", ("b",), "value_b")
        assert cache_get("test", ("a",)) == "value_a"
        assert cache_get("test", ("b",)) == "value_b"

    def test_different_categories_independent(self):
        cache_set("cat1", ("key",), "val1")
        cache_set("cat2", ("key",), "val2")
        assert cache_get("cat1", ("key",)) == "val1"
        assert cache_get("cat2", ("key",)) == "val2"

    def test_overwrite(self):
        cache_set("test", ("key",), "old")
        cache_set("test", ("key",), "new")
        assert cache_get("test", ("key",)) == "new"

    def test_none_value(self):
        # None is a valid cached value — distinct from "not found"
        cache_set("test", ("null",), None)
        # cache_get returns None for both "not found" and "cached None"
        # This is a known limitation — documented behavior
        result = cache_get("test", ("null",))
        assert result is None  # Can't distinguish cached None from miss

    def test_large_value(self):
        large = "x" * 100_000
        cache_set("test", ("large",), large)
        assert cache_get("test", ("large",)) == large

    def test_never_raises_on_get(self):
        """cache_get should never raise, even with corrupted data."""
        # Write invalid JSON
        p = CACHE_ROOT / "test" / "corrupt.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not valid json{{{")
        # Should return None, not raise
        assert cache_get("test", ("corrupt",)) is None


class TestCacheInvalidate:
    def test_invalidate_existing(self):
        cache_set("test", ("key",), "value")
        assert cache_invalidate("test", ("key",))
        assert cache_get("test", ("key",)) is None

    def test_invalidate_nonexistent(self):
        assert cache_invalidate("test", ("nonexistent",))


class TestCacheClear:
    def test_clear_category(self):
        cache_set("cat1", ("a",), "1")
        cache_set("cat1", ("b",), "2")
        cache_set("cat2", ("c",), "3")
        assert cache_clear("cat1") == 2
        assert cache_get("cat1", ("a",)) is None
        assert cache_get("cat2", ("c",)) == "3"  # other category untouched

    def test_clear_all(self):
        cache_set("cat1", ("a",), "1")
        cache_set("cat2", ("b",), "2")
        assert cache_clear() >= 2
        assert cache_get("cat1", ("a",)) is None
        assert cache_get("cat2", ("b",)) is None


class TestCacheStats:
    def test_empty_stats(self):
        stats = cache_stats()
        assert stats["total_entries"] == 0
        assert stats["enabled"] is True

    def test_stats_after_writes(self):
        cache_set("file", ("a",), "data1")
        cache_set("cmd", ("b",), "data2")
        stats = cache_stats()
        assert stats["total_entries"] == 2
        assert stats["total_bytes"] > 0
        assert "file" in stats["categories"]
        assert "cmd" in stats["categories"]

    def test_hit_rate_tracking(self):
        cache_set("test", ("key",), "value")
        cache_get("test", ("key",))    # hit
        cache_get("test", ("miss",))   # miss
        stats = cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


# ---------------------------------------------------------------------------
# File cache (mtime-validated)
# ---------------------------------------------------------------------------

class TestFileCache:
    def test_file_cache_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            file_cache_set(path, 1, 500, '{"content": "test"}')
            assert file_cache_get(path, 1, 500) == '{"content": "test"}'
        finally:
            os.unlink(path)

    def test_file_cache_invalidated_on_modify(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("original")
            path = f.name
        try:
            file_cache_set(path, 1, 500, '{"content": "original"}')
            assert file_cache_get(path, 1, 500) is not None
            # Modify the file
            time.sleep(0.05)  # ensure mtime changes
            with open(path, "w") as f:
                f.write("modified")
            # Cache should miss (mtime changed)
            assert file_cache_get(path, 1, 500) is None
        finally:
            os.unlink(path)

    def test_file_cache_nonexistent_file(self):
        assert file_cache_get("/nonexistent/path.txt", 1, 500) is None
        assert file_cache_set("/nonexistent/path.txt", 1, 500, "data") is False

    def test_file_cache_different_offsets(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content\n" * 100)
            path = f.name
        try:
            file_cache_set(path, 1, 10, "first_10")
            file_cache_set(path, 11, 20, "next_10")
            assert file_cache_get(path, 1, 10) == "first_10"
            assert file_cache_get(path, 11, 20) == "next_10"
            assert file_cache_get(path, 1, 20) is None  # different params = miss
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Read-only command detection
# ---------------------------------------------------------------------------

class TestReadOnlyDetection:
    """Tests for is_read_only_command."""

    @pytest.mark.parametrize("cmd", [
        "ls -la /tmp",
        "df -h",
        "nvidia-smi",
        "ps aux",
        "git log --oneline -5",
        "cat /etc/hostname",
        "grep -r 'pattern' /src",
        "kubectl get pods",
        "docker ps -a",
        "systemctl status nginx",
        "pip list",
        "python3 --version",
        "free -h",
        "uptime",
        "uname -a",
    ])
    def test_read_only_commands(self, cmd):
        assert is_read_only_command(cmd), f"Should be read-only: {cmd}"

    @pytest.mark.parametrize("cmd", [
        "apt install nginx",
        "rm -rf /tmp/test",
        "pip install requests",
        "docker run ubuntu",
        "systemctl restart nginx",
        "systemctl stop nginx",
        "mv file1 file2",
        "cp src dst",
        "chmod 755 file",
        "chown root file",
        "mkdir /tmp/new",
        "touch /tmp/new",
        "tee /tmp/file",
        "kill -9 1234",
    ])
    def test_mutating_commands(self, cmd):
        assert not is_read_only_command(cmd), f"Should NOT be read-only: {cmd}"

    def test_sudo_prefix_stripped(self):
        assert is_read_only_command("sudo df -h")
        assert is_read_only_command("sudo nvidia-smi")
        assert not is_read_only_command("sudo rm -rf /")

    def test_env_var_prefix_stripped(self):
        assert is_read_only_command("LANG=C ls -la")

    def test_command_substitution_rejected(self):
        """Commands with $() or backticks should not be cached."""
        assert not is_read_only_command("echo $(date)")
        assert not is_read_only_command("echo `date`")
        assert not is_read_only_command("ls $(pwd)")

    def test_random_rejected(self):
        assert not is_read_only_command("echo $RANDOM")

    def test_empty_command(self):
        assert not is_read_only_command("")
        assert not is_read_only_command("   ")


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

class TestDispatchIntegration:
    """Tests for check_cache_before_dispatch and store_cache_after_dispatch."""

    def test_read_file_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            # No cache yet
            assert check_cache_before_dispatch("read_file", {"path": path}) is None
            # Store result
            store_cache_after_dispatch("read_file", {"path": path},
                                       json.dumps({"content": "hello"}))
            # Now should hit
            result = check_cache_before_dispatch("read_file", {"path": path})
            assert result is not None
            assert "hello" in result
        finally:
            os.unlink(path)

    def test_terminal_read_only_cached(self):
        assert check_cache_before_dispatch("terminal", {"command": "df -h"}) is None
        store_cache_after_dispatch("terminal", {"command": "df -h"},
                                   json.dumps({"output": "Filesystem..."}))
        result = check_cache_before_dispatch("terminal", {"command": "df -h"})
        assert result is not None

    def test_terminal_mutating_not_cached(self):
        store_cache_after_dispatch("terminal", {"command": "apt install foo"},
                                   json.dumps({"output": "done"}))
        assert check_cache_before_dispatch("terminal", {"command": "apt install foo"}) is None

    def test_search_files_cached(self):
        args = {"pattern": "*.py", "target": "files", "directory": "/tmp"}
        assert check_cache_before_dispatch("search_files", args) is None
        store_cache_after_dispatch("search_files", args, json.dumps({"files": []}))
        assert check_cache_before_dispatch("search_files", args) is not None

    def test_error_results_not_cached(self):
        store_cache_after_dispatch("read_file", {"path": "/nope"},
                                   json.dumps({"error": "File not found"}))
        assert check_cache_before_dispatch("read_file", {"path": "/nope"}) is None

    def test_unknown_tool_passthrough(self):
        """Unknown tools should not be cached."""
        assert check_cache_before_dispatch("some_custom_tool", {"arg": 1}) is None


# ---------------------------------------------------------------------------
# Cross-platform
# ---------------------------------------------------------------------------

class TestCrossPlatform:
    def test_default_root_linux(self):
        with mock.patch("tools.ram_cache.platform.system", return_value="Linux"):
            root = _default_cache_root()
            assert "shm" in str(root) or "tmp" in str(root)

    def test_default_root_macos(self):
        with mock.patch("tools.ram_cache.platform.system", return_value="Darwin"):
            root = _default_cache_root()
            assert "shm" not in str(root)  # macOS should use tempdir

    def test_default_root_windows(self):
        with mock.patch("tools.ram_cache.platform.system", return_value="Windows"):
            root = _default_cache_root()
            assert "shm" not in str(root)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_key_with_special_characters(self):
        cache_set("test", ("/path/with spaces/file.txt", 1, 500), "data")
        assert cache_get("test", ("/path/with spaces/file.txt", 1, 500)) == "data"

    def test_concurrent_writes_safe(self):
        """Multiple rapid writes to same key should not corrupt."""
        import threading
        import tools.ram_cache as rc
        root = rc.CACHE_ROOT  # capture monkeypatched root
        errors = []

        def writer(val):
            try:
                # Use the module directly (threads share process state)
                cache_set("test", ("concurrent",), f"value_{val}")
            except Exception as e:
                errors.append(e)

        # Pre-create directory so threads don't race on mkdir
        (root / "test").mkdir(parents=True, exist_ok=True)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        result = cache_get("test", ("concurrent",))
        assert result is not None and result.startswith("value_")

    def test_make_key_deterministic(self):
        k1 = _make_key(("a", "b", "c"))
        k2 = _make_key(("a", "b", "c"))
        assert k1 == k2

    def test_make_key_order_independent_for_dicts(self):
        """JSON sort_keys ensures dict key order doesn't matter."""
        k1 = _make_key(({"b": 2, "a": 1},))
        k2 = _make_key(({"a": 1, "b": 2},))
        assert k1 == k2

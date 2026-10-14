"""Tests for atomic file write operations in utils.py.

Covers:
  - atomic_json_write: crash-safe JSON persistence
  - atomic_yaml_write: crash-safe YAML persistence
  - atomic_roundtrip_yaml_update: comment-preserving single-key YAML updates
  - atomic_replace: symlink-preserving os.replace wrapper
  - _preserve_file_mode / _restore_file_mode: permission preservation
  - safe_json_loads: defensive JSON parsing
  - normalize_proxy_url / normalize_proxy_env_vars: proxy URL normalization
"""

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from utils import (
    atomic_json_write,
    atomic_replace,
    atomic_yaml_write,
    atomic_roundtrip_yaml_update,
    normalize_proxy_env_vars,
    normalize_proxy_url,
    safe_json_loads,
    _preserve_file_mode,
    _restore_file_mode,
)


# =========================================================================
# atomic_json_write
# =========================================================================


class TestAtomicJsonWrite:
    """Test crash-safe JSON file writing."""

    def test_creates_new_file(self, tmp_path):
        """Writing to a non-existent path creates the file."""
        target = tmp_path / "data.json"
        atomic_json_write(target, {"key": "value"})
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == {"key": "value"}

    def test_overwrites_existing_file(self, tmp_path):
        """Writing to an existing file replaces its content."""
        target = tmp_path / "data.json"
        target.write_text('{"old": true}', encoding="utf-8")
        atomic_json_write(target, {"new": True})
        assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}

    def test_creates_parent_directories(self, tmp_path):
        """Missing parent directories are created automatically."""
        target = tmp_path / "deep" / "nested" / "dir" / "data.json"
        atomic_json_write(target, [1, 2, 3])
        assert json.loads(target.read_text(encoding="utf-8")) == [1, 2, 3]

    def test_unicode_content(self, tmp_path):
        """Non-ASCII content is written correctly (ensure_ascii=False)."""
        target = tmp_path / "unicode.json"
        data = {"greeting": "こんにちは", "emoji": "🚀"}
        atomic_json_write(target, data)
        result = json.loads(target.read_text(encoding="utf-8"))
        assert result == data

    def test_custom_indent(self, tmp_path):
        """Custom indent parameter is respected."""
        target = tmp_path / "indented.json"
        atomic_json_write(target, {"a": 1}, indent=4)
        content = target.read_text(encoding="utf-8")
        assert "    " in content  # 4-space indent

    def test_dump_kwargs_forwarded(self, tmp_path):
        """Extra kwargs are forwarded to json.dump (e.g. default=str)."""
        from datetime import datetime

        target = tmp_path / "custom.json"
        data = {"timestamp": datetime(2025, 1, 1, 12, 0, 0)}
        atomic_json_write(target, data, default=str)
        result = json.loads(target.read_text(encoding="utf-8"))
        assert result["timestamp"] == "2025-01-01 12:00:00"

    def test_no_temp_file_left_on_success(self, tmp_path):
        """Successful write leaves no temp files behind."""
        subdir = tmp_path / "json_out"
        subdir.mkdir()
        target = subdir / "clean.json"
        atomic_json_write(target, {"ok": True})
        files = list(subdir.iterdir())
        assert len(files) == 1
        assert files[0].name == "clean.json"

    def test_no_temp_file_left_on_failure(self, tmp_path):
        """Failed write cleans up the temp file."""
        target = tmp_path / "fail.json"

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": Unserializable()})

        # No temp files should remain
        files = list(tmp_path.iterdir())
        assert all(not f.name.startswith(".") for f in files)

    def test_original_file_intact_on_failure(self, tmp_path):
        """If write fails, the original file content is preserved."""
        target = tmp_path / "preserved.json"
        target.write_text('{"original": true}', encoding="utf-8")

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": Unserializable()})

        assert json.loads(target.read_text(encoding="utf-8")) == {"original": True}

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX file modes not enforced on NTFS",
    )
    def test_preserves_file_permissions(self, tmp_path):
        """Original file permissions are preserved after atomic write."""
        target = tmp_path / "perms.json"
        target.write_text("{}", encoding="utf-8")
        os.chmod(target, 0o644)

        atomic_json_write(target, {"updated": True})
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o644

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_preserves_symlink(self, tmp_path):
        """Writing through a symlink updates the real file, symlink survives."""
        real_file = tmp_path / "real.json"
        real_file.write_text('{"v": 1}', encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(real_file)

        atomic_json_write(link, {"v": 2})

        # Symlink still exists and points to real file
        assert link.is_symlink()
        assert link.resolve() == real_file.resolve()
        # Content updated in real file
        assert json.loads(real_file.read_text(encoding="utf-8")) == {"v": 2}

    def test_accepts_string_path(self, tmp_path):
        """String paths are accepted (not just Path objects)."""
        target = str(tmp_path / "string_path.json")
        atomic_json_write(target, {"str": True})
        assert json.loads(Path(target).read_text(encoding="utf-8")) == {"str": True}


# =========================================================================
# atomic_yaml_write
# =========================================================================


class TestAtomicYamlWrite:
    """Test crash-safe YAML file writing."""

    def test_creates_new_file(self, tmp_path):
        """Writing to a non-existent path creates the file."""
        target = tmp_path / "config.yaml"
        atomic_yaml_write(target, {"model": "gpt-4", "temperature": 0.7})
        assert target.exists()
        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result == {"model": "gpt-4", "temperature": 0.7}

    def test_overwrites_existing_file(self, tmp_path):
        """Writing to an existing file replaces its content."""
        target = tmp_path / "config.yaml"
        target.write_text("old: true\n", encoding="utf-8")
        atomic_yaml_write(target, {"new": True})
        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result == {"new": True}

    def test_creates_parent_directories(self, tmp_path):
        """Missing parent directories are created automatically."""
        target = tmp_path / "a" / "b" / "config.yaml"
        atomic_yaml_write(target, {"nested": "dir"})
        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result == {"nested": "dir"}

    def test_extra_content_appended(self, tmp_path):
        """extra_content parameter appends text after YAML dump."""
        target = tmp_path / "config.yaml"
        extra = "\n# Commented-out section:\n# debug: true\n"
        atomic_yaml_write(target, {"active": True}, extra_content=extra)
        content = target.read_text(encoding="utf-8")
        assert "# Commented-out section:" in content
        assert "# debug: true" in content

    def test_sort_keys_option(self, tmp_path):
        """sort_keys parameter controls key ordering."""
        target = tmp_path / "sorted.yaml"
        data = {"zebra": 1, "alpha": 2, "middle": 3}
        atomic_yaml_write(target, data, sort_keys=True)
        content = target.read_text(encoding="utf-8")
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert lines[0].startswith("alpha")

    def test_no_temp_file_left_on_success(self, tmp_path):
        """Successful write leaves no temp files behind."""
        subdir = tmp_path / "yaml_out"
        subdir.mkdir()
        target = subdir / "clean.yaml"
        atomic_yaml_write(target, {"ok": True})
        files = list(subdir.iterdir())
        assert len(files) == 1

    def test_original_file_intact_on_failure(self, tmp_path):
        """If write fails, the original file content is preserved."""
        target = tmp_path / "preserved.yaml"
        target.write_text("original: true\n", encoding="utf-8")

        # Force a failure by making yaml.dump raise
        with patch("utils.yaml.dump", side_effect=RuntimeError("forced")):
            with pytest.raises(RuntimeError):
                atomic_yaml_write(target, {"bad": True})

        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result == {"original": True}

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_preserves_symlink(self, tmp_path):
        """Writing through a symlink updates the real file, symlink survives."""
        real_file = tmp_path / "real.yaml"
        real_file.write_text("v: 1\n", encoding="utf-8")
        link = tmp_path / "link.yaml"
        link.symlink_to(real_file)

        atomic_yaml_write(link, {"v": 2})

        assert link.is_symlink()
        assert yaml.safe_load(real_file.read_text(encoding="utf-8")) == {"v": 2}


# =========================================================================
# atomic_roundtrip_yaml_update
# =========================================================================


class TestAtomicRoundtripYamlUpdate:
    """Test comment-preserving single-key YAML updates."""

    def test_updates_existing_key(self, tmp_path):
        """Updating an existing key preserves other content."""
        target = tmp_path / "config.yaml"
        target.write_text("model: gpt-4\ntemperature: 0.7\n", encoding="utf-8")

        atomic_roundtrip_yaml_update(target, "temperature", 0.9)

        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.9

    def test_adds_new_key(self, tmp_path):
        """Adding a new key to an existing file."""
        target = tmp_path / "config.yaml"
        target.write_text("model: gpt-4\n", encoding="utf-8")

        atomic_roundtrip_yaml_update(target, "max_tokens", 4096)

        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result["model"] == "gpt-4"
        assert result["max_tokens"] == 4096

    def test_nested_key_path(self, tmp_path):
        """Dotted key paths create nested structures."""
        target = tmp_path / "config.yaml"
        target.write_text("model: gpt-4\n", encoding="utf-8")

        atomic_roundtrip_yaml_update(target, "logging.level", "DEBUG")

        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result["logging"]["level"] == "DEBUG"

    def test_deeply_nested_key_path(self, tmp_path):
        """Multiple levels of nesting work correctly."""
        target = tmp_path / "config.yaml"
        target.write_text("top: value\n", encoding="utf-8")

        atomic_roundtrip_yaml_update(target, "a.b.c.d", "deep")

        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result["a"]["b"]["c"]["d"] == "deep"

    def test_creates_file_if_missing(self, tmp_path):
        """Creates the file if it doesn't exist."""
        target = tmp_path / "new.yaml"
        atomic_roundtrip_yaml_update(target, "key", "value")

        assert target.exists()
        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result["key"] == "value"

    def test_preserves_comments(self, tmp_path):
        """Comments in the YAML file are preserved."""
        target = tmp_path / "config.yaml"
        content = "# Main config\nmodel: gpt-4  # the model\ntemp: 0.7\n"
        target.write_text(content, encoding="utf-8")

        atomic_roundtrip_yaml_update(target, "temp", 0.9)

        updated = target.read_text(encoding="utf-8")
        assert "# Main config" in updated
        assert "# the model" in updated

    def test_preserves_unicode(self, tmp_path):
        """Unicode content is preserved during roundtrip."""
        target = tmp_path / "config.yaml"
        target.write_text("name: 日本語テスト\n", encoding="utf-8")

        atomic_roundtrip_yaml_update(target, "added", "新しい値")

        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result["name"] == "日本語テスト"
        assert result["added"] == "新しい値"

    def test_overwrites_non_dict_intermediate(self, tmp_path):
        """If an intermediate key holds a non-dict, it's replaced with a dict."""
        target = tmp_path / "config.yaml"
        target.write_text("logging: simple_string\n", encoding="utf-8")

        atomic_roundtrip_yaml_update(target, "logging.level", "INFO")

        result = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert result["logging"]["level"] == "INFO"


# =========================================================================
# atomic_replace
# =========================================================================


class TestAtomicReplace:
    """Test symlink-preserving atomic replace."""

    def test_replaces_regular_file(self, tmp_path):
        """Basic replace of a regular file."""
        target = tmp_path / "target.txt"
        target.write_text("old", encoding="utf-8")

        tmp_file = tmp_path / "tmp.txt"
        tmp_file.write_text("new", encoding="utf-8")

        atomic_replace(tmp_file, target)

        assert target.read_text(encoding="utf-8") == "new"
        assert not tmp_file.exists()

    def test_creates_new_file(self, tmp_path):
        """Replace to a non-existent target creates the file."""
        target = tmp_path / "new.txt"
        tmp_file = tmp_path / "tmp.txt"
        tmp_file.write_text("content", encoding="utf-8")

        atomic_replace(tmp_file, target)

        assert target.read_text(encoding="utf-8") == "content"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_preserves_symlink_target(self, tmp_path):
        """When target is a symlink, the real file is updated and symlink survives."""
        real_file = tmp_path / "real.txt"
        real_file.write_text("original", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        tmp_file = tmp_path / "tmp.txt"
        tmp_file.write_text("updated", encoding="utf-8")

        atomic_replace(tmp_file, link)

        assert link.is_symlink()
        assert real_file.read_text(encoding="utf-8") == "updated"

    def test_returns_resolved_path(self, tmp_path):
        """Returns the real path used for the replace."""
        target = tmp_path / "target.txt"
        target.write_text("old", encoding="utf-8")
        tmp_file = tmp_path / "tmp.txt"
        tmp_file.write_text("new", encoding="utf-8")

        result = atomic_replace(tmp_file, target)
        assert result == str(target)


# =========================================================================
# _preserve_file_mode / _restore_file_mode
# =========================================================================


class TestFileMode:
    """Test permission preservation helpers."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX file modes not enforced on NTFS",
    )
    def test_preserve_existing_file(self, tmp_path):
        """Captures mode bits of an existing file."""
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")
        os.chmod(target, 0o755)

        mode = _preserve_file_mode(target)
        assert mode == 0o755

    def test_preserve_nonexistent_file(self, tmp_path):
        """Returns None for non-existent files."""
        target = tmp_path / "missing.txt"
        mode = _preserve_file_mode(target)
        assert mode is None

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX file modes not enforced on NTFS",
    )
    def test_restore_mode(self, tmp_path):
        """Restores mode bits on a file."""
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")
        os.chmod(target, 0o600)

        _restore_file_mode(target, 0o755)
        assert stat.S_IMODE(target.stat().st_mode) == 0o755

    def test_restore_none_is_noop(self, tmp_path):
        """Passing None mode does nothing (no crash)."""
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")
        _restore_file_mode(target, None)  # Should not raise


# =========================================================================
# safe_json_loads
# =========================================================================


class TestSafeJsonLoads:
    """Test defensive JSON parsing."""

    def test_valid_json(self):
        assert safe_json_loads('{"key": "value"}') == {"key": "value"}

    def test_valid_json_array(self):
        assert safe_json_loads("[1, 2, 3]") == [1, 2, 3]

    def test_valid_json_number(self):
        assert safe_json_loads("42") == 42

    def test_invalid_json_returns_default(self):
        assert safe_json_loads("not json") is None

    def test_invalid_json_custom_default(self):
        assert safe_json_loads("not json", default={}) == {}

    def test_empty_string_returns_default(self):
        assert safe_json_loads("", default="fallback") == "fallback"

    def test_none_input_returns_default(self):
        assert safe_json_loads(None) is None

    def test_none_input_custom_default(self):
        assert safe_json_loads(None, default=[]) == []

    def test_type_error_returns_default(self):
        """Non-string input that causes TypeError returns default."""
        assert safe_json_loads(12345, default="err") == "err"


# =========================================================================
# normalize_proxy_url
# =========================================================================


class TestNormalizeProxyUrl:
    """Test proxy URL normalization."""

    def test_socks_rewritten_to_socks5(self):
        """socks:// is rewritten to socks5:// for httpx compatibility."""
        result = normalize_proxy_url("socks://127.0.0.1:7890")
        assert result == "socks5://127.0.0.1:7890"

    def test_socks_case_insensitive(self):
        """SOCKS:// (uppercase) is also rewritten."""
        result = normalize_proxy_url("SOCKS://192.168.1.1:1080")
        assert result == "socks5://192.168.1.1:1080"

    def test_socks5_unchanged(self):
        """socks5:// URLs are returned as-is."""
        result = normalize_proxy_url("socks5://127.0.0.1:7890")
        assert result == "socks5://127.0.0.1:7890"

    def test_http_proxy_unchanged(self):
        """http:// proxy URLs are returned as-is."""
        result = normalize_proxy_url("http://proxy.corp:8080")
        assert result == "http://proxy.corp:8080"

    def test_https_proxy_unchanged(self):
        """https:// proxy URLs are returned as-is."""
        result = normalize_proxy_url("https://proxy.corp:8443")
        assert result == "https://proxy.corp:8443"

    def test_none_returns_none(self):
        assert normalize_proxy_url(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_proxy_url("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_proxy_url("   ") is None


# =========================================================================
# normalize_proxy_env_vars
# =========================================================================


class TestNormalizeProxyEnvVars:
    """Test in-place proxy env var normalization."""

    def test_rewrites_socks_env_var(self, monkeypatch):
        """HTTPS_PROXY=socks://... is rewritten in-place."""
        monkeypatch.setenv("HTTPS_PROXY", "socks://127.0.0.1:7890")
        normalize_proxy_env_vars()
        assert os.environ["HTTPS_PROXY"] == "socks5://127.0.0.1:7890"

    def test_leaves_http_proxy_unchanged(self, monkeypatch):
        """HTTP_PROXY=http://... is not modified."""
        monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")
        normalize_proxy_env_vars()
        assert os.environ["HTTP_PROXY"] == "http://proxy:8080"

    def test_handles_missing_env_vars(self, monkeypatch):
        """Does not crash when proxy env vars are unset."""
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy"):
            monkeypatch.delenv(key, raising=False)
        normalize_proxy_env_vars()  # Should not raise

    def test_rewrites_lowercase_variant(self, monkeypatch):
        """Lowercase all_proxy is also normalized."""
        monkeypatch.setenv("all_proxy", "socks://10.0.0.1:1080")
        normalize_proxy_env_vars()
        assert os.environ["all_proxy"] == "socks5://10.0.0.1:1080"

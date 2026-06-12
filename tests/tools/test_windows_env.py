"""Unit tests for ``tools/environments/windows_env.py``.

Tests cover:
- ``_expand_registry_string`` — behavior with REG_EXPAND_SZ and fallback
- ``_read_registry_value`` — success / missing / non-string
- ``_merge_dedup_paths`` — deduplication and ordering
- ``refresh_env_from_registry`` — integration with mocked winreg

All tests mock ``sys.platform`` and Windows APIs so they run on any host.
"""

from __future__ import annotations

import os
import sys
import winreg
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
# _expand_registry_string
# =========================================================================


class TestExpandRegistryString:
    """``_expand_registry_string`` — Win32 ExpandEnvironmentStringsW wrapper."""

    def test_no_percent_no_change(self):
        """A string without ``%`` is returned unchanged."""
        from tools.environments.windows_env import _expand_registry_string

        # No %% → no expansion attempted
        assert _expand_registry_string(r"C:\Windows") == r"C:\Windows"

    def test_expand_via_ctypes(self):
        """When the ctypes API succeeds, its result is returned."""
        from tools.environments.windows_env import _expand_registry_string

        with patch(
            "tools.environments.windows_env.ctypes.windll.kernel32.ExpandEnvironmentStringsW"
        ) as mock_expand:
            # Two calls: first gets buffer size, second fills buffer
            mock_expand.side_effect = [
                42,  # nchars returned (includes null)
                None,  # write to buffer
            ]
            with patch(
                "tools.environments.windows_env.ctypes.create_unicode_buffer",
                return_value=MagicMock(value="C:\\ProgramData"),
            ) as mock_buf:
                result = _expand_registry_string("%PROGRAMDATA%")
                assert result == "C:\\ProgramData"
                mock_buf.assert_called_once_with(42)

    def test_ctypes_fallback_to_expandvars(self):
        """When ctypes raises, fall back to ``os.path.expandvars``."""
        from tools.environments.windows_env import _expand_registry_string

        with patch(
            "tools.environments.windows_env.ctypes.windll.kernel32.ExpandEnvironmentStringsW",
            side_effect=OSError("access denied"),
        ):
            result = _expand_registry_string("%PATH%")
            # os.path.expandvars will expand using the current process env
            expected = os.path.expandvars("%PATH%")
            assert result == expected


# =========================================================================
# _read_registry_value
# =========================================================================


class TestReadRegistryValue:
    """``_read_registry_value`` — registry read with error handling."""

    def test_value_found(self):
        """A valid string value returns ``(value, type)``."""
        from tools.environments.windows_env import _read_registry_value

        mock_key = MagicMock()
        mock_key.__enter__.return_value = mock_key
        with patch(
            "tools.environments.windows_env.winreg.OpenKey",
            return_value=mock_key,
        ) as mock_open:
            with patch(
                "tools.environments.windows_env.winreg.QueryValueEx",
                return_value=(r"C:\Windows;C:\System32", 1),
            ):
                val, rtype = _read_registry_value(0x80000002, "SomeKey", "Path")
            assert val == r"C:\Windows;C:\System32"
            assert rtype == 1

    def test_value_missing(self):
        """A missing key returns ``(None, None)``."""
        from tools.environments.windows_env import _read_registry_value

        with patch(
            "tools.environments.windows_env.winreg.OpenKey",
            side_effect=FileNotFoundError,
        ):
            val, rtype = _read_registry_value(0x80000002, "Missing", "Path")
            assert val is None
            assert rtype is None

    def test_non_string_value(self):
        """Non-string registry values return ``(None, None)``."""
        from tools.environments.windows_env import _read_registry_value

        mock_key = MagicMock()
        mock_key.__enter__.return_value = mock_key
        with patch(
            "tools.environments.windows_env.winreg.OpenKey",
            return_value=mock_key,
        ):
            with patch(
                "tools.environments.windows_env.winreg.QueryValueEx",
                return_value=(b"binarydata", 3),
            ):
                val, rtype = _read_registry_value(0x80000002, "SomeKey", "Binary")
            assert val is None
            assert rtype is None


# =========================================================================
# _merge_dedup_paths
# =========================================================================


class TestMergeDedupPaths:
    """``_merge_dedup_paths`` — deduplication logic."""

    def test_basic_merge(self):
        """Simple semicolon paths are merged."""
        from tools.environments.windows_env import _merge_dedup_paths

        result = _merge_dedup_paths(
            r"C:\Windows;C:\System32",
            r"C:\Python39",
        )
        assert result == r"C:\Windows;C:\System32;C:\Python39"

    def test_dedup_case_insensitive(self):
        """Duplicates (case-insensitive) are removed, first occurrence wins."""
        from tools.environments.windows_env import _merge_dedup_paths

        result = _merge_dedup_paths(
            r"C:\Windows;C:\System32",
            r"c:\windows;C:\Program Files",
        )
        assert result == r"C:\Windows;C:\System32;C:\Program Files"

    def test_empty_and_whitespace(self):
        """Empty entries and whitespace-only entries are skipped."""
        from tools.environments.windows_env import _merge_dedup_paths

        result = _merge_dedup_paths(
            r"C:\Windows;; ;C:\System32",
        )
        assert result == r"C:\Windows;C:\System32"

    def test_multiple_sources(self):
        """Multiple positional sources are all merged."""
        from tools.environments.windows_env import _merge_dedup_paths

        result = _merge_dedup_paths(
            r"C:\A",
            r"C:\B",
            r"C:\C",
        )
        assert result == r"C:\A;C:\B;C:\C"


# =========================================================================
# refresh_env_from_registry
# =========================================================================


class TestRefreshEnvFromRegistry:
    """``refresh_env_from_registry`` — end-to-end with mocked registry."""

    def test_non_windows_noop(self):
        """On non-Windows platforms the function is a no-op."""
        with patch("sys.platform", "linux"):
            from tools.environments.windows_env import refresh_env_from_registry

            original_path = os.environ.get("PATH", "")
            original_pathext = os.environ.get("PATHEXT", "")

            refresh_env_from_registry()

            assert os.environ.get("PATH", "") == original_path
            assert os.environ.get("PATHEXT", "") == original_pathext

    def test_updates_path_and_pathext(self):
        """On Windows, PATH and PATHEXT are refreshed from mock registry."""
        from tools.environments.windows_env import (
            _read_registry_value,
            refresh_env_from_registry,
        )

        # We'll mock _read_registry_value to return known values
        def mock_read_registry(hive, subkey, name):
            if name == "Path":
                if "HKEY_LOCAL_MACHINE" in str(hive) or "SYSTEM" in subkey:
                    return (r"C:\Windows;C:\System32", 1)  # REG_SZ
                else:
                    return (r"C:\Users\test\bin", 1)  # REG_SZ
            if name == "PATHEXT":
                if "HKEY_LOCAL_MACHINE" in str(hive) or "SYSTEM" in subkey:
                    return (".COM;.EXE;.BAT;.CMD", 1)  # REG_SZ
                else:
                    return None, None
            return None, None

        with patch(
            "tools.environments.windows_env._read_registry_value",
            side_effect=mock_read_registry,
        ), patch("sys.platform", "win32"):
            refresh_env_from_registry()

            path = os.environ.get("PATH", "")
            assert r"C:\Windows" in path
            assert r"C:\System32" in path
            assert r"C:\Users\test\bin" in path

            pathext = os.environ.get("PATHEXT", "")
            assert ".COM" in pathext
            assert ".EXE" in pathext

    def test_reg_expand_sz_is_expanded(self):
        """REG_EXPAND_SZ values are expanded before merging."""
        from tools.environments.windows_env import refresh_env_from_registry

        def mock_read_registry(hive, subkey, name):
            if name == "Path":
                if "SYSTEM" in subkey:
                    # REG_EXPAND_SZ = 2
                    return ("%SystemRoot%;%SystemRoot%\\System32", 2)
                else:
                    return None, None
            if name == "PATHEXT":
                return None, None
            return None, None

        with patch(
            "tools.environments.windows_env._read_registry_value",
            side_effect=mock_read_registry,
        ), patch(
            "tools.environments.windows_env._expand_registry_string",
            return_value=r"C:\Windows;C:\Windows\System32",
        ), patch(
            "sys.platform", "win32"
        ):
            refresh_env_from_registry()

            path = os.environ.get("PATH", "")
            assert r"C:\Windows" in path
            assert r"C:\Windows\System32" in path

    def test_merge_dedup_applied(self):
        """Duplicate entries from HKLM + HKCU are deduplicated."""
        from tools.environments.windows_env import refresh_env_from_registry

        def mock_read_registry(hive, subkey, name):
            if name == "Path":
                return (r"C:\Windows;C:\Windows\System32", 1)  # same in both
            if name == "PATHEXT":
                return None, None
            return None, None

        with patch(
            "tools.environments.windows_env._read_registry_value",
            side_effect=mock_read_registry,
        ), patch("sys.platform", "win32"):
            refresh_env_from_registry()

            path = os.environ.get("PATH", "")
            parts = [p.strip() for p in path.split(";") if p.strip()]
            # Each directory should appear at most once
            assert len(parts) == len(set(p.lower() for p in parts))

"""Windows ARM64 agent-browser resolution (issue #77051).

agent-browser ships only ``agent-browser-win32-x64.exe``. On Windows ARM64 the
npm JS wrapper still looks for ``win32-arm64``, and a 0-byte stub left by older
postinstalls spawns as ``EFTYPE``. Hermes must prefer the x64 PE (or fail with
an actionable error) instead of routing through that wrapper.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_constants as hc
import tools.browser_tool as bt


@pytest.fixture(autouse=True)
def _clear_browser_caches():
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    yield
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False


def _write_fake_exe(path: Path, *, size: int = 4096) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ" + b"\0" * (size - 2))
    return path


def _shim_with_package(tmp_path: Path, *, empty_arm64: bool = True) -> tuple[Path, Path]:
    pkg_bin = tmp_path / "node_modules" / "agent-browser" / "bin"
    native = _write_fake_exe(pkg_bin / "agent-browser-win32-x64.exe", size=8192)
    if empty_arm64:
        (pkg_bin / "agent-browser-win32-arm64.exe").write_bytes(b"")
    shim = tmp_path / "node_modules" / ".bin" / "agent-browser.cmd"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text("@echo off\n")
    return shim, native


class TestNativeBinaryNameSelection:
    def test_windows_arm64_prefers_x64_then_arm64_name(self):
        names = hc.agent_browser_native_binary_names(windows_arm64=True)
        assert names[0] == "agent-browser-win32-x64.exe"
        assert "agent-browser-win32-arm64.exe" in names

    def test_windows_x64_only_lists_x64(self, monkeypatch):
        monkeypatch.setattr(hc.sys, "platform", "win32")
        names = hc.agent_browser_native_binary_names(windows_arm64=False)
        assert names == ("agent-browser-win32-x64.exe",)


class TestResolveAndHeal:
    def test_resolve_prefers_usable_x64_over_empty_arm64_stub(self, tmp_path):
        shim, native = _shim_with_package(tmp_path)
        resolved = hc.resolve_agent_browser_native_binary(
            str(shim), windows_arm64=True
        )
        assert resolved == str(native)

    def test_resolve_from_prefix_shim(self, tmp_path):
        pkg_bin = tmp_path / "node_modules" / "agent-browser" / "bin"
        native = _write_fake_exe(pkg_bin / "agent-browser-win32-x64.exe", size=8192)
        shim = tmp_path / "agent-browser.cmd"
        shim.write_text("@echo off\n")
        assert (
            hc.resolve_agent_browser_native_binary(str(shim), windows_arm64=True)
            == str(native)
        )

    def test_heal_copies_x64_over_empty_arm64_stub(self, tmp_path):
        pkg_bin = tmp_path / "bin"
        x64 = _write_fake_exe(pkg_bin / "agent-browser-win32-x64.exe", size=4096)
        arm64 = pkg_bin / "agent-browser-win32-arm64.exe"
        arm64.write_bytes(b"")

        healed = hc.heal_agent_browser_windows_arm64_stub(
            pkg_bin, windows_arm64=True
        )
        assert healed == str(x64)
        assert arm64.stat().st_size == x64.stat().st_size

    def test_heal_noop_when_not_windows_arm64(self, tmp_path):
        pkg_bin = tmp_path / "bin"
        _write_fake_exe(pkg_bin / "agent-browser-win32-x64.exe")
        assert (
            hc.heal_agent_browser_windows_arm64_stub(pkg_bin, windows_arm64=False)
            is None
        )

    def test_reject_empty_native_binary(self, tmp_path):
        stub = tmp_path / "agent-browser-win32-arm64.exe"
        stub.write_bytes(b"")
        assert hc._agent_browser_native_binary_usable(stub) is False

    def test_prefer_replaces_shim_with_native(self, tmp_path, monkeypatch):
        shim, native = _shim_with_package(tmp_path)
        # Filename selection keys off sys.platform; is_windows_arm64 alone is
        # not enough on Linux CI (would pick linux-* names and miss the .exe).
        monkeypatch.setattr(hc, "is_windows_arm64", lambda: True)
        monkeypatch.setattr(hc.sys, "platform", "win32")
        assert hc.prefer_agent_browser_native_binary(str(shim)) == str(native)

    def test_prefer_passes_through_npx_sentinel(self):
        assert (
            hc.prefer_agent_browser_native_binary("npx agent-browser")
            == "npx agent-browser"
        )


class TestFindAgentBrowserWinArm64:
    def test_coerce_heals_stub_and_returns_x64(self, tmp_path, monkeypatch):
        shim, native = _shim_with_package(tmp_path)
        arm64 = tmp_path / "node_modules" / "agent-browser" / "bin" / (
            "agent-browser-win32-arm64.exe"
        )
        monkeypatch.setattr(hc, "is_windows_arm64", lambda: True)
        monkeypatch.setattr(bt.sys, "platform", "win32")

        coerced = bt._coerce_agent_browser_cmd(str(shim))
        assert coerced == str(native)
        assert arm64.stat().st_size >= 1024

    def test_find_returns_native_x64_from_path_shim(self, tmp_path, monkeypatch):
        shim, native = _shim_with_package(tmp_path)
        monkeypatch.setattr(hc, "is_windows_arm64", lambda: True)
        monkeypatch.setattr(bt.sys, "platform", "win32")
        monkeypatch.setattr(bt, "get_hermes_home", lambda: tmp_path / "empty-home")
        (tmp_path / "empty-home").mkdir()

        with patch(
            "tools.browser_tool.shutil.which",
            side_effect=lambda cmd, path=None: str(shim) if cmd == "agent-browser" else None,
        ), patch(
            "tools.browser_tool.agent_browser_runnable", return_value=True
        ), patch(
            "tools.browser_tool._merge_browser_path", return_value=""
        ):
            assert bt._find_agent_browser() == str(native)

    def test_find_raises_actionable_error_when_x64_missing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(hc, "is_windows_arm64", lambda: True)
        monkeypatch.setattr(bt.sys, "platform", "win32")
        monkeypatch.setattr(bt, "get_hermes_home", lambda: tmp_path)

        with patch("tools.browser_tool.shutil.which", return_value=None), patch(
            "tools.browser_tool._merge_browser_path", return_value=""
        ), patch(
            "tools.browser_tool.resolve_agent_browser_native_binary",
            return_value=None,
        ), patch(
            "hermes_cli.dep_ensure.ensure_dependency",
            return_value=False,
        ):
            with pytest.raises(FileNotFoundError, match="Windows ARM64"):
                bt._find_agent_browser()

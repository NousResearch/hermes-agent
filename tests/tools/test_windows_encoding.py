"""Tests for Windows UTF-8 encoding helpers."""
import subprocess
import sys
import pytest

# Skip entire module on non-Windows
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")


class TestPsWithUtf8:
    """Verify ps_with_utf8() prepends encoding directives correctly."""

    def test_prepends_preamble_on_windows(self):
        from tools.environments.windows_env import ps_with_utf8
        result = ps_with_utf8("Get-ChildItem")
        assert "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8" in result
        assert "$OutputEncoding=[System.Text.Encoding]::UTF8" in result
        assert result.endswith("Get-ChildItem")

    def test_idempotent_no_double_prepend(self):
        from tools.environments.windows_env import ps_with_utf8
        once = ps_with_utf8("dir")
        twice = ps_with_utf8(once)
        # Should NOT double-prepend — the helper is idempotent
        assert twice.count("[Console]::OutputEncoding") == 1

    def test_passthrough_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        # Re-import to pick up patched platform
        import importlib
        import tools.environments.windows_env as we
        importlib.reload(we)
        result = we.ps_with_utf8("ls -la")
        assert "[Console]::OutputEncoding" not in result
        assert result == "ls -la"


class TestPowerShellUtf8Output:
    """End-to-end: PowerShell emits UTF-8 when preamble is present."""

    UTF8_CHARS = "caf\u00e9 \u4f60\u597d r\u00e9sum\u00e9"  # caf\u00e9 \u4f60\u597d r\u00e9sum\u00e9

    def test_without_preamble_loses_unicode(self):
        """Without preamble, non-ASCII may be garbled/missing (cp1252 path)."""
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Write-Output '{self.UTF8_CHARS}'"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        # On a cp1252 system, stdout will likely NOT contain the full UTF-8 string.
        # This test documents the baseline problem — it may pass or fail depending
        # on system config, so we assert loosely.
        assert r.returncode == 0
        # At minimum, some output was produced
        assert len(r.stdout.strip()) > 0

    def test_with_preamble_preserves_unicode(self):
        """With preamble, PowerShell emits correct UTF-8 and Python decodes it."""
        from tools.environments.windows_env import ps_with_utf8
        cmd = ps_with_utf8(f"Write-Output '{self.UTF8_CHARS}'")
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        assert r.returncode == 0
        assert self.UTF8_CHARS in r.stdout

    def test_with_preamble_cjk_characters(self):
        """Chinese/Japanese/Korean characters survive the round-trip."""
        from tools.environments.windows_env import ps_with_utf8
        cjk = "\u4f60\u597d\u4e16\u754c \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4"
        cmd = ps_with_utf8(f"Write-Output '{cjk}'")
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        assert r.returncode == 0
        assert cjk in r.stdout

    def test_with_preamble_emoji(self):
        """Emoji survives the round-trip."""
        from tools.environments.windows_env import ps_with_utf8
        emoji = "\U0001f600 \U0001f4a5 \u2728"  # \ud83d\ude00 \ud83d\udca5 \u2728
        cmd = ps_with_utf8(f"Write-Output '{emoji}'")
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        assert r.returncode == 0
        assert emoji in r.stdout

    def test_stderr_also_utf8_with_preamble(self):
        """PowerShell error output is also UTF-8 encoded."""
        from tools.environments.windows_env import ps_with_utf8
        cmd = ps_with_utf8("Write-Error '\u9519\u8bef\u6d88\u606f'")
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        # Write-Error goes to stderr
        assert "\u9519\u8bef\u6d88\u606f" in (r.stderr or "")


class TestSubprocessEncodingHardening:
    """Verify subprocess.run calls are hardened with encoding='utf-8'."""

    def test_text_true_without_encoding_can_fail(self):
        """text=True without encoding= uses system code page — may crash."""
        # This simulates the pre-fix pattern from clipboard.py, claw.py, etc.
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
             "$OutputEncoding=[System.Text.Encoding]::UTF8;"
             "Write-Output '\u00e9'"],
            capture_output=True, text=True,  # NO encoding= — uses system default
            timeout=10,
        )
        # May fail or produce garbled output depending on system code page
        assert r.returncode == 0

    def test_text_true_with_encoding_works(self):
        """text=True WITH encoding='utf-8' always works."""
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
             "$OutputEncoding=[System.Text.Encoding]::UTF8;"
             "Write-Output '\u00e9'"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        assert r.returncode == 0
        assert "\u00e9" in r.stdout


class TestBootstrapConsoleCodePage:
    """Verify hermes_bootstrap sets console code page to UTF-8."""

    def test_console_cp_is_utf8_after_bootstrap(self):
        import ctypes
        import hermes_bootstrap
        hermes_bootstrap.apply_windows_utf8_bootstrap()
        kernel32 = ctypes.windll.kernel32
        cp_out = kernel32.GetConsoleOutputCP()
        cp_in = kernel32.GetConsoleCP()
        assert cp_out == 65001, f"ConsoleOutputCP={cp_out}, expected 65001"
        assert cp_in == 65001, f"ConsoleCP={cp_in}, expected 65001"

    def test_disable_env_var_respected(self, monkeypatch):
        monkeypatch.setenv("HERMES_DISABLE_WINDOWS_UTF8", "1")
        import ctypes
        import importlib
        import hermes_bootstrap
        # Disable env var causes bootstrap to no-op
        result = hermes_bootstrap.apply_windows_utf8_bootstrap()
        assert result is False

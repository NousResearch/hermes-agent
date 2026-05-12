"""Unit tests for the Windows setup-wizard UI fixes (#24345).

Pins three behaviours that together resolve the "Setup wizard looks
broken on Windows" symptoms users reported in the issue:

  1. ``hermes_cli.stdio._enable_virtual_terminal_processing`` actually
     issues the SetConsoleMode call (and never blows up if ctypes /
     kernel32 are unhappy).
  2. ``hermes_cli.stdio.is_legacy_windows_console`` is the gate the
     rest of the CLI uses to decide "render ANSI or not", so its
     tri-state contract (None / True / False) must hold across the
     happy path, the legacy-console path, and POSIX.
  3. ``hermes_cli.colors.should_use_color`` and the ``curses_ui``
     fallback helpers honour that gate -- otherwise the fix would be
     a no-op for the exact users who reported the bug.

Every test mocks ``sys.platform`` and the ctypes layer so the suite
runs identically on Linux CI and a real Windows host.
"""
from __future__ import annotations

import sys
from typing import Iterator
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_stdio_module(monkeypatch) -> Iterator[object]:
    """Reload ``hermes_cli.stdio`` so the per-process configured flag resets.

    Same pattern as the existing ``test_windows_native_support`` fixture --
    each test starts with a clean slate so the idempotency latch doesn't
    swallow follow-up ``configure_windows_stdio()`` calls.
    """
    sys.modules.pop("hermes_cli.stdio", None)
    import hermes_cli.stdio as _s
    _s._CONFIGURED = False
    _s._VT_PROCESSING_ENABLED = None
    yield _s
    sys.modules.pop("hermes_cli.stdio", None)


# ---------------------------------------------------------------------------
# _enable_virtual_terminal_processing
# ---------------------------------------------------------------------------


class TestEnableVirtualTerminalProcessing:
    """The SetConsoleMode wrapper that fixes the literal-escape bug."""

    def test_success_path_calls_setconsolemode_on_both_handles(
        self, fresh_stdio_module, monkeypatch,
    ):
        """Happy path: kernel32 returns success, both stdout+stderr get the flag."""
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)

        # Stub the entire ctypes.windll.kernel32 surface.
        fake_kernel32 = MagicMock()
        fake_kernel32.GetStdHandle.side_effect = lambda _id: 100 + _id  # nonzero handle
        fake_kernel32.GetConsoleMode.return_value = 1  # success → fills mode
        fake_kernel32.SetConsoleMode.return_value = 1
        fake_ctypes = MagicMock()
        fake_ctypes.windll.kernel32 = fake_kernel32
        # byref / POINTER / c_void_p need to exist with sane defaults.
        fake_ctypes.byref.side_effect = lambda x: x
        fake_ctypes.c_void_p.side_effect = lambda v: MagicMock(value=v)
        monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
        # ctypes.wintypes is consulted for DWORD / HANDLE / BOOL.
        fake_wintypes = MagicMock()
        fake_wintypes.DWORD.side_effect = lambda *a: MagicMock(value=0)
        monkeypatch.setitem(sys.modules, "ctypes.wintypes", fake_wintypes)

        result = stdio._enable_virtual_terminal_processing()

        assert result is True
        # GetStdHandle hit for both stdout (-11) and stderr (-12).
        called_ids = [c.args[0] for c in fake_kernel32.GetStdHandle.call_args_list]
        assert -11 in called_ids and -12 in called_ids
        # SetConsoleMode called twice (stdout + stderr).
        assert fake_kernel32.SetConsoleMode.call_count == 2

    def test_legacy_host_returns_false_when_setconsolemode_fails(
        self, fresh_stdio_module, monkeypatch,
    ):
        """Legacy Console Host: GetConsoleMode succeeds but SetConsoleMode rejects
        the VT flag (e.g. pre-Win10 1809).  We must report False so callers
        suppress ANSI emission."""
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)

        fake_kernel32 = MagicMock()
        fake_kernel32.GetStdHandle.return_value = 42
        fake_kernel32.GetConsoleMode.return_value = 1
        fake_kernel32.SetConsoleMode.return_value = 0  # rejected on both handles
        fake_ctypes = MagicMock()
        fake_ctypes.windll.kernel32 = fake_kernel32
        fake_ctypes.byref.side_effect = lambda x: x
        fake_ctypes.c_void_p.side_effect = lambda v: MagicMock(value=v)
        monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
        fake_wintypes = MagicMock()
        fake_wintypes.DWORD.side_effect = lambda *a: MagicMock(value=0)
        monkeypatch.setitem(sys.modules, "ctypes.wintypes", fake_wintypes)

        result = stdio._enable_virtual_terminal_processing()
        assert result is False

    def test_returns_false_when_ctypes_blows_up(
        self, fresh_stdio_module, monkeypatch,
    ):
        """ctypes import fails (e.g. weirdly stripped CI runner).  Must not
        propagate -- the rest of stdio configuration still needs to run."""
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)

        # Make ctypes.windll.kernel32 access blow up.
        fake_ctypes = MagicMock()
        fake_ctypes.windll.kernel32 = property(
            lambda self: (_ for _ in ()).throw(OSError("boom"))
        )
        monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

        # No raise; just False.
        assert stdio._enable_virtual_terminal_processing() is False

    def test_treats_redirected_handle_as_neutral(
        self, fresh_stdio_module, monkeypatch,
    ):
        """When stdout/stderr is redirected to a pipe/file, GetConsoleMode
        returns 0 (no console).  ANSI escapes pass through pipes fine, so
        a redirected handle must NOT poison the legacy-console verdict."""
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)

        fake_kernel32 = MagicMock()
        fake_kernel32.GetStdHandle.return_value = 42
        fake_kernel32.GetConsoleMode.return_value = 0  # not a console
        fake_kernel32.SetConsoleMode.return_value = 1
        fake_ctypes = MagicMock()
        fake_ctypes.windll.kernel32 = fake_kernel32
        fake_ctypes.byref.side_effect = lambda x: x
        fake_ctypes.c_void_p.side_effect = lambda v: MagicMock(value=v)
        monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
        fake_wintypes = MagicMock()
        fake_wintypes.DWORD.side_effect = lambda *a: MagicMock(value=0)
        monkeypatch.setitem(sys.modules, "ctypes.wintypes", fake_wintypes)

        # Both handles "no console" → any_success path treats this as fine,
        # returning True (== "ANSI is safe, no need to suppress").
        assert stdio._enable_virtual_terminal_processing() is True


# ---------------------------------------------------------------------------
# is_legacy_windows_console
# ---------------------------------------------------------------------------


class TestIsLegacyWindowsConsole:
    """Tri-state gate used by colors.py and curses_ui.py."""

    def test_false_on_posix_regardless_of_state(self, fresh_stdio_module, monkeypatch):
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: False)
        stdio._VT_PROCESSING_ENABLED = False  # would say "legacy" on Windows
        assert stdio.is_legacy_windows_console() is False

    def test_false_when_vt_enabled_on_windows(self, fresh_stdio_module, monkeypatch):
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        stdio._VT_PROCESSING_ENABLED = True
        assert stdio.is_legacy_windows_console() is False

    def test_true_only_when_vt_explicitly_failed_on_windows(
        self, fresh_stdio_module, monkeypatch,
    ):
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        stdio._VT_PROCESSING_ENABLED = False
        assert stdio.is_legacy_windows_console() is True

    def test_false_when_unconfigured_on_windows(
        self, fresh_stdio_module, monkeypatch,
    ):
        """Pre-configure state must not poison the gate -- assume modern
        until proven otherwise so we don't disable colours on POSIX-style
        early imports."""
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        stdio._VT_PROCESSING_ENABLED = None
        assert stdio.is_legacy_windows_console() is False


# ---------------------------------------------------------------------------
# configure_windows_stdio wiring
# ---------------------------------------------------------------------------


class TestConfigureWindowsStdioVtIntegration:
    """``configure_windows_stdio`` must invoke the VT enable hook and
    publish the result via ``_VT_PROCESSING_ENABLED`` so callers can
    read it through :func:`is_legacy_windows_console`."""

    def test_vt_enable_called_after_code_page_flip(
        self, fresh_stdio_module, monkeypatch,
    ):
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.delenv("HERMES_DISABLE_WINDOWS_UTF8", raising=False)
        # Stub the inner calls so we can assert ordering.
        order: list[str] = []
        monkeypatch.setattr(
            stdio,
            "_flip_console_code_page_to_utf8",
            lambda: order.append("flip"),
        )
        monkeypatch.setattr(
            stdio,
            "_enable_virtual_terminal_processing",
            lambda: (order.append("vt"), True)[1],
        )
        monkeypatch.setattr(stdio, "_reconfigure_stream", lambda *a, **kw: order.append("reconfigure"))
        monkeypatch.setattr(stdio, "_default_windows_editor", lambda: "notepad")

        stdio.configure_windows_stdio()

        # Code-page flip must precede VT enable (matches the docstring
        # contract -- VT processing needs a sane code page to make
        # sense to the user).
        assert order.index("flip") < order.index("vt")
        # VT enable must precede stream reconfiguration so the new mode
        # is in effect by the time we re-wrap stdout/stderr.
        assert order.index("vt") < order.index("reconfigure")

    def test_vt_failure_marks_legacy_console(
        self, fresh_stdio_module, monkeypatch,
    ):
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.delenv("HERMES_DISABLE_WINDOWS_UTF8", raising=False)
        monkeypatch.setattr(stdio, "_flip_console_code_page_to_utf8", lambda: None)
        monkeypatch.setattr(
            stdio, "_enable_virtual_terminal_processing", lambda: False
        )
        monkeypatch.setattr(stdio, "_reconfigure_stream", lambda *a, **kw: None)
        monkeypatch.setattr(stdio, "_default_windows_editor", lambda: "notepad")

        stdio.configure_windows_stdio()
        assert stdio.is_legacy_windows_console() is True

    def test_vt_success_does_not_mark_legacy_console(
        self, fresh_stdio_module, monkeypatch,
    ):
        stdio = fresh_stdio_module
        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.delenv("HERMES_DISABLE_WINDOWS_UTF8", raising=False)
        monkeypatch.setattr(stdio, "_flip_console_code_page_to_utf8", lambda: None)
        monkeypatch.setattr(
            stdio, "_enable_virtual_terminal_processing", lambda: True
        )
        monkeypatch.setattr(stdio, "_reconfigure_stream", lambda *a, **kw: None)
        monkeypatch.setattr(stdio, "_default_windows_editor", lambda: "notepad")

        stdio.configure_windows_stdio()
        assert stdio.is_legacy_windows_console() is False


# ---------------------------------------------------------------------------
# colors.should_use_color gating
# ---------------------------------------------------------------------------


class TestShouldUseColorGating:
    """``should_use_color`` must return False on a legacy Windows console
    EVEN when stdout.isatty() is True -- otherwise the wizard banner
    keeps printing literal ←[35m escapes (#24345)."""

    @pytest.fixture(autouse=True)
    def _reset_color_module(self, monkeypatch):
        # Force a fresh import so the inline ``from hermes_cli.stdio
        # import is_legacy_windows_console`` inside should_use_color
        # picks up the module under test.
        sys.modules.pop("hermes_cli.colors", None)
        sys.modules.pop("hermes_cli.stdio", None)
        yield
        sys.modules.pop("hermes_cli.colors", None)
        sys.modules.pop("hermes_cli.stdio", None)

    def test_color_disabled_on_legacy_windows_console(self, monkeypatch):
        # Reload stdio with the legacy flag tripped.
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: True)

        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)

        assert colors.should_use_color() is False

    def test_color_enabled_on_modern_windows_console(self, monkeypatch):
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: False)

        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)

        assert colors.should_use_color() is True

    def test_color_disabled_when_no_color_set(self, monkeypatch):
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: False)

        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("TERM", raising=False)

        assert colors.should_use_color() is False

    def test_color_disabled_when_term_dumb(self, monkeypatch):
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: False)

        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")

        assert colors.should_use_color() is False

    def test_color_disabled_when_stdout_redirected(self, monkeypatch):
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: False)

        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)

        assert colors.should_use_color() is False

    def test_color_unaffected_when_legacy_detector_blows_up(self, monkeypatch):
        """A broken detector must NOT disable colours on POSIX -- we'd
        regress every Linux/macOS user with one stack trace."""
        # Simulate hermes_cli.stdio failing to import / not exposing the symbol.
        sys.modules.pop("hermes_cli.stdio", None)

        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)

        # Use monkeypatch.setattr on sys.modules to inject a broken stdio.
        broken = MagicMock()
        broken.is_legacy_windows_console.side_effect = RuntimeError("nope")
        monkeypatch.setitem(sys.modules, "hermes_cli.stdio", broken)

        assert colors.should_use_color() is True


# ---------------------------------------------------------------------------
# curses_ui ASCII glyph + windows-curses hint
# ---------------------------------------------------------------------------


class TestCursesUiAsciiFallback:
    """``_use_ascii_safe_glyphs`` must mirror the legacy-console gate,
    and ``_glyph`` must pick the right variant per platform."""

    @pytest.fixture(autouse=True)
    def _reset_modules(self):
        sys.modules.pop("hermes_cli.stdio", None)
        sys.modules.pop("hermes_cli.curses_ui", None)
        yield
        sys.modules.pop("hermes_cli.stdio", None)
        sys.modules.pop("hermes_cli.curses_ui", None)

    def test_ascii_glyphs_on_legacy_windows_console(self, monkeypatch):
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr(cu, "_is_windows", lambda: True)
        # Force the stdio gate to report legacy.
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: True)
        monkeypatch.delenv("HERMES_ASCII_GLYPHS", raising=False)

        assert cu._use_ascii_safe_glyphs() is True
        assert cu._glyph("\u2713", "x") == "x"
        assert cu._glyph("\u2192", ">") == ">"

    def test_unicode_glyphs_on_modern_windows_console(self, monkeypatch):
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr(cu, "_is_windows", lambda: True)
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: False)
        monkeypatch.delenv("HERMES_ASCII_GLYPHS", raising=False)

        assert cu._use_ascii_safe_glyphs() is False
        assert cu._glyph("\u2713", "x") == "\u2713"

    def test_unicode_glyphs_on_posix(self, monkeypatch):
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr(cu, "_is_windows", lambda: False)
        monkeypatch.delenv("HERMES_ASCII_GLYPHS", raising=False)

        assert cu._use_ascii_safe_glyphs() is False
        assert cu._glyph("\u2713", "x") == "\u2713"

    @pytest.mark.parametrize("val", ["1", "true", "True", "yes"])
    def test_env_override_forces_ascii_glyphs(self, monkeypatch, val):
        """HERMES_ASCII_GLYPHS=1 must work on any platform -- screen
        readers, OCR pipelines, and exotic Linux fonts all benefit."""
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr(cu, "_is_windows", lambda: False)
        monkeypatch.setenv("HERMES_ASCII_GLYPHS", val)

        assert cu._use_ascii_safe_glyphs() is True
        assert cu._glyph("\u2713", "x") == "x"

    def test_windows_curses_hint_appears_once(self, monkeypatch, capsys):
        """The 'install windows-curses' tip is a one-time nudge so it
        doesn't carpet-bomb a wizard that hits the fallback for every
        section."""
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr(cu, "_is_windows", lambda: True)
        cu._WINDOWS_CURSES_HINT_SHOWN = False

        cu._maybe_show_windows_curses_hint()
        cu._maybe_show_windows_curses_hint()
        cu._maybe_show_windows_curses_hint()
        out = capsys.readouterr().out
        # Exactly one occurrence of the install command.
        assert out.count("pip install windows-curses") == 1

    def test_windows_curses_hint_noop_on_posix(self, monkeypatch, capsys):
        """Linux/macOS users don't need (and don't want) the hint."""
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr(cu, "_is_windows", lambda: False)
        cu._WINDOWS_CURSES_HINT_SHOWN = False

        cu._maybe_show_windows_curses_hint()
        out = capsys.readouterr().out
        assert "windows-curses" not in out


# ---------------------------------------------------------------------------
# Regression anchor — fails if the VT enable wiring is removed.
# ---------------------------------------------------------------------------


class TestBug24345Repro:
    """Bug-shape anchor for #24345.

    Pre-fix behaviour: ``configure_windows_stdio`` never called
    ``SetConsoleMode``, so VT processing stayed off on the legacy
    Windows Console Host and every ANSI escape printed as literal
    text.  ``is_legacy_windows_console`` didn't even exist.

    This anchor pins the post-fix contract: on a Windows host where
    ``_enable_virtual_terminal_processing`` reports failure,
    ``is_legacy_windows_console`` MUST return True and
    ``should_use_color`` MUST suppress ANSI emission.  If a future
    refactor decouples the gate from the enable hook, this anchor
    fires immediately.
    """

    @pytest.fixture(autouse=True)
    def _reset(self):
        sys.modules.pop("hermes_cli.stdio", None)
        sys.modules.pop("hermes_cli.colors", None)
        yield
        sys.modules.pop("hermes_cli.stdio", None)
        sys.modules.pop("hermes_cli.colors", None)

    def test_legacy_console_kills_color_emission_end_to_end(self, monkeypatch):
        import hermes_cli.stdio as stdio
        monkeypatch.setattr(stdio, "is_windows", lambda: True)
        monkeypatch.delenv("HERMES_DISABLE_WINDOWS_UTF8", raising=False)
        monkeypatch.setattr(stdio, "_flip_console_code_page_to_utf8", lambda: None)
        # Force VT processing to fail (legacy host).
        monkeypatch.setattr(
            stdio, "_enable_virtual_terminal_processing", lambda: False
        )
        monkeypatch.setattr(stdio, "_reconfigure_stream", lambda *a, **kw: None)
        monkeypatch.setattr(stdio, "_default_windows_editor", lambda: "notepad")

        stdio.configure_windows_stdio()

        # Contract 1: the gate flips.
        assert stdio.is_legacy_windows_console() is True, (
            "#24345 regression: VT-enable failed but legacy-console gate "
            "did NOT report True -- callers won't know to suppress ANSI."
        )

        # Contract 2: colors stop being emitted.
        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        coloured = colors.color("hello", colors.Colors.MAGENTA)
        assert coloured == "hello", (
            "#24345 regression: should_use_color() returned True on a "
            "legacy Windows console -- the setup wizard banner will "
            f"print literal ANSI escapes again.  Got: {coloured!r}"
        )

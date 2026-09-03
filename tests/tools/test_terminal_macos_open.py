"""Regression tests for the macOS `open` frontmost raise-ladder.

Issue #95261: on macOS, `open <file>` / `open -a <App> <file>` returns exit 0
and genuinely opens the document, but the window lands BEHIND the Hermes
desktop window (Hermes is typically maximised), so the user sees nothing happen
while the agent reports success.

The fix appends a verified raise-ladder after the `open` that brings the opened
app to the front. These tests prove the transform actually happens on Darwin
(monkeypatching ``platform.system()``) and is a pure no-op on non-Darwin CI.

Regression proof: if ``_transform_macos_open_command`` is reverted (i.e. stops
transforming on Darwin), ``test_bare_open_file_is_transformed_on_darwin`` and
``test_open_dash_a_file_is_transformed_on_darwin`` FAIL.
"""

import platform

import tools.terminal_tool as terminal_tool

# Markers that must appear in a transformed command.
LADDER_MARKERS = ("osascript", "activate", "lsappinfo", "frontmost")


def _darwin(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")


def _assert_transformed(command):
    transformed = terminal_tool._transform_macos_open_command(command)
    assert transformed is not None
    assert transformed != command
    assert transformed.startswith(command + "; ")
    for marker in LADDER_MARKERS:
        assert marker in transformed
    return transformed


def _assert_unchanged(command):
    assert terminal_tool._transform_macos_open_command(command) == command


# --- Darwin: file-opening `open` invocations ARE transformed ---------------

def test_bare_open_file_is_transformed_on_darwin(monkeypatch):
    _darwin(monkeypatch)
    _assert_transformed("open /path/to/file.pdf")


def test_open_dash_a_file_is_transformed_on_darwin(monkeypatch):
    _darwin(monkeypatch)
    _assert_transformed("open -a Preview /path/to/file.pdf")


def test_open_dash_a_app_with_spaces_is_transformed_on_darwin(monkeypatch):
    _darwin(monkeypatch)
    transformed = _assert_transformed("open -a 'Google Chrome' /path/to/file.pdf")
    # App name with spaces must be preserved in the ladder.
    assert "Google Chrome" in transformed


# --- Non-Darwin: pure no-op ------------------------------------------------

def test_bare_open_file_unchanged_on_linux(monkeypatch):
    # Default CI platform is Linux; do NOT patch to Darwin.
    _assert_unchanged("open /path/to/file.pdf")


def test_open_dash_a_file_unchanged_on_linux(monkeypatch):
    _assert_unchanged("open -a Preview /path/to/file.pdf")


# --- Darwin: non-file-opening `open` invocations are NOT transformed -------

def test_open_with_no_args_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open")


def test_open_dash_R_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -R /path/to/file.pdf")


def test_open_dash_e_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -e /path/to/file.pdf")


def test_open_dash_t_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -t /path/to/file.pdf")


def test_open_dash_f_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -f")


def test_open_dash_g_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -g /path/to/file.pdf")


def test_open_dash_n_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -n /path/to/file.pdf")


def test_open_dash_W_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -W /path/to/file.pdf")


def test_open_dash_h_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -h")


def test_open_dash_b_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -b com.apple.Preview")


def test_open_dash_D_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -D /path/to/file.pdf")


def test_open_dash_a_with_no_file_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open -a Preview")


# --- Darwin: non-`open` commands are NOT transformed ------------------------

def test_echo_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("echo hello")


def test_ls_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("ls -la")


def test_cat_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("cat file.txt")


def test_compound_command_not_transformed(monkeypatch):
    _darwin(monkeypatch)
    _assert_unchanged("open /path/to/file.pdf && echo done")


def test_none_input_returns_none(monkeypatch):
    _darwin(monkeypatch)
    assert terminal_tool._transform_macos_open_command(None) is None

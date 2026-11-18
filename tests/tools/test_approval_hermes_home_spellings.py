"""A sed -i of Hermes's own config.yaml is gated whichever way the shell spells HERMES_HOME (#144033).

The home fold knew only the absolute native path. On Windows the default home is
``%LOCALAPPDATA%\\hermes``, and the agent's Git Bash terminal writes that file as
``/c/Users/<me>/AppData/Local/hermes/config.yaml``, ``~/AppData/Local/hermes/config.yaml``,
``$HOME/AppData/Local/hermes/config.yaml`` or ``$LOCALAPPDATA/hermes/config.yaml``. Every one of
those passed as safe, so an in-place edit there rewrote the approval policy (approvals.mode, yolo,
allowlist) mid-session. Platform-independent: the spellings are folded from the resolved home, and a
custom HERMES_HOME under the user home reproduces the same gap on any OS.
"""

import pytest

from tools import approval_home_fold
from tools.approval import detect_dangerous_command

_PROFILE = r"C:\Users\tester"


def _is_dangerous(cmd: str) -> bool:
    return bool(detect_dangerous_command(cmd)[0])


@pytest.fixture
def windows_default_home(monkeypatch):
    """The Windows default HERMES_HOME, reached through the profile the way Git Bash spells it."""
    monkeypatch.setenv("HOME", _PROFILE)
    monkeypatch.setenv("USERPROFILE", _PROFILE)
    monkeypatch.setenv("LOCALAPPDATA", _PROFILE + r"\AppData\Local")
    monkeypatch.setenv("HERMES_HOME", _PROFILE + r"\AppData\Local\hermes")


class TestSpellingsOfTheHermesHome:
    def test_the_native_and_forward_slash_forms_were_always_gated(self, windows_default_home):
        assert _is_dangerous("sed -i s/a/b/ ~/.hermes/config.yaml") is True
        assert _is_dangerous("sed -i s/a/b/ C:/Users/tester/AppData/Local/hermes/config.yaml") is True

    @pytest.mark.parametrize("target", [
        "/c/Users/tester/AppData/Local/hermes/config.yaml",  # Git Bash drive form
        "/C/Users/tester/AppData/Local/hermes/config.yaml",
        "~/AppData/Local/hermes/config.yaml",  # home-relative
        "$HOME/AppData/Local/hermes/config.yaml",
        "${HOME}/AppData/Local/hermes/config.yaml",
        "$LOCALAPPDATA/hermes/config.yaml",  # the variable the default is built from
        "${LOCALAPPDATA}/hermes/config.yaml",
        "${USERPROFILE}/AppData/Local/hermes/config.yaml",
        r"%LOCALAPPDATA%\hermes\config.yaml",
    ])
    def test_every_spelling_of_the_windows_default_home_is_gated(self, windows_default_home, target):
        assert _is_dangerous(f"sed -i s/a/b/ {target}") is True

    def test_the_hermes_env_file_is_gated_too(self, windows_default_home):
        assert _is_dangerous("sed -i s/a/b/ /c/Users/tester/AppData/Local/hermes/.env") is True
        assert _is_dangerous("sed -i s/a/b/ $LOCALAPPDATA/hermes/.env") is True

    def test_the_long_in_place_flag_and_perl_are_gated(self, windows_default_home):
        assert _is_dangerous("sed --in-place s/a/b/ ~/AppData/Local/hermes/config.yaml") is True
        assert _is_dangerous("perl -pi -e s/a/b/ ${HOME}/AppData/Local/hermes/config.yaml") is True

    def test_a_custom_home_under_the_user_home_is_gated_on_any_os(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        monkeypatch.setenv("HERMES_HOME", "/home/tester/hermes-data")
        assert _is_dangerous("sed -i s/a/b/ ~/hermes-data/config.yaml") is True
        assert _is_dangerous("sed -i s/a/b/ $HOME/hermes-data/config.yaml") is True

    @pytest.mark.platforms("windows")
    def test_a_home_spelled_in_another_case_still_folds_on_windows(self, monkeypatch):
        monkeypatch.setenv("HOME", r"c:\users\tester")
        monkeypatch.setenv("HERMES_HOME", r"C:\Users\tester\AppData\Local\hermes")
        assert _is_dangerous("sed -i s/a/b/ ~/AppData/Local/hermes/config.yaml") is True


class TestPathBelowCaseFold:
    """``_path_below`` folds case only on a Windows host, so a HOME and HERMES_HOME that differ
    only in case still nest there. Runs on every lane by stubbing the host fact."""

    @pytest.mark.parametrize(("family", "expected"), [("win32", "AppData/Local/hermes"), ("linux", None)])
    def test_case_fold_follows_the_host_os_family(self, monkeypatch, family, expected):
        monkeypatch.setattr(approval_home_fold.facts, "os_family", lambda: family)
        assert approval_home_fold._path_below(r"c:\users\tester", r"C:\Users\tester\AppData\Local\hermes") == expected


class TestUnrelatedPathsStaySafe:
    """Folding a spelling must not turn a sibling config.yaml into Hermes's own policy file."""

    @pytest.mark.parametrize("target", [
        "/c/Users/tester/projects/app/config.yaml",
        "~/projects/app/config.yaml",
        "$LOCALAPPDATA/other/config.yaml",
        "/c/Users/tester/AppData/Local/hermes2/config.yaml",
    ])
    def test_a_sibling_config_yaml_is_not_flagged_as_hermes_policy(self, windows_default_home, target):
        assert _is_dangerous(f"sed -i s/a/b/ {target}") is False

    def test_a_sibling_of_a_custom_home_is_not_flagged(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        monkeypatch.setenv("HERMES_HOME", "/home/tester/hermes-data")
        assert _is_dangerous("sed -i s/a/b/ ~/hermes-data-old/config.yaml") is False
        assert _is_dangerous("sed -i s/a/b/ $HOME/hermes-data-old/config.yaml") is False


class TestShellEscapedSpaceInWindowsHome:
    r"""Git Bash writes an unquoted path with a space as ``John\ Smith``. The fold has to see through
    the backslash, or a profile whose name contains a space loses the gate on its config.yaml."""

    def test_the_escaped_profile_space_still_gates_a_config_edit(self, monkeypatch):
        profile = r"C:\Users\John Smith"
        monkeypatch.setenv("HOME", profile)
        monkeypatch.setenv("USERPROFILE", profile)
        monkeypatch.setenv("LOCALAPPDATA", profile + r"\AppData\Local")
        monkeypatch.setenv("HERMES_HOME", profile + r"\AppData\Local\hermes")
        assert _is_dangerous(r"sed -i s/a/b/ /c/Users/John\ Smith/AppData/Local/hermes/config.yaml") is True
        assert _is_dangerous(r"sed -i s/a/b/ /c/Users/John\ Smith/AppData/Local/hermes2/config.yaml") is False


class TestEscapedSpaceFollowsShellQuoting:
    r"""``\ `` is the same character as a space only where the shell reads it as an escape. Folding it
    anywhere else names a directory the command does not touch, and gates a harmless command as an
    edit of Hermes's own home."""

    @pytest.fixture
    def spaced_hermes_home(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        monkeypatch.setenv("HERMES_HOME", "/home/tester/hermes data")

    @pytest.fixture
    def spaced_user_home(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/John Smith")
        monkeypatch.setenv("HERMES_HOME", "/home/John Smith/hermes")

    def test_the_escaped_home_itself_is_still_gated(self, spaced_hermes_home):
        assert _is_dangerous(r"mv /home/tester/hermes\ data /home/tester/backup") is True
        assert _is_dangerous(r"sed -i s/a/b/ /home/tester/hermes\ data/config.yaml") is True

    def test_an_escaped_space_after_the_home_continues_the_name(self, spaced_hermes_home):
        # `hermes\ data\ backup` is one directory, a sibling of the home, not the home plus a word.
        assert _is_dangerous(r"mv /home/tester/hermes\ data\ backup /home/tester/backup") is False

    def test_an_escaped_space_after_a_spaceless_home_continues_the_name(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        monkeypatch.setenv("HERMES_HOME", "/home/tester/hermes")
        assert _is_dangerous(r"mv /home/tester/hermes\ backup /home/tester/backup") is False

    def test_an_unquoted_escaped_space_folds(self, spaced_user_home):
        assert _is_dangerous(r"sed -i s/a/b/ /home/John\ Smith/hermes/config.yaml") is True
        assert _is_dangerous(r"sed -i s/a/b/ /home/John\ Smith/hermes2/config.yaml") is False

    @pytest.mark.parametrize("command", [
        r"sed -i s/a/b/ '/home/John\ Smith/hermes/config.yaml'",
        r'sed -i s/a/b/ "/home/John\ Smith/hermes/config.yaml"',
        r"mv '/home/John\ Smith/hermes' /tmp/x",
    ])
    def test_a_quoted_backslash_is_a_literal_name_character(self, spaced_user_home, command):
        assert _is_dangerous(command) is False

    @pytest.mark.parametrize("command", [
        r'bash -c "sed -i s/a/b/ /home/John\ Smith/hermes/config.yaml"',
        r'echo "$(sed -i s/a/b/ /home/John\ Smith/hermes/config.yaml)"',
    ])
    def test_an_escape_read_by_an_inner_shell_still_folds(self, spaced_user_home, command):
        assert _is_dangerous(command) is True

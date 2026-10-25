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
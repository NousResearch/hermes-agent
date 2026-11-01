"""Regression coverage for option operands in the #94534 non-interactive fix.

All commands are parser inputs only; no sudo process or real prompt is allowed.
"""

import pytest

from tools import terminal_tool_sudo


@pytest.mark.parametrize("configured_password", [False, True])
@pytest.mark.parametrize("command", [
    "sudo -D / -n id",
    "sudo --chdir / --non-interactive id",
    "sudo --chroot / -n id",
    "sudo -R / -n id",
    "sudo -D '/path with spaces' -n id",
    "sudo --chdir '/path with spaces' -n id",
    "sudo --chroot '/path with spaces' -n id",
    "sudo -D/ -n id",
    "sudo --chdir=/ -n id",
    "sudo --chroot=/ -n id",
    "sudo -ED / -n id",
    "sudo -D -n -n id",
    "env X=1 /usr/bin/sudo -D / -n id",
    "sudo -nu root id",
    "sudo -n docker inspect x --format '{{json .Mounts}}' 2>&1",
    "sudo --host localhost -n id",
    "sudo -h localhost -n id",
    "sudo -Hn id",
    "sudo -Bn id",
    "sudo -An id",
    "sudo -Nn id",
    "sudo -Sn id",
    'sudo "-n" id',
    'sudo --user root "--non-interactive" id',
    "sudo '-D' / '-n' id",
])
def test_noninteractive_directory_options_never_prompt_or_inject(
    monkeypatch, command, configured_password
):
    if configured_password:
        monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    else:
        monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(terminal_tool_sudo, "_no_sudo_user", lambda: False)
    monkeypatch.setattr(terminal_tool_sudo, "_get_cached_sudo_password", lambda: "")

    def fail_interaction(*args, **kwargs):
        raise AssertionError("non-interactive sudo must not probe or prompt")

    monkeypatch.setattr(terminal_tool_sudo, "_prompt_for_sudo_password", fail_interaction)
    assert terminal_tool_sudo._transform_sudo_command(
        command, sudo_nopasswd_check=fail_interaction
    ) == (command, None)


@pytest.mark.parametrize("command", [
    "sudo -D -n id",
    "sudo --chdir -n id",
    "sudo --chroot -n id",
    "sudo -Dn id",
    "sudo --chdir=-n id",
    "sudo --chroot=-n id",
    "sudo -D / id -n",
    "sudo --chdir / -- -n",
    "sudo -u -n id",
    "sudo -un id",
    "sudo ls -n",
    "sudo -- -n",
    "sudo --host -n id",
    "sudo -h -n id",
    "sudo -hn id",
    "sudo -h localhost id -n",
    "sudo '-D' '-n' id",
    "sudo -D / # -n is only a comment",
    "sudo -D # -n is only a comment",
    'sudo "unterminated -n',
])
def test_option_values_and_child_flags_keep_password_path(monkeypatch, command):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    assert terminal_tool_sudo._transform_sudo_command(command) == (
        command.replace("sudo ", "sudo -S -p '' ", 1), "testpass\n"
    )


@pytest.mark.parametrize("separator", [" && ", " || ", "; ", "\n"])
def test_directory_probe_does_not_change_adjacent_interactive_sudo(monkeypatch, separator):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    probe = "sudo -D / -n id"
    command = probe + separator + "sudo id"
    assert terminal_tool_sudo._transform_sudo_command(command) == (
        probe + separator + "sudo -S -p '' id", "testpass\n"
    )

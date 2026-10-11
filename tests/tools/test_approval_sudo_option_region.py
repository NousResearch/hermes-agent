"""The sudo privilege-flag rules read only sudo's own option region.

sudo stops parsing options at the first non-option word, so a dash word after the command name
belongs to the sudo'd program. The old ``\\bsudo\\b[^;|&\\n]*?\\s+...`` rules bridged across the
command and read its flags as sudo's: ``sudo ls -la``, ``sudo cp -a``, ``sudo -n sqlite3 -readonly``
(combined-flag rule) and ``sudo systemctl list-units --state=failed`` / ``sudo ls --all``
(``--st*`` / ``--a*`` long rule). sudo's own -S/-A/-s/-a before the command, including after option
operands, quoted option values and VAR=value words, must still be flagged.
"""

import time

import pytest

from tools.approval import detect_dangerous_command

_SUDO_DESCS = (
    "sudo with privilege flag (stdin/askpass/shell/list)",
    "sudo with combined-flag privilege escalation",
)


@pytest.mark.parametrize("cmd", [
    "sudo -nS cmd",
    "sudo -n -s",
    "sudo -sa",
    "sudo -las",
    "sudo -su root",
    "sudo -S id",
    "sudo -A id",
    "sudo -s",
    "sudo -a",
    "sudo --stdin id",
    "sudo --askpass id",
    "sudo --st id",
    "sudo -i --stdin apt upgrade",
    "sudo -n -u root -S id",
    "sudo -u root -nS id",
    "sudo -nu root -S id",
    "sudo --user root -S id",
    "sudo --user root --stdin id",
    "sudo --us root -S id",
    "sudo --user=root -S id",
    "sudo -uroot -S id",
    "sudo -kn -sa id",
    # Option values with spaces, quoted separately, attached, or after `=`.
    'sudo -p "Password: " -S id',
    "sudo -p 'pw: ' -S id",
    "sudo -p'pw: ' -S id",
    'sudo --prompt="pw: " -A id',
    "sudo -g wheel -C 5 -D /tmp -T 10 -A id",
    "sudo -C 3 -A id",
    "sudo FOO=bar -S id",
    # An option operand never starts with `-`, so -S is still sudo's flag here.
    "sudo -u -S id",
    "echo pw | sudo -n -u root -S id",
    "ls; sudo -n -A id",
    "SUDO_ASKPASS=/tmp/x sudo -A apt update",
])
def test_privilege_flags_before_the_command_are_flagged(cmd):
    is_dangerous, _key, desc = detect_dangerous_command(cmd)
    assert is_dangerous is True, cmd
    assert desc in _SUDO_DESCS, (cmd, desc)


@pytest.mark.parametrize("cmd", [
    "sudo ls -la /opt/app/data/",
    "ssh host1 'sudo ls -la /opt/app/data/'",
    'ssh host1 "sudo ls -la /opt/x && sudo cat /opt/y"',
    "ssh host1 'sudo ls -la /opt/x; sudo grep -rl foo /opt/y'",
    "sudo cp -a /src /dst",
    "sudo rsync -a /srv/app/ /opt/app/",
    "sudo ss -atp",
    "sudo df -a",
    "sudo tar -xaf /tmp/b.tar",
    'sudo -n sqlite3 -readonly /var/lib/app/app.db "select 1"',
    'sudo -n -u app sqlite3 --readonly /var/lib/app/app.db "select 1"',
    "sudo -u postgres psql -As -c 'select 1'",
    "sudo -n sysctl -a",
    "sudo -u root sysctl -a",
    "sudo --user=root sysctl -a",
    "sudo -n apt-get -s dist-upgrade",
    # Long options of the command no longer match the --st* / --a* rule.
    "sudo -n /usr/bin/systemctl list-units --state=failed",
    "sudo -n systemctl list-timers --all",
    "sudo ls --all /opt",
    "sudo apt --assume-yes install curl",
    "sudo tar --atime-preserve -xf a.tar",
    # -H and -P take no value (case matters): the next word is the command.
    "sudo -H ls -la /opt/x",
    "sudo -H -u alice env FOO=1 tar -xas archive.tar",
    "sudo --preserve-env=PATH ls -la /opt/x",
    "sudo -nv ls -la /opt/x",
    "sudo --non-interactive ls -la /opt/x",
    "sudo -- ls -las",
])
def test_flags_of_the_sudoed_program_are_not_sudo_flags(cmd):
    _is_dangerous, _key, desc = detect_dangerous_command(cmd)
    assert desc not in _SUDO_DESCS, (cmd, desc)


def test_option_region_scan_is_linear_on_adversarial_input():
    for cmd in ("sudo " + "-u " * 4000 + "x", "sudo " + "-p 'a b' " * 2000 + "x"):
        start = time.monotonic()
        detect_dangerous_command(cmd)
        assert time.monotonic() - start < 5

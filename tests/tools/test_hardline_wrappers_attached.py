"""Regression coverage for the wrapper-prefix and attached-quote hardline bypasses (#131563).

Two bypass classes against the hardline floor:

(a) Wrapper prefixes: `_CMDPOS` only skipped ``exec|nohup|setsid|time``, so
    ``timeout 1 shutdown now`` (and nice/stdbuf/prlimit/flock/unbuffer/xargs/
    command/builtin/eval/screen/tmux, plus option values owned by sudo/env)
    reached the dangerous command without the floor firing.

(b) Attached-quote interpreter flags: ``bash -c'shutdown now'`` merges flag and
    payload into one token, so ``_bash_exec_payload`` saw no ``-c`` option at
    all and returned no payload to scan.
"""

import pytest

from tools.approval_detection import detect_hardline_command


@pytest.mark.parametrize(
    "cmd",
    [
        "timeout 1 shutdown now",
        "nice shutdown now",
        "stdbuf -o0 shutdown now",
        "prlimit --nproc=64 shutdown now",
        "flock /tmp/hermes.lock shutdown now",
        "unbuffer shutdown now",
        "xargs shutdown now",
        "command shutdown now",
        "builtin shutdown now",
        "eval shutdown now",
        "screen shutdown now",
        "tmux shutdown now",
        "nohup timeout 1 shutdown now",
        "sudo -u alice shutdown now",
        "env -i shutdown now",
        "ls; timeout 5 mkfs.ext4 /dev/sda1",
        "timeout 30 rm -rf /",
    ],
)
def test_wrapper_prefixes_reach_the_hardline_floor(cmd):
    assert detect_hardline_command(cmd)[0] is True, cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "bash -c'shutdown now'",
        "sh -c'shutdown now'",
        'sh -c"reboot"',
        "bash -ec'rm -rf /'",
        "bash -lc'shutdown now'",
    ],
)
def test_attached_quote_flag_payloads_reach_the_hardline_floor(cmd):
    assert detect_hardline_command(cmd)[0] is True, cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "timeout 60 curl https://example.com/big.tar.gz -o /tmp/x",
        "nice -n 10 make build",
        "stdbuf -oL grep -r pattern logs/",
        "flock /tmp/hermes.lock python backup.py",
        "grep timeout /var/log/syslog",
        'echo "bash -c\'shutdown now\'" is just data',
    ],
)
def test_benign_wrapper_and_attached_forms_stay_quiet(cmd):
    assert detect_hardline_command(cmd) == (False, None), cmd

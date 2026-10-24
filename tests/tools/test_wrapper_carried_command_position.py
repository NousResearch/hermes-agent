"""Regression coverage for #71996 wrapper-carried command positions.

Verdict changes versus tip, measured by
``evals/postmortem/review_probes/wrapper_projection_parity_probe.py`` over the
3580 distinct whitespace-containing string constants in the matching test
files (this head's corpus, evaluated on both trees, with deny globs
``rm -rf*`` / ``reboot`` and allowlist ``su *`` / ``doas *`` / ``watch *`` /
``ls``), are exactly:

- hardline False -> True (9): ``su -c 'rm -rf /'``,
  ``script -c 'rm -rf /' /dev/null``, ``flock /tmp/l -c 'rm -rf /'``,
  ``doas sh -c ':(){ :|:& };:'``, ``runuser -u root -- reboot``,
  ``runuser -u root reboot``, ``runuser -l root -c reboot``,
  ``runuser -l root -c 'rm -rf /'``, and ``su root -c reboot``;
- dangerous: no change (payloads only change that verdict when they
  independently match ``DANGEROUS_PATTERNS``, and those inputs already
  matched on the raw text);
- user deny glob False -> True (16): the hardline inputs above except the
  ``doas sh -c`` fork bomb, plus ``doas rm -rf /``, ``flock /tmp/l rm -rf /``,
  ``systemd-run --scope rm -rf /``, ``xargs -I {} rm -rf {}``,
  ``xargs rm -rf /``, ``watch -n1 rm -rf /tmp/x``, ``su -c '/sbin/reboot'``
  and ``su root -c '/sbin/reboot'`` (the last two through the deny side's
  basename projection only);
- permanent allowlist True -> False, glob entries only (4):
  ``su -c 'rm -rf /'``, ``su root -c reboot`` (under ``su *``),
  ``doas rm -rf /``, and ``watch -n1 rm -rf /tmp/x``.

Nothing else changed. Absolute-path spellings such as ``/sbin/reboot`` stay
outside the hardline and dangerous projections until #82830.
"""

import os
import time

import pytest

from tools.approval_detection import (
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_word_spans,
    _wrapper_carried_payloads,
    detect_dangerous_command,
    detect_hardline_command,
)


def _command_names(command: str) -> list[str]:
    return [
        os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower()
        for _, _, word in _iter_shell_command_word_spans(command)
    ]


@pytest.mark.parametrize(
    ("command", "expected_prefix"),
    [
        ("doas rm -rf /", ["doas", "rm"]),
        ("xargs -I {} rm -rf {}", ["xargs", "rm"]),
        ("watch -n1 rm -rf /tmp/x", ["watch", "rm"]),
        ("watch -n 1 ls", ["watch", "ls"]),
        ("flock /tmp/l rm -rf /", ["flock", "rm"]),
        ("systemd-run --scope rm -rf /", ["systemd-run", "rm"]),
    ],
)
def test_wrapper_walk_reaches_carried_executable(command, expected_prefix):
    names = _command_names(command)
    assert names[:len(expected_prefix)] == expected_prefix


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("su - alice", ["su"]),
        ("script session.log", ["script"]),
        ("flock -n /tmp/l true", ["flock", "true"]),
        ("xargs echo", ["xargs", "echo"]),
        ("doas ls", ["doas", "ls"]),
        ("watch ls", ["watch", "ls"]),
        ("watch -c ls", ["watch", "ls"]),
        ("watch -d ls", ["watch", "ls"]),
        ("xargs -e echo", ["xargs", "echo"]),
        ("flock -w 5 /tmp/l ls", ["flock", "ls"]),
        ("systemd-run --on-active 5m ls", ["systemd-run", "ls"]),
        ("systemd-run --on-clock-change ls", ["systemd-run", "ls"]),
        ("xargs", ["xargs"]),
        ("doas -C /tmp/doas.conf rm -rf /", ["doas"]),
        ("doas -C/tmp/doas.conf rm -rf /", ["doas"]),
        ("systemd-run -S rm -rf /", ["systemd-run"]),
        ("systemd-run --shell rm -rf /", ["systemd-run"]),
    ],
)
def test_wrapper_walk_preserves_option_and_non_execution_boundaries(command, expected):
    assert _command_names(command) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("su -c 'rm -rf /'", ["rm -rf /"]),
        ("su root -c reboot", ["reboot"]),
        ("runuser -l root -c reboot", ["reboot"]),
        ("script -c 'rm -rf /' /dev/null", ["rm -rf /"]),
        ("script -E always -c ls out.txt", ["ls"]),
        ("flock /tmp/l -c 'rm -rf /'", ["rm -rf /"]),
        ("flock -c 'ls' /tmp/l", ["ls"]),
        ("runuser -u root -- reboot", ["reboot"]),
        ("runuser -u root reboot", ["reboot"]),
        ("runuser -u root git clean -fdx", ["git clean -fdx"]),
        ("runuser -u root -- printf '%s %s' one two", ["printf '%s %s' one two"]),
        ("runuser -u root printf '%s %s' one two", ["printf '%s %s' one two"]),
        ("su -c '/sbin/reboot'", ["/sbin/reboot"]),
        ("su - alice", []),
        ("script session.log", []),
        ("flock -n /tmp/l true", []),
        ("watch -c ls", []),
    ],
)
def test_carrier_payload_extraction(command, expected):
    assert list(_wrapper_carried_payloads(command)) == expected


@pytest.mark.parametrize(
    "command",
    [
        "su -c 'rm -rf /'",
        "script -c 'rm -rf /' /dev/null",
        "flock /tmp/l -c 'rm -rf /'",
        "doas sh -c ':(){ :|:& };:'",
        "runuser -u root -- reboot",
        "runuser -u root reboot",
        "su root -c reboot",
        "runuser -l root -c reboot",
    ],
)
def test_expected_hardline_crossovers(command):
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize(
    "command",
    [
        "su -c ls",
        "script -c ls out.txt",
        'flock -c "ls" /tmp/l',
        "su -c ls alice",
        "runuser -u root -- ls",
    ],
)
def test_benign_carrier_payload_does_not_create_dangerous_finding(command):
    assert detect_dangerous_command(command)[0] is False


@pytest.mark.parametrize(
    "command",
    [
        "su -c '/sbin/reboot'",
        "su root -c '/sbin/reboot'",
    ],
)
def test_absolute_path_hardline_projection_remains_deferred(command):
    assert detect_hardline_command(command)[0] is False
    assert detect_dangerous_command(command)[0] is False


_PERFORMANCE_COMMANDS = tuple(
    f"{wrapper} printf item-{index}"
    for index in range(40)
    for wrapper in (
        "doas",
        "xargs",
        "watch",
        "flock /tmp/hermes-wrapper-lock",
        "systemd-run --scope",
    )
)


def test_wrapper_projection_detection_has_bounded_runtime():
    started = time.perf_counter()
    for command in _PERFORMANCE_COMMANDS:
        detect_hardline_command(command)
        detect_dangerous_command(command)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"200 hardline+dangerous classifications took {elapsed:.3f}s"

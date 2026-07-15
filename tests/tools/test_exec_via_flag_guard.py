"""Exec-via-flag escape detection (port of Kilo-Org/kilocode#11890).

Otherwise read-only commands can execute an arbitrary program through a
flag: ``sort --compress-program``, ``rg --pre`` / ``--hostname-bin``,
``ag --pager``, ``man -P`` / ``--pager`` / ``-H`` / ``--html``.  Two layers
are under test:

1. The mechanism itself is flagged dangerous (approval required) even when
   the payload is opaque (``--pre=sh``).
2. A hardline payload smuggled through the flag value reaches the
   unconditional floor via ``_exec_flag_payloads`` detection variants.
"""

import pytest

from tools.approval import (
    _exec_flag_payloads,
    detect_dangerous_command,
    detect_hardline_command,
)


@pytest.mark.parametrize("command", [
    "sort --compress-program='rm -rf ~' names.txt",
    'sort -S 1b --compress-program "sh" names.txt',
    "sort --compress-program=sh names.txt",
    "rg --pre sh -e . names.txt",
    "rg --pre=sh -e . names.txt",
    "rg --hostname-bin=sh pattern",
    "ag --pager sh foo",
    "ag --pager=sh foo",
    "man -P sh ls",
    "man -Psh ls",
    "man --pager=sh ls",
    "man --html=sh ls",
])
def test_exec_via_flag_is_dangerous(command):
    is_dangerous, _key, desc = detect_dangerous_command(command)
    assert is_dangerous, f"exec-via-flag escape not detected: {command}"


@pytest.mark.parametrize("command", [
    # A hardline payload inside the flag value must reach the unconditional
    # floor, not just the softer dangerous prompt.
    "sort --compress-program='rm -rf --no-preserve-root /' names.txt",
    'sort --compress-program "rm -rf --no-preserve-root /" names.txt',
    "rg --pre 'rm -rf --no-preserve-root /' -e . x",
    "man -P 'rm -rf --no-preserve-root /' ls",
    # man -H (browser) is invisible to the lowercased dangerous-pattern
    # matcher (would collide with `man -h` help), but the payload extractor
    # runs on the original-case command, so a dangerous payload still
    # reaches the floor.
    "man -H 'rm -rf --no-preserve-root /' ls",
    "ag --pager 'rm -rf --no-preserve-root /' foo",
])
def test_hardline_payload_in_exec_flag_hits_floor(command):
    is_hardline, desc = detect_hardline_command(command)
    assert is_hardline, f"hardline payload escaped the floor: {command}"
    assert desc == "recursive delete of root filesystem"


@pytest.mark.parametrize("command", [
    "sort names.txt",
    "sort -k2 -n data.csv -o out.csv",
    "rg 'compress-program' docs/",
    "rg -n pattern src/",
    "rg --pretty pattern src/",       # --pre must not match --pretty
    "ag foo src/",
    "man ls",
    "man -k pager",                    # -k arg happens to be the word "pager"
    "grep -P 'foo(?=bar)' file.txt",   # -P is perl-regex on grep, not a pager
    "pip install --pre somepackage",   # --pre without rg context
    "git log --oneline",
])
def test_legit_commands_not_flagged(command):
    is_hardline, _ = detect_hardline_command(command)
    is_dangerous, _key, desc = detect_dangerous_command(command)
    assert not is_hardline, f"legit command hardlined: {command}"
    assert not (is_dangerous and "via" in (desc or "")), (
        f"legit command flagged as exec-via-flag: {command} ({desc})"
    )


def test_exec_flag_payload_extraction():
    payloads = list(_exec_flag_payloads(
        "sort --compress-program='rm -rf /' a.txt"))
    assert payloads == ["rm -rf /"]

    payloads = list(_exec_flag_payloads("man -Psh ls"))
    assert payloads == ["sh"]

    payloads = list(_exec_flag_payloads("rg --pre=sh -e . x"))
    assert payloads == ["sh"]

    # Long flag with no rg/man/sort context still extracts, but a benign
    # payload matches no dangerous pattern (defense-in-depth only).
    assert list(_exec_flag_payloads("sort data.txt")) == []

    # A flag value that is itself another flag is not a payload.
    assert list(_exec_flag_payloads("man --pager --html ls")) == []

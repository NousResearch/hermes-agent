"""Regression for #132191: hardline detection must survive brace-expansion and ANSI-C quoting.

The hardline floor is documented as never bypassable, but two ordinary bash word-spellings reached
the detector as unrecognised text, so the command ran with no approval:

    {shutdown,now}            # brace expansion -> `shutdown now`
    $'\\x72\\x6d' -rf /         # ANSI-C quoting  -> `rm -rf /`

These are detector-only checks: the strings are classified, never executed.
"""

import pytest

from tools.approval_detection import detect_hardline_command


def _hl(cmd: str) -> bool:
    return detect_hardline_command(cmd)[0]


# Plain spellings are the control: they must already be hardline.
@pytest.mark.parametrize("cmd", ["shutdown now", "rm -rf /"])
def test_plain_spelling_is_hardline(cmd):
    assert _hl(cmd) is True


@pytest.mark.parametrize("cmd", [
    "{shutdown,now}",          # expands to: shutdown now
    "{rm,-rf,/}",              # expands to: rm -rf /
])
def test_brace_expansion_spelling_is_hardline(cmd):
    assert _hl(cmd) is True


@pytest.mark.parametrize("cmd", [
    "$'\\x72\\x65\\x62\\x6f\\x6f\\x74'",       # hex -> reboot
    "$'\\162\\145\\142\\157\\157\\164'",       # octal -> reboot
    "$'\\x72\\x6d' -rf /",                      # hex -> rm, then -rf /
])
def test_ansi_c_quoting_spelling_is_hardline(cmd):
    assert _hl(cmd) is True


# The deobfuscation must not start flagging benign commands.
@pytest.mark.parametrize("cmd", [
    "echo {a,b,c}",                    # brace list in argument position, harmless program
    "git commit -m 'reboot the box'",  # 'reboot' is quoted prose, not a command
    "echo $'hello world'",             # ANSI-C string that is not a command name
    "ls -la {src,tests}",              # brace list of real directories
])
def test_benign_commands_stay_non_hardline(cmd):
    assert _hl(cmd) is False

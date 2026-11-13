"""Bash/zsh ANSI-C quoting (``$'...'``) must not hide a command from the approval floors (#76218)."""

import pytest

from tools.approval import detect_dangerous_command, detect_hardline_command


@pytest.mark.parametrize("command", [
    r"$'\x72\x6d' -rf /",
    r"$'\162\155' -rf /",
    r"$'\u0072m' -rf /",
    r"r$'m' -rf /",
    r"rm -rf $'\x2f'",
    r"rm -rf $'/'",
    r"$'\x72\x6d' -rf $'\x2f'",
    r"$'\x72\x6d' -rf ~",
    r'echo "$($' + r"'\x72\x6d' -rf /)" + '"',
    '$"rm" -rf /',
])
def test_ansi_c_spelled_wipe_hits_hardline(command):
    assert detect_hardline_command(command)[0] is True


def test_escaped_quote_inside_ansi_c_does_not_hide_next_command():
    # `\'` does not close a $'...' span; reading it as `$` + `'\'` left `; rm -rf /` "inside" a quote.
    assert detect_hardline_command(r"echo $'\'' ; rm -rf /")[0] is True
    assert detect_hardline_command("echo $'\\''\nrm -rf /")[0] is True


def test_ansi_c_encoded_absolute_home_folds_to_home_rule(monkeypatch):
    home = "/home/ansi-user"
    monkeypatch.setenv("HOME", home)
    encoded = "".join(f"\\x{ord(ch):02x}" for ch in f"{home}/*")
    assert detect_hardline_command(f"rm -rf $'{encoded}'")[0] is True


@pytest.mark.parametrize("command", [
    r"echo $'\x72\x6d' -rf /",
    r"echo $'(reboot)'",
    r"echo $'\x28reboot\x29'",
    "git commit -m $'subj\\n\\nrm -rf /'",
    r"echo '$'\x72\x6d'' -rf /",
    r"echo $'unterminated -rf /",
])
def test_ansi_c_data_is_not_promoted_to_a_hardline_command(command):
    assert detect_hardline_command(command)[0] is False


def test_double_quoted_dollar_quote_stays_literal():
    # Inside double quotes `$'` is literal text, not ANSI-C quoting.
    assert detect_dangerous_command('echo "$\'\\x72\\x6d -rf /\'"')[0] is False

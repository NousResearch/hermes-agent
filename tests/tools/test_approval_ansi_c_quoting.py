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


@pytest.mark.parametrize("command", [
    r"echo $$'\'; mkfs.ext4 /dev/sda; echo ''",
    r"echo $$'\' ; rm -rf / ; echo ''",
    r"kill $$'\';rm -rf /;'",
])
def test_pid_expansion_before_quote_is_not_ansi_c(command):
    # `$$` is the PID, so `'\'` is a closed single-quoted backslash and the next command runs.
    assert detect_hardline_command(command)[0] is True
    assert detect_dangerous_command(command)[0] is True


def test_surrogate_escape_stays_encoded():
    # bash turns \ud800 into raw bytes (`$'\ud800rm'` runs a command named b"\xed\xa0\x80rm", not
    # rm), so it must not decode into a lone surrogate the rest of the pipeline cannot encode.
    from tools.approval_ansi_c import expand_ansi_c_quotes

    expand_ansi_c_quotes(r"$'\ud800rm' -rf /").encode("utf-8")


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


@pytest.mark.parametrize("command", [
    "$\\\n'\\x72\\x6d' -rf /",
    "$\\\n\\\n'\\x72\\x6d' -rf /",
])
def test_line_continuation_between_dollar_and_quote_is_still_ansi_c(command):
    # bash drops `\<newline>` before tokenizing, so `$\<newline>'...'` is a $'...' span.
    assert detect_hardline_command(command)[0] is True


def test_hash_after_escaped_space_is_not_a_comment():
    # `\ #` keeps `#` inside the word `\ #`, so it is literal and `; rm -rf /` runs.
    assert detect_hardline_command(r"echo \ # $'\'' ; rm -rf /")[0] is True
    # An escaped backslash before the space leaves the space unescaped: that `#` is a comment.
    assert detect_hardline_command(r"echo \\ # $'\'' ; rm -rf /")[0] is False


@pytest.mark.parametrize("command", [
    "bash -c \"$'\\\\x72\\\\x6d' -rf /\"",
    "bash -c \"rm -rf $'\\\\x2f'\"",
    "bash -c \"'rm' -rf /\"",
])
def test_shell_c_payload_is_expanded_as_its_own_shell_input(command):
    assert detect_hardline_command(command)[0] is True

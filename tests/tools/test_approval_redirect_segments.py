"""Output redirections must not hide later program-bearing arguments (#155470)."""

import pytest

from tools.approval import detect_dangerous_command
from tools.approval_applescript import OSASCRIPT_RISKY_DESCRIPTION
from tools.approval_detection import _shell_command_segment, _shell_segment_tokens


@pytest.mark.parametrize("redirect", ["2>&1", ">&2", "&>/dev/null", ">& both", "2>& 1", ">| out"])
def test_redirect_between_applescript_arguments(redirect):
    command = (f"osascript -e 'get name of me' {redirect} "
               "-e 'tell app \"Keynote\" to close front document saving no'")
    assert detect_dangerous_command(command)[1] == OSASCRIPT_RISKY_DESCRIPTION


def test_redirect_before_powershell_payload():
    found, description, _ = detect_dangerous_command('pwsh -NoProfile 2>&1 -Command "gci | ri -r"')
    assert found
    assert "delete" in description.lower()


@pytest.mark.parametrize("redirect", ["2>&1", ">&2", "&>/dev/null", ">& both", "2>& 1", ">| out"])
def test_read_only_trailing_redirect(redirect):
    assert detect_dangerous_command(f"osascript -e 'get name of me' {redirect}") == (False, None, None)


@pytest.mark.parametrize("suffix", [
    " & echo quit", " && echo quit", "; echo quit", " | echo quit", "\necho quit", " # quit",
    # A quoted or escaped `>` is a literal, so the `&` after it is a real separator.
    r" \>& echo quit", r" &\> echo quit", " '>'& echo quit",
    r" \>| echo quit", " '>'| echo quit",
])
def test_real_separators_still_bound_command(suffix):
    command = "osascript -e 'get name of me'" + suffix
    assert "echo quit" not in _shell_command_segment(command, 0)


@pytest.mark.parametrize("redirect,expected", [
    ("2>&1", ["2", ">", "&1"]),
    (">&2", [">", "&2"]),
    ("&>/dev/null", ["&", ">", "/dev/null"]),
])
def test_redirect_tokenization(redirect, expected):
    assert _shell_segment_tokens(redirect, 0) == expected


@pytest.mark.parametrize("redirect", ["0</dev/null", "0<&0", "0<&3"])
@pytest.mark.parametrize("later", ["", " -e 'get version of me'"])
def test_fd_input_redirect_is_opaque_manual_only(redirect, later):
    from tools.approval import SMART_MANUAL_ONLY_DESCRIPTIONS
    from tools.approval_applescript import OSASCRIPT_OPAQUE_DESCRIPTION

    found, description, _ = detect_dangerous_command(
        f"osascript -e 'get name of me' {redirect}{later}"
    )
    assert found
    assert description == OSASCRIPT_OPAQUE_DESCRIPTION
    assert description in SMART_MANUAL_ONLY_DESCRIPTIONS


@pytest.mark.parametrize("suffix", [
    " '0' '<' /dev/null", " '0</dev/null'", r" 0\</dev/null",
    " | echo '0</dev/null'", "; echo '0<&3'", " && echo '0</dev/null'",
    " >| out | echo quit", " >& both ; echo quit",
])
def test_input_literals_and_separated_commands_stay_read_only(suffix):
    assert detect_dangerous_command("osascript -e 'get name of me'" + suffix) == (False, None, None)

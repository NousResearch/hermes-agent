"""The command-approval parser guard must bound nesting depth, not just size.

The quote-aware command-start walk recurses once per ``$(...)``/backtick level with no depth
parameter, so ~1000 nested substitutions (a 3 KB command, far under every size ceiling) exhausted
the interpreter stack inside ``detect_hardline_command()``/``detect_dangerous_command()`` and
surfaced as an internal-exception envelope instead of a verdict (#137066). The hardline floor is
the non-bypassable check that runs before yolo / mode=off / allowlists; it must return a verdict,
never raise."""

import pytest

from tools.approval import detect_dangerous_command, detect_hardline_command
from tools.approval_detection import (
    _PARSER_LIMIT_DESCRIPTION,
    _command_parser_limit_exceeded,
    _command_substitution_depth,
)


def _nested_substitutions(depth: int, suffix: str = "") -> str:
    command = "echo hi"
    for _ in range(depth):
        command = "$(" + command + ")"
    return command + suffix


@pytest.mark.parametrize("suffix", ["", "; true"], ids=["plain", "separator-suffixed"])
@pytest.mark.parametrize("depth", [1000, 10000])
def test_deeply_nested_substitutions_fail_closed(depth, suffix):
    # A single separator anywhere disables the separator-free length check, so the suffixed
    # variant used to crash all the way up to the 128 KB ceiling.
    command = _nested_substitutions(depth, suffix)
    assert _command_parser_limit_exceeded(command)
    assert detect_hardline_command(command) == (True, _PARSER_LIMIT_DESCRIPTION)
    assert detect_dangerous_command(command) == (
        True,
        _PARSER_LIMIT_DESCRIPTION,
        _PARSER_LIMIT_DESCRIPTION,
    )


def test_unterminated_nesting_is_bounded_too():
    # Unterminated openers still recurse (the walk descends to the end on stop_unterminated),
    # so the depth probe counts them rather than waiting for a balanced closer.
    command = "$(" * 1000 + "echo hi"
    assert _command_substitution_depth(command) == 1000
    assert detect_hardline_command(command) == (True, _PARSER_LIMIT_DESCRIPTION)


def test_double_quoted_nesting_is_executable_and_bounded():
    # Substitutions inside double quotes execute in a real shell and the walk descends into
    # them, so the depth probe must count them too.
    command = '"' + _nested_substitutions(1000) + '"'
    assert _command_substitution_depth(command) == 1000
    assert detect_hardline_command(command) == (True, _PARSER_LIMIT_DESCRIPTION)


def test_depth_probe_is_quote_aware():
    assert _command_substitution_depth("echo $(date)") == 1
    assert _command_substitution_depth("echo $(basename $(pwd))") == 2
    assert _command_substitution_depth("echo `basename \\`pwd\\``") == 2
    assert (
        _command_substitution_depth("echo '$(date)'") == 0
    )  # single-quoted: literal data
    assert _command_substitution_depth('echo "$(basename $(pwd))"') == 2
    assert (
        _command_substitution_depth("echo (date)") == 0
    )  # a bare subshell paren is not a substitution
    assert _command_substitution_depth("echo \\$(date)") == 0  # escaped: not executed


def test_realistic_nesting_still_reaches_the_patterns():
    # Guard headroom: benign everyday nesting must keep flowing through the real detectors...
    assert _command_parser_limit_exceeded("echo $(cat $(ls))") is False
    assert detect_hardline_command("echo $(cat $(ls))") == (False, None)
    # ...and a hardline command nested at a realistic depth must still be caught by the
    # patterns, not absorbed by the depth ceiling.
    assert detect_hardline_command("echo $(sudo rm -rf /)") == (
        True,
        "recursive delete of root filesystem",
    )

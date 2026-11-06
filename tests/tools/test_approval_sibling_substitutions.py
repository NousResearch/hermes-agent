"""Regression coverage for #140091 sibling command substitutions."""

import time

from tools.approval_detection import (
    _iter_shell_command_word_spans,
    detect_dangerous_command,
    detect_hardline_command,
)


def test_backtick_sibling_command_words_end_at_their_own_closer():
    command = "rm `first` `second`"

    assert [word for _, _, word in _iter_shell_command_word_spans(command)] == ["rm", "first", "second"]


def test_shell_payload_inside_backtick_substitution_is_still_inspected():
    blocked, reason = detect_hardline_command("echo `bash -c 'rm -rf /'` `true`")

    assert blocked is True
    assert reason == "recursive delete of root filesystem"


def test_many_sibling_backticks_remain_bounded():
    command = "rm " + " ".join("`item`" for _ in range(400))

    started = time.monotonic()
    assert detect_hardline_command(command) == (False, None)
    assert detect_dangerous_command(command) == (False, None, None)
    assert time.monotonic() - started < 5.0

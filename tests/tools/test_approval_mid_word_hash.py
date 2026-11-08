"""A ``#`` is a comment only where a word begins: after an escaped space, a ``$(...)`` closer or a
quote it continues the word, so the commands after it must still be inspected."""

import pytest

from tools.approval_detection import detect_dangerous_command, detect_hardline_command

_MID_WORD = [
    "echo $(echo \\ #); @@",
    "echo \\ #; @@",
    "echo \\;#; @@",
    "echo $(id)#; @@",
    "echo `id`#; @@",
    'echo ""#; @@',
    "echo '#'#; @@",
    "echo \\ #'\nfoo'; @@ #'",
    "echo \"$(echo \\ #)\"; @@",
    "cat <(true)#; @@",
    "cat >(true)#; @@",
    "a=(1 2)#; @@",
    "a+=(1)#; @@",
    "cat <(echo \")\")#; @@",
    "echo \"(\"; cat <(true)#; @@",
    "cat <(true)#\n@@",
    "echo $(cat <(id)#; @@\n)",
    "(a)#'\n@@\n#'",
]
_REAL_COMMENTS = ["echo a # @@", "(a)# @@", "a;# @@", "echo \\\\ # @@", "echo $(id) # @@",
                  "(a)#; @@", "{ a; }# @@", "(a)\n# @@", "cat <(true) # @@"]


@pytest.mark.parametrize("template", _MID_WORD)
def test_mid_word_hash_does_not_hide_hardline(template):
    assert detect_hardline_command(template.replace("@@", "reboot"))[0]


@pytest.mark.parametrize("template", _MID_WORD)
def test_mid_word_hash_does_not_hide_dangerous(template):
    assert detect_dangerous_command(template.replace("@@", "python3 -c 'import os'"))[0]


@pytest.mark.parametrize("template", _MID_WORD)
def test_mid_word_hash_benign_stays_clean(template):
    assert not detect_hardline_command(template.replace("@@", "echo ok"))[0]
    assert not detect_dangerous_command(template.replace("@@", "echo ok"))[0]


@pytest.mark.parametrize("template", _REAL_COMMENTS)
def test_real_comments_stay_inert(template):
    assert not detect_hardline_command(template.replace("@@", "reboot"))[0]
    assert not detect_dangerous_command(template.replace("@@", "python3 -c 'import os'"))[0]


@pytest.mark.parametrize("command", [
    "echo \"\"#; bash -c 'reboot'",
    "echo \"\"#; sh -c 'reboot'",
    "echo \\ #; bash -c 'reboot'",
])
def test_exec_flag_payload_is_read_from_the_raw_text(command):
    assert detect_hardline_command(command)[0]
    assert detect_dangerous_command(command)[0]


def test_many_group_closers_followed_by_hash_do_not_stall():
    import time

    command = "(a)#\n" * 12000 + "cat <(true)#; reboot"
    started = time.monotonic()
    assert detect_hardline_command(command)[0]
    assert time.monotonic() - started < 10


_ARITHMETIC = [
    "false && echo $(( 1 # )); @@",
    "echo $(( 1 # ))\n@@",
    "echo $(( (1) # ))'\n'; @@",
    "x=$(( 2 #3 )); @@",
    "echo $((1 + $(( 2 # )))); @@",
    'echo "$(( 1 # ))"; @@',
    "echo $(( 1 # )) # tail\n@@",
]
_ARITHMETIC_COMMENTS = [
    "echo $(( 1 )) # @@",
    "# $(( 1 # )) @@",
    "echo '$(( 1 # ))' # @@",
    "echo $(( 16#ff )) # @@",
]


@pytest.mark.parametrize("template", _ARITHMETIC)
def test_hash_inside_arithmetic_is_not_a_comment(template):
    assert detect_hardline_command(template.replace("@@", "reboot"))[0]
    assert detect_dangerous_command(template.replace("@@", "python3 -c 'import os'"))[0]
    assert not detect_hardline_command(template.replace("@@", "echo ok"))[0]


@pytest.mark.parametrize("template", _ARITHMETIC_COMMENTS)
def test_comments_around_arithmetic_stay_inert(template):
    assert not detect_hardline_command(template.replace("@@", "reboot"))[0]


def test_unterminated_arithmetic_marks_nothing_and_stays_fast():
    import time

    started = time.monotonic()
    detect_hardline_command("echo $(( 1 # " * 3000 + "\nreboot #")
    assert time.monotonic() - started < 10


@pytest.mark.parametrize("payload", ["reboot", "python3 -c 'import os'"])
def test_comment_after_line_continuation_does_not_hide_command(payload):
    command = "echo a \\\n# '\n" + payload
    detector = detect_hardline_command if payload == "reboot" else detect_dangerous_command
    assert detector(command)[0]


@pytest.mark.parametrize("payload", ["reboot", "python3 -c 'import os'"])
def test_ambiguous_arithmetic_comment_reading_cannot_hide_command(payload):
    command = "echo $(( id # '\n)\n" + payload + "\n)"
    detector = detect_hardline_command if payload == "reboot" else detect_dangerous_command
    assert detector(command)[0]


@pytest.mark.parametrize("prefix,expected", [
    ("echo a \\\n", True),
    ("echo a\\\n", False),
    ("echo a\\\\\n", True),
    ("echo a\\\\\\\n", False),
    ("cat <(true)\\\n", False),
])
def test_continued_line_comment_boundary(prefix, expected):
    from tools.approval_detection_spans import _is_shell_comment_start

    command = prefix + "# comment"
    assert _is_shell_comment_start(command, len(prefix)) is expected


def test_ambiguous_arithmetic_comment_with_double_quote():
    assert detect_hardline_command('echo $(( id # "\n)\nreboot\n)')[0]

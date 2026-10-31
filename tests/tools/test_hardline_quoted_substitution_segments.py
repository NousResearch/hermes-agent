"""A double-quoted command substitution is its own quote context for top-level segmentation.

In bash, ``"$(...)"`` starts a fresh quote context: ``echo "$(grep -ciE "PG::|x" f)"`` is one
command whose grep pattern is ``PG::|x``. The top-level segmenter used to toggle quote state on
the inner quotes, so the pattern's ``|`` looked like a top-level pipe, grep was cut into a
fragment with an unterminated quote, and the whole read-only line hit the unconditional
"command parser limit or malformed executable payload" block (a reported
``docker logs | grep -ciE "PG::|ConnectionBad|Redis::"`` health one-liner).
"""
import pytest

from tools.approval_detection import detect_dangerous_command, detect_hardline_command

_REPORTED = (
    'ls ~/.config/app/ | grep -i app; curl -s -o /dev/null -w "root=%{http_code}\\n" '
    'http://127.0.0.1:8080/; for c in web scheduler worker; do echo "$c: $(docker logs '
    '--since 2m app-$c-1 2>&1 | grep -ciE "PG::|ConnectionBad|Redis::|refused|closed the '
    'connection")"; done; docker logs --since 1m app-web-1 2>&1 | grep -iE '
    '"PG::|ConnectionBad|Redis::" | tail -3'
)


@pytest.mark.parametrize("command", [
    _REPORTED,
    'echo "$(grep -ciE "a|b" f)"',
    'echo "$(cat f | grep "a|b")"',
    'echo "n=$(docker logs x 2>&1 | grep -c "a;b&c")"',
    'echo "`grep -c "a|b" f`"',
])
def test_quoted_substitution_grep_pattern_is_not_malformed(command):
    assert detect_hardline_command(command) == (False, None)
    assert detect_dangerous_command(command)[0] is False


@pytest.mark.parametrize(("command", "description"), [
    ('echo "$(grep -c "a|b" f; reboot)"', "system shutdown/reboot"),
    ('echo "$(grep -c "a|b" f)"; reboot', "system shutdown/reboot"),
    ('echo "$(rm -rf --no-preserve-root /)"', "recursive delete of root filesystem"),
    ('echo "$(echo "x" && rm -rf ~)"', "recursive delete of home directory"),
    ('echo "$(grep \'unterminated)"', "command parser limit or malformed executable payload"),
])
def test_hardline_commands_inside_or_after_a_quoted_substitution_still_block(command, description):
    assert detect_hardline_command(command) == (True, description)


def test_interpreter_payload_inside_quoted_substitution_still_needs_approval():
    # Fail-closed: the interpreter scan still flags it (description may be the generic one).
    is_dangerous, _, _ = detect_dangerous_command('echo "$(python3 -c "print(1)")"')
    assert is_dangerous
    assert detect_hardline_command('echo "$(python3 -c "print(1)")"') == (False, None)


# Shell -c payloads nested in a quoted substitution next to a grep pattern with a separator. The
# fresh-quote-context segmenter keeps the substitution whole, so the nested `bash` command must be
# tokenized only up to its own boundary (not through the enclosing `)"`), or its payload never
# reaches the hardline floor. Detection-only strings: nothing here is executed.
@pytest.mark.parametrize(("command", "description"), [
    ('echo "$(grep -c "a|b" f; bash -c \'reboot\')"', "system shutdown/reboot"),
    ('echo "$(grep -c "a|b" f; bash -c \'rm -rf /\')"', "recursive delete of root filesystem"),
    ('echo "`grep -c "a|b" f; bash -c \'reboot\'`"', "system shutdown/reboot"),
    ('echo "`grep -c "a|b" f; sh -c \'rm -rf /\'`"', "recursive delete of root filesystem"),
    ('echo "$(grep -c \'a|b\' f && bash -lc "reboot")"', "system shutdown/reboot"),
    ('echo "$(grep -c "a;b" f; zsh -c \'reboot\'\n)"', "system shutdown/reboot"),
    ('x="$(grep -c "a|b" f | bash -c \'reboot\')"', "system shutdown/reboot"),
    ('echo "$(echo "$(grep -c "a|b" f; sh -c \\"reboot\\")")"', "system shutdown/reboot"),
    ('echo "$(echo "x$(grep "a|b" f; bash -c \'rm -rf /\')y")"', "recursive delete of root filesystem"),
    ('echo "$(bash -c \'reboot\')"', "system shutdown/reboot"),
])
def test_shell_payload_in_quoted_substitution_reaches_hardline_floor(command, description):
    assert detect_hardline_command(command) == (True, description)


@pytest.mark.parametrize("command", [
    'echo "$(grep -c "a|b" f; bash -c \'echo "(x)"\')"',
    'echo "`grep -c "a|b" f; bash -c \'echo ok\'`"',
])
def test_benign_shell_payload_in_quoted_substitution_is_not_hardline(command):
    assert detect_hardline_command(command) == (False, None)
    assert detect_dangerous_command(command)[0] is True  # shell -c still needs approval


# Invalid shell (bash -n rejects it): the outer double quote is left open after the balanced inner
# substitution is skipped. The fresh-context reading only applies to balanced input; otherwise the
# grep is reported as malformed, as before the segmenter change.
@pytest.mark.parametrize("command", [
    'echo "$(grep "a|b" f)',
    'echo "`grep "a|b" f`',
    'echo "$(grep -c "a;b" f)" "',
])
def test_unterminated_outer_quote_after_substitution_fails_closed(command):
    assert detect_hardline_command(command) == (True, "command parser limit or malformed executable payload")


# Redirections before `-c`: `&` in `2>&1` / `&>` is part of the operator, not a separator, and the
# redirection (operator + target) is not in the program's argv, so it cannot end option parsing.
_REDIRECTIONS = ["2>&1", "2>/dev/null", "&>/dev/null", ">&2", "2>>log", ">| out", "<<<x", "{fd}>out", "2> /dev/null"]


@pytest.mark.parametrize("redirection", _REDIRECTIONS)
@pytest.mark.parametrize("form", [
    "bash {r} -c 'reboot'",
    'echo "$(grep -c "a|b" f; bash {r} -c \'reboot\')"',
    'echo "`grep -c "a|b" f; bash {r} -c \'reboot\'`"',
    'echo "$(echo "x$(grep "a|b" f; sh {r} -c \'reboot\')y")"',
])
def test_redirection_before_shell_c_reaches_hardline_floor(redirection, form):
    command = form.format(r=redirection)
    assert detect_hardline_command(command) == (True, "system shutdown/reboot")


@pytest.mark.parametrize("command", [
    'echo "$(grep -c "a|b" f; bash 2>&1 -c \'rm -rf /\')"',
    "zsh 2>/dev/null -lc 'rm -rf /'",
])
def test_redirected_shell_root_delete_reaches_hardline_floor(command):
    assert detect_hardline_command(command) == (True, "recursive delete of root filesystem")


@pytest.mark.parametrize("command", [
    # A positional script ends option parsing: `-c` after it is a script argument, not a flag.
    "bash 2>&1 script.sh -c reboot",
    "bash script.sh 2>/dev/null -c reboot",
    # Redirections inside a quoted substitution stay benign for read-only grep.
    'echo "$(grep -c "a|b" f 2>&1)"',
    'echo "`grep -c "a|b" f 2>/dev/null`"',
    'ls 2>&1 | grep -c "a|b"',
])
def test_redirection_controls_are_not_hardline(command):
    assert detect_hardline_command(command) == (False, None)

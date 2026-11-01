"""The hardline floor judges shell quoting on the AUTHOR's text, not on normalized text.

``_normalize_command_for_detection`` strips backslash escapes so ``r\\m`` cannot hide ``rm``. That
is right for pattern matching but wrong for quote tracking: ``\\"`` becomes ``"`` and flips quote
parity. Two consumers paid for it in opposite directions:

* false positive — a shell-valid ``grep -o "[^\\"]*" f`` lexed as unterminated and hit the
  unconditional block (118 of 125 hardline blocks in one week of real agent use, all benign greps);
* bypass — ``cat "f\\"n.txt"; rm -rf --no-preserve-root /`` put the ``; rm`` start "inside" a
  phantom quote, so no command start was marked and the floor let it through.

Direction from #85922 (@Soju06): read quote state from the raw command (only quoted newlines masked).
"""
import pytest

from tools.approval_detection import detect_dangerous_command, detect_hardline_command


@pytest.mark.parametrize("command", [
    'grep -o "[^\\"]*" f',
    r'grep -n "^from\|^__all__\|^    \"" tools/environments/__init__.py | head -15',
    r'egrep -n "alpha\"beta|gamma" input.txt',
    r'grep -v "^./tests/\|def \|\"\"\"" x.py | grep -v "read_only=True"',
    'grep -n "; reboot" f.txt',
    'echo "value is ${HOME}/x"',
])
def test_escaped_quotes_in_a_valid_grep_pattern_are_not_malformed(command):
    assert detect_hardline_command(command) == (False, None)
    assert detect_dangerous_command(command) == (False, None, None)


@pytest.mark.parametrize(("command", "description"), [
    (r'cat "f\"n.txt"; rm -rf --no-preserve-root /', "recursive delete of root filesystem"),
    (r'echo "a\"b"; reboot', "system shutdown/reboot"),
    (r'echo "a\"b" && rm -rf ~', "recursive delete of home directory"),
    (r'echo "a\"b"; rm${IFS}-rf${IFS}/', "recursive delete of root filesystem"),
    (r'grep -n "prefix \"quoted\" suffix" input.txt; reboot', "system shutdown/reboot"),
    ("printf \\\\\nreboot", "system shutdown/reboot"),
    ("grep 'unterminated", "command parser limit or malformed executable payload"),
])
def test_escaped_quote_before_a_hardline_command_does_not_hide_it(command, description):
    assert detect_hardline_command(command) == (True, description)

"""A grep nested inside a command substitution must lex as its own simple command.

The quoted-grep scanner tokenizes from the grep's position to decide which quoted operand is inert data.
It used to read to the end of the segment, so for ``"$(grep … | cut …)"`` it swallowed the enclosing
command's closing quote, called the quoting unbalanced, and the whole command was reported as a hardline
block. Every such report was a false positive (546 in one run); the lexer now stops at the end of the
simple command: an unquoted ``;`` ``|`` ``&`` newline, or the ``)`` / backtick closing its substitution.
"""
import pytest

from tools.approval_detection import _quoted_grep_pattern_spans, _shell_tokens_with_spans, detect_hardline_command


@pytest.mark.parametrize("command", [
    'sed -n "$(grep -n X f | cut -d: -f1),+3p" f',
    'sed -n "$(grep -n \'^### 1.\' d.md | cut -d: -f1),$(grep -n \'^### 3.\' d.md | cut -d: -f1)p" d.md',
    'echo "$(grep -c x f)"',
    'x="$(grep -n foo bar | head -1)"; echo $x',
    'for f in a b; do echo "$f sig=$(grep -cE \'^x\' $f)"; done',
    'echo `grep -c x f`',
])
def test_grep_inside_a_substitution_is_not_malformed(command):
    _spans, malformed = _quoted_grep_pattern_spans(command)
    assert malformed is False
    assert detect_hardline_command(command) == (False, None)


def test_lexer_stops_at_the_end_of_the_simple_command():
    seg = 'sed -n "$(grep -n X f | cut -d: -f1),+3p" f'
    toks = _shell_tokens_with_spans(seg, seg.index("grep"))
    assert [t[0] for t in toks] == ["grep", "-n", "X", "f"]


def test_genuinely_unbalanced_quoting_still_fails_closed():
    assert _shell_tokens_with_spans("grep 'unterminated", 0) is None
    assert detect_hardline_command("grep 'unterminated")[0] is True


def test_hardline_patterns_unchanged():
    for command in ("rm -rf /", 'echo "$(rm -rf /)"', "sudo shutdown -h now"):
        assert detect_hardline_command(command)[0] is True, command

"""A grep nested inside a command substitution must lex as its own simple command.

The quoted-grep scanner tokenizes from the grep's position to decide which quoted operand is inert data.
It used to read to the end of the segment, so for ``"$(grep … | cut …)"`` it swallowed the enclosing
command's closing quote, called the quoting unbalanced, and the whole command was reported as a hardline
block. Every such report was a false positive (546 in one run); the lexer now stops at the end of the
simple command: an unquoted ``;`` ``|`` ``&`` newline, or the ``)`` / backtick closing its substitution.

Bounding the lexer only fixed the tokenizer layer. The segmenter that splits the command into
top-level pieces had the same blind spot from the other direction and is covered by
``TestNestedQuotesInsideASubstitutionStayLocal`` below.
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




def test_genuinely_unbalanced_quoting_still_fails_closed():
    assert _shell_tokens_with_spans("grep 'unterminated", 0) is None
    assert detect_hardline_command("grep 'unterminated")[0] is True


def test_hardline_patterns_unchanged():
    for command in ("rm -rf /", 'echo "$(rm -rf /)"', "sudo shutdown -h now"):
        assert detect_hardline_command(command)[0] is True, command


class TestExecutableSubstitutionBodiesStayExecutable:
    """Bounding the grep lexer removed the old 'malformed' floor; the newline masker must then treat a
    substitution body inside double quotes as CODE, or a newline-separated hardline command hides as an
    operand (independent review witness: blocked on main as 'malformed', approved on the first fix)."""

    @pytest.mark.parametrize("cmd", [
        'echo "$(grep -P \'safe\' /dev/null\nreboot)"',
        'echo "$(grep x f; reboot)"',
        'echo "$(grep x f && reboot)"',
        'echo "$(grep x f | shutdown -h now)"',
        'echo "$(echo $(grep x f)\nreboot)"',
        'echo "`grep x f\nreboot`"',
    ])
    def test_hardline_command_inside_a_quoted_substitution_blocks_as_itself(self, cmd):
        blocked, reason = detect_hardline_command(cmd)
        assert blocked and reason == "system shutdown/reboot"

    def test_public_guard_blocks_without_offering_approval(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        from tools.approval import check_dangerous_command
        calls = []
        result = check_dangerous_command('echo "$(grep -P \'safe\' /dev/null\nreboot)"', "local",
                                         approval_callback=lambda *a, **k: calls.append(a) or False)
        assert result["approved"] is False and "hardline" in result["message"].lower() and calls == []

    @pytest.mark.parametrize("cmd", [
        "grep -e `echo needle` file",            # backtick OPERAND (opened after the grep) is not a closer
        "grep -F 'sudo reboot' notes.md",        # data pattern
        'git commit -m "fix\nsudo reboot handling"',  # quoted data newline
    ])
    def test_benign_shapes_stay_allowed(self, cmd):
        assert detect_hardline_command(cmd) == (False, None)


class TestNestedQuotesInsideASubstitutionStayLocal:
    """Bounding the grep lexer was not enough: the SEGMENTER also had to recognise substitutions.

    ``_iter_top_level_shell_segments`` scanned with the default empty ``subst``, so a ``$(...)``
    inside a double-quoted string was plain chars and the first ``"`` of a quoted operand nested
    inside it closed the OUTER quote. A ``|`` in that operand -- a regex alternation like
    ``"^#|^$"``, or the next pipeline stage inside the substitution -- was then read as a
    top-level separator and cut the segment mid-string. The truncated segment carried an
    unterminated quote, the grep tokenizer failed closed, and a benign command was reported as
    ``malformed executable payload``: an unconditional hardline block, so no approval prompt was
    ever offered and yolo could not bypass it. Four blocked commands in one session had this
    shape; all were read-only diagnostics.
    """

    @pytest.mark.parametrize("cmd", [
        'echo "  n: $(journalctl -u ssh | grep -ciE "Failed|Invalid" | head -1)"',
        'echo "  d: $(diff <(a | grep -vE "^#|^$") <(b | grep -vE "^#|^$"))"',
        'echo "  k: $(ls ~/.ssh/ | grep -cE "^id_(rsa|ed25519)")"',
        'x="$(ps aux | grep -E "nginx|sshd" | wc -l)"; echo "$x"',
    ])
    def test_pipe_in_a_nested_quoted_operand_is_not_malformed(self, cmd):
        _spans, malformed = _quoted_grep_pattern_spans(cmd)
        assert malformed is False
        assert detect_hardline_command(cmd) == (False, None)

    @pytest.mark.parametrize("cmd,reason", [
        ('echo "  x: $(grep -E "a|b" f; reboot)"', "system shutdown/reboot"),
        ('echo "  x: $(grep -E "a|b" f | shutdown -h now)"', "system shutdown/reboot"),
        ('echo "  x: $(grep -E "a|b" f && rm -rf /)"', "recursive delete of root filesystem"),
    ])
    def test_hardline_hidden_behind_a_nested_operand_still_blocks(self, cmd, reason):
        blocked, got = detect_hardline_command(cmd)
        assert blocked and got == reason, (cmd, got)

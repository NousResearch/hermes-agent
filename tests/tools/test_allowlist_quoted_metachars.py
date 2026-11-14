"""Tests for the quote-aware allowlist shell-operator check.

Port of can1357/oh-my-pi#7553: `command_allowlist` glob rules (e.g.
``cargo *``) used to reject any command whose *quoted arguments* contained
shell metacharacters — a cargo benchmark regex filter like
``'^layer3/write/(a|b)$'`` disqualified the whole command even though the
metacharacters are literal to the shell. The matcher is now quote-aware,
while still rejecting genuinely compound commands and quoted payloads that
a ``-c``/``-e``-style option would hand to another interpreter.
"""

import pytest

from tools.approval import _command_matches_permanent_allowlist
from tools.approval_floors import _has_allowlist_shell_operator


class TestHasAllowlistShellOperator:
    # ------------------------------------------------------------------
    # Simple commands stay simple
    # ------------------------------------------------------------------

    def test_plain_command(self):
        assert not _has_allowlist_shell_operator("git status")

    def test_quoted_metacharacters_are_literal(self):
        # The motivating case: cargo bench regex filter (omp issue #7552).
        cmd = (
            "cargo bench --manifest-path layers/layer3/Cargo.toml "
            "--bench standardized_criterion -- "
            "'^layer3/write/file-wal/batch-(10|1000|10000)$'"
        )
        assert not _has_allowlist_shell_operator(cmd)

    def test_double_quoted_literal_metachars(self):
        assert not _has_allowlist_shell_operator('grep -r "a|b;c" src')

    def test_escaped_metachar_is_literal(self):
        assert not _has_allowlist_shell_operator("grep foo\\;bar file.txt")

    def test_unquoted_dollar_variable_is_simple(self):
        # Historical behavior: only `$(` was compound, bare $VAR was not.
        assert not _has_allowlist_shell_operator("echo $HOME")

    def test_unquoted_parens_alone_are_not_compound(self):
        # Parens without $ were never matched by the old regex either.
        assert not _has_allowlist_shell_operator("pytest -k (a and b)")

    # ------------------------------------------------------------------
    # Genuinely compound commands still rejected
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("cmd", [
        "git status; rm -rf /tmp/x",
        "git status && make",
        "git status || make",
        "cat foo | grep bar",
        "echo hi > /etc/passwd",
        "cat < seed",
        "echo `rm x`",
        "echo $(rm x)",
        "git status\nrm x",
        "git status & disown",
    ])
    def test_unquoted_operators_compound(self, cmd):
        assert _has_allowlist_shell_operator(cmd)

    def test_dollar_inside_double_quotes_is_active(self):
        # Expansion still happens inside double quotes.
        assert _has_allowlist_shell_operator('echo "$(rm x)"')
        assert _has_allowlist_shell_operator('echo "`rm x`"')
        assert _has_allowlist_shell_operator('echo "$HOME"')

    def test_unterminated_quote_is_compound(self):
        assert _has_allowlist_shell_operator("echo 'unterminated")

    # ------------------------------------------------------------------
    # Reinterpreted-argument options: quoted payloads become executable
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("cmd", [
        "sh -c 'rm -rf /tmp/x; echo done'",
        'bash -c "make | tee log"',
        "git -c alias.x='!touch /tmp/pwn; printf ok' x",
        'git -c alias.x="!touch /tmp/pwn; printf ok" x',
        "node --eval 'require(\"child_process\").exec(\"id\")>1'",
        "perl -e 'system(\"id\");'",
    ])
    def test_quoted_payload_with_interpreter_option(self, cmd):
        assert _has_allowlist_shell_operator(cmd)

    def test_interpreter_option_without_quoted_metachars_ok(self):
        # -c with a payload containing control chars (parens) is flagged...
        assert _has_allowlist_shell_operator("python -c 'print(1)'")
        # ...but a clean payload with no control characters at all is fine.
        assert not _has_allowlist_shell_operator("python -c 'import sys'")


class TestAllowlistGlobWithQuotedArgs:
    def test_cargo_glob_matches_quoted_regex_filter(self, monkeypatch):
        import tools.approval as mod
        monkeypatch.setattr(mod, "_permanent_approved", {"cargo *"})
        cmd = (
            "cargo bench --bench standardized_criterion -- "
            "'^layer3/write/file-wal/batch-(10|1000|10000)$'"
        )
        assert _command_matches_permanent_allowlist(cmd)

    def test_glob_still_refuses_compound(self, monkeypatch):
        import tools.approval as mod
        monkeypatch.setattr(mod, "_permanent_approved", {"cargo *"})
        assert not _command_matches_permanent_allowlist("cargo build && rm -rf /tmp/x")

    def test_glob_refuses_git_alias_payload(self, monkeypatch):
        import tools.approval as mod
        monkeypatch.setattr(mod, "_permanent_approved", {"git *"})
        assert not _command_matches_permanent_allowlist(
            "git -c alias.x='!touch /tmp/pwn; printf ok' x"
        )


# Issue #145763: bash evaluates these operands as array subscripts, re-expanding a
# single-quoted substitution. Probe strings are assembled at runtime.
_SUB = "$(" + "id" + ")"
_TICK = "`" + "id" + "`"


class TestSubscriptEvaluatingBuiltins:
    @pytest.mark.parametrize("cmd", [
        f"test -v 'a[{_SUB}]'",
        f"test -v 'a[{_TICK}]'",
        f"[ -v 'a[{_SUB}]' ]",
        f"[[ -v 'a[{_SUB}]' ]]",
        f"[[ 1 -eq 'a[{_SUB}]' ]]",
        f"printf -v 'a[{_SUB}]' x",
        f"unset 'a[{_SUB}]'",
        f"read 'a[{_SUB}]'",
        f"declare 'a[{_SUB}]=1'",
        f"local 'a[{_SUB}]=1'",
        f"let 'b=a[{_SUB}]'",
        f"printf -v'a[{_SUB}]' x",
        f"builtin test -v 'a[{_SUB}]'",
        f"LC_ALL=C test -v 'a[{_SUB}]'",
    ])
    def test_quoted_substitution_in_evaluated_operand_is_compound(self, cmd):
        assert _has_allowlist_shell_operator(cmd)

    @pytest.mark.parametrize("cmd", [
        "test -f foo",
        "test -v SHELL",
        "test -v 'a[x]'",
        "[[ -v 'a[1]' ]]",
        f"test -n '{_SUB}'",
        f"printf '%s' '{_SUB}'",
        "cargo bench -- '^a(b|c)$'",
    ])
    def test_literal_operands_stay_simple(self, cmd):
        assert not _has_allowlist_shell_operator(cmd)

    def test_test_glob_refuses_subscript_substitution(self, monkeypatch):
        import tools.approval as mod
        monkeypatch.setattr(mod, "_permanent_approved", {"test *", "[[ *"})
        assert not _command_matches_permanent_allowlist(f"test -v 'a[{_SUB}]'")
        assert not _command_matches_permanent_allowlist(f"[[ -v 'a[{_TICK}]' ]]")
        assert _command_matches_permanent_allowlist("test -f /etc/hostname")

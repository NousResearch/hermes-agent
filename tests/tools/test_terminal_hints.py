"""Tests for tools/terminal_hints.py — output-pattern failure hints."""

from unittest.mock import patch as mock_patch


from tools.terminal_hints import annotate_failure, annotate_masked_success


class TestAnnotateFailureBasics:
    def test_success_never_annotated(self):
        assert annotate_failure("python x.py", 0, "python: command not found") is None

    def test_empty_output_falls_to_exit_code_tier(self):
        assert "126" in annotate_failure("./run.sh", 126, "")
        assert "SIGKILL" in annotate_failure("big_job", 137, "")
        assert "timeout" in annotate_failure("sleep 999", 124, "")

    def test_unknown_failure_returns_none(self):
        assert annotate_failure("./x", 1, "some unrecognized error") is None

    def test_only_first_matching_hint(self):
        out = 'CONFLICT (content): Merge conflict in a.py\npython: command not found'
        hint = annotate_failure("git merge x && python t.py", 1, out)
        assert "conflict" in hint.lower()
        assert "python3" not in hint


class TestGhUnknownJsonField:
    def test_field_name_extracted(self):
        out = 'Unknown JSON field: "authorAssociation"\nAvailable fields:\n  additions\n  author'
        hint = annotate_failure("gh pr view 1 --json authorAssociation", 1, out)
        assert "authorAssociation" in hint


class TestCommandNotFound:
    def test_bare_python_gets_python3_hint(self):
        out = "/usr/bin/bash: line 1: python: command not found"
        hint = annotate_failure("python x.py", 127, out)
        assert "python3" in hint

    def test_bare_pip_gets_pip3_hint(self):
        out = "bash: pip: command not found"
        hint = annotate_failure("pip install x", 127, out)
        assert "pip3" in hint or "-m pip" in hint

    def test_generic_command(self):
        out = "bash: line 3: shellcheck: command not found"
        hint = annotate_failure("shellcheck s.sh", 127, out)
        assert "shellcheck" in hint


class TestModuleNotFound:
    def test_module_named(self):
        out = ("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
               "ModuleNotFoundError: No module named 'requests'")
        hint = annotate_failure("python3 x.py", 1, out)
        assert "requests" in hint

    def test_dotted_module(self):
        out = "ImportError: No module named 'hermes_cli.main'"
        hint = annotate_failure("python3 -m hermes_cli.main", 1, out)
        assert "hermes_cli" in hint


class TestGitShapes:

    def test_branch_already_exists(self):
        out = "fatal: a branch named 'fix/x' already exists"
        hint = annotate_failure("git checkout -b fix/x", 128, out)
        assert "fix/x" in hint

    def test_rate_limit(self):
        out = "GraphQL: API rate limit already exceeded for user ID 1."
        hint = annotate_failure("gh pr list", 1, out)
        assert "rate limit" in hint.lower()


class TestPermissionDenied:
    def test_permission_denied(self):
        hint = annotate_failure("touch /etc/x", 1, "touch: cannot touch '/etc/x': Permission denied")
        assert "Permission denied" in hint


class TestPayloadQuoting:
    """Generated code embedding a natural-language payload whose punctuation
    collides with the source's own quoting (issue #47630)."""

    # The issue's acceptance criterion: a realistic gh-issue-body payload with
    # an em dash, smart quotes and an apostrophe must get the file-handoff hint.
    def test_github_body_payload_names_file_handoff(self):
        out = (
            '  File "post_issue.py", line 1\n'
            '    body = \'Reporting — the “new” field isn\'t saving\'\n'
            "                                                                      ^\n"
            "SyntaxError: unterminated string literal (detected at line 1)\n"
        )
        hint = annotate_failure("python3 post_issue.py", 1, out)
        assert hint is not None
        # Names the file handoff concretely, not a retry of the same source.
        assert "write_file" in hint
        assert "--body-file" in hint
        assert "gh api -F body=@<file>" in hint
        assert "open(path).read()" in hint
        # Product invariant: Hermes must never normalize or rewrite the user's
        # payload, so the hint must not read as permission to edit it. Pinned as
        # the clause that carries the rule, not the whole sentence, so the
        # wording stays free to tighten.
        assert "Leave the payload" in hint
        # Scoping: only a stray character in the generated CODE may be swapped
        # for ASCII; payload characters must never be edited.
        assert "stray in the generated code" in hint

    def test_smart_quote_used_as_delimiter(self):
        out = ('  File "<string>", line 1\n'
               "    x = “hello”\n"
               "        ^ \n"
               "SyntaxError: invalid character '“' (U+201C)")
        hint = annotate_failure("python3 -c 'x = “hello”'", 1, out)
        assert hint is not None
        assert "--body-file" in hint

    def test_non_quote_typographic_character_fires_hint(self):
        # This test exists to block a future "narrow it to quote characters"
        # change: these are typographic punctuation marks that cannot stand in
        # code position — a smart quote or guillemet used as a delimiter, or a
        # middot / em dash / prime / acute standing where an operator or
        # literal belongs — and Python emits the same
        # `invalid character '<ch>' (U+XXXX)` message for every one of them
        # (guillemets, low-9 quotes, fullwidth quote, prime, acute, middot),
        # so the pattern must stay broad. Captured on Python 3.14:
        # a middot in code position (source `x = · 5`) yields exactly this.
        out = ("  File \"<string>\", line 1\n"
               "    x = · 5\n"
               "        ^\n"
               "SyntaxError: invalid character '·' (U+00B7)")
        hint = annotate_failure("python3 -c 'x = · 5'", 1, out)
        assert hint is not None
        assert "--body-file" in hint

    def test_non_printable_character_not_flagged(self):
        # Boundary of the class: invisible characters produce a DIFFERENT
        # captured message — `SyntaxError: invalid non-printable character
        # U+00A0` (NBSP) / `... U+200B` (ZWSP) — a different cause with a
        # different remedy, so the payload-quoting hint must not fire.
        assert annotate_failure("python3 -c 'x = \u00a0 5'", 1,
            "SyntaxError: invalid non-printable character U+00A0") is None
        assert annotate_failure("python3 -c 'x = \u200b5'", 1,
            "SyntaxError: invalid non-printable character U+200B") is None

    def test_shell_unmatched_quote(self):
        out = ("sh: -c: line 0: unexpected EOF while looking for matching `''\n"
               "sh: -c: line 1: syntax error: unexpected end of file")
        hint = annotate_failure("sh -c \"gh issue comment 1 --body 'It isn't broken'\"", 2, out)
        assert hint is not None
        assert "write_file" in hint

    def test_triple_quoted_unterminated_literal(self):
        # Multi-line gh body inside a """...""" literal: 'triple-quoted' is
        # interposed, so the plain-literal pattern alone misses this.
        out = ('  File "post_issue.py", line 1\n'
               '    body = """unterminated\n'
               "             ^\n"
               "SyntaxError: unterminated triple-quoted string literal (detected at line 1)\n")
        hint = annotate_failure("python3 post_issue.py", 1, out)
        assert hint is not None
        assert "write_file" in hint
        assert "--body-file" in hint

    def test_pre_310_unterminated_literal_wording(self):
        # Python 3.9 and earlier word the same two failures differently — both
        # messages captured verbatim on 3.9.6 (`body = 'Reporting` and an
        # unterminated """...""" literal). They are specific to an unterminated
        # literal, so they carry the same remedy; the nested-apostrophe payload
        # on 3.9.6 is the generic `SyntaxError: invalid syntax` that
        # test_unrelated_syntax_errors_not_flagged pins as deliberately unhinted.
        hint = annotate_failure("python3 post_issue.py", 1,
            '  File "post_issue.py", line 1\n'
            "    body = 'Reporting\n"
            "                     ^\n"
            "SyntaxError: EOL while scanning string literal\n")
        assert hint is not None
        assert "--body-file" in hint
        hint = annotate_failure("python3 post_issue.py", 1,
            '  File "post_issue.py", line 3\n'
            '    body = """Reporting — the new field isn\'t saving\n'
            "                                                      ^\n"
            "SyntaxError: EOF while scanning triple-quoted string literal\n")
        assert hint is not None
        assert "write_file" in hint

    def test_zsh_unmatched_quote(self):
        # zsh words the same failure differently from bash/sh.
        out = "zsh:1: unmatched '"
        hint = annotate_failure(
            'zsh -c "gh issue comment 1 --body \'It isn\'t broken\'"', 1, out)
        assert hint is not None
        assert "write_file" in hint

    def test_bare_unmatched_word_not_flagged(self):
        # The zsh pattern must be anchored to the `zsh:` error prefix — an
        # unrelated tool printing 'unmatched' must not fire the hint.
        assert annotate_failure("grep foo bar.txt", 1,
                                "grep: unmatched something") is None
        assert annotate_failure("sed -e 's/[a-'", 1,
                                "sed: unmatched brace") is None

    def test_zsh_pattern_left_boundary(self):
        # Boundary on the `zsh:` prefix itself: the message may follow a
        # prefix on the same line (`docker: zsh:1: ...`), but a longer token
        # merely ENDING in 'zsh' is a different program and must not fire.
        assert annotate_failure("docker run --rm zsh -c 'x'", 1,
                                "docker: zsh:1: unmatched '") is not None
        assert annotate_failure("myzsh run.zsh", 1,
                                "myzsh:1: unmatched '") is None
        assert annotate_failure("./note_zsh.sh", 1,
                                "note_zsh:1: unmatched '") is None

    def test_unrelated_syntax_errors_not_flagged(self):
        # Scope guard for the deliberately broad `invalid character` rule:
        # real Python compile errors of unrelated cause share the
        # `SyntaxError:` prefix but are not payload-quoting collisions. If
        # the pattern were ever broadened to a bare `SyntaxError`, these fail.
        # `invalid syntax` is also what Python 3.9.6 reports for a payload
        # apostrophe that leaves a stray token behind (`body = 'It isn't
        # broken'`), so this boundary is the one genuine gap the 3.9-and-
        # earlier wording leaves open.
        assert annotate_failure("python3 x.py", 1,
            '  File "x.py", line 1\n    if x\n       ^\n'
            "SyntaxError: expected ':'") is None
        assert annotate_failure("python3 x.py", 1,
            '  File "x.py", line 1\n    def\n       ^^^\n'
            "SyntaxError: invalid syntax") is None

    def test_unrelated_python_failure_not_flagged(self):
        out = ('Traceback (most recent call last):\n  File "x.py", line 2, in <module>\n'
               "KeyError: 'body'")
        assert annotate_failure("python3 x.py", 1, out) is None

    def test_unrelated_shell_failure_not_flagged(self):
        assert annotate_failure("ls /no/such/dir", 1,
                                "ls: /no/such/dir: No such file or directory") is None


class TestBoundedScan:
    def test_pattern_beyond_scan_window_ignored(self):
        out = "x" * 5000 + "\npython: command not found"
        assert annotate_failure("noop", 1, out) is None

    def test_hint_functions_cannot_crash_annotate(self):
        # A hint raising must not propagate.
        with mock_patch("tools.terminal_hints._OUTPUT_HINTS", [lambda c, o: 1 / 0]):
            assert annotate_failure("x", 1, "boom") is None




class TestMaskedSuccess:
    """annotate_masked_success: exit-0 pipelines that hide real failures."""

    def _fail_out(self):
        return (
            "error[E0308]: mismatched types\n"
            "error: could not compile `llama_manager` due to 11 previous errors\n"
        )

    def test_cargo_pipe_tail_flagged(self):
        hint = annotate_masked_success(
            "cargo build --release 2>&1 | tail -20", self._fail_out()
        )
        assert hint is not None


    def test_or_echo_fallback_flagged(self):
        hint = annotate_masked_success(
            'cargo build || echo "BUILD FAILED"',
            "error: could not compile `x`\nBUILD FAILED\n",
        )
        assert hint and "||" in hint

    def test_or_true_flagged(self):
        hint = annotate_masked_success(
            "pytest tests/ || true",
            "FAILED tests/test_x.py::test_y - AssertionError\n= 3 failed in 1.2s\n",
        )
        assert hint is not None

    def test_bare_command_not_flagged(self):
        # No pipe, no ||: exit 0 with error-looking output is not our call.
        assert annotate_masked_success("cargo build", self._fail_out()) is None

    def test_clean_output_pipe_not_flagged(self):
        assert (
            annotate_masked_success(
                "cargo build 2>&1 | tail -20",
                "   Compiling llama_manager v0.1.0\n    Finished release\n",
            )
            is None
        )

    def test_grep_pipeline_excluded(self):
        # Search output legitimately CONTAINS failure text; grep|head exit 0
        # is a real success (matches found).
        assert (
            annotate_masked_success(
                "grep -rn 'error\\[E' build.log | head -20",
                "build.log:12:error[E0308]: mismatched types\n",
            )
            is None
        )

    def test_rg_pipeline_excluded(self):
        assert (
            annotate_masked_success(
                "rg 'could not compile' logs/ | tail -5",
                "logs/b.txt:error: could not compile `x`\n",
            )
            is None
        )

    def test_pipe_into_grep_not_flagged(self):
        # grep as CONSUMER filters and its exit status is meaningful
        # (0 = matched) — not a passthrough truncation consumer.
        assert (
            annotate_masked_success(
                "cargo build 2>&1 | grep -c error",
                "error[E0308]: mismatched types\n",
            )
            is None
        )

    def test_or_operator_alone_not_confused_with_pipe(self):
        # `a || b` must not match the single-pipe regex.
        assert (
            annotate_masked_success(
                "test -f x || stat y",
                "error[E0308]\n",  # would need a masking shape too
            )
            is None
        )

    def test_generic_error_word_not_enough(self):
        # Weak signal ("error" substring) must not fire the note.
        assert (
            annotate_masked_success(
                "make install 2>&1 | tail -30",
                "checking error handling support... yes\nInstall complete.\n",
            )
            is None
        )

    def test_pytest_summary_via_tee(self):
        hint = annotate_masked_success(
            "pytest tests/ | tee /tmp/out.log",
            "FAILED tests/a.py::test_b\n=========== 2 failed, 10 passed ===========\n",
        )
        assert hint is not None

    def test_empty_inputs(self):
        assert annotate_masked_success("", "") is None
        assert annotate_masked_success("cargo build | tail", "") is None
        assert annotate_masked_success("", "error[E0308]") is None

    def test_echo_printf_heads_excluded(self):
        assert (
            annotate_masked_success(
                "printf 'a.rs:1:error[E0308]: x\\n' | cat | tail -3",
                "a.rs:1:error[E0308]: x\n",
            )
            is None
        )
        assert (
            annotate_masked_success(
                'echo "error: could not compile x" | tee log.txt',
                "error: could not compile x\n",
            )
            is None
        )

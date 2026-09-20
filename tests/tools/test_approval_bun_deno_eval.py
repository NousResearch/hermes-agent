"""Bun and Deno evaluate inline scripts through forms the interpreter flag
tables did not know: ``bun -e/--eval``, and Deno's bare ``eval`` subcommand
(``deno eval "code"``). They must reach the same approval classification as
``node -e`` / ``python -c``. See issue #116012.
"""

import pytest

from tools.approval import detect_dangerous_command


class TestBunDenoInlineScriptExecution:
    @pytest.mark.parametrize(
        "cmd, expected_key",
        [
            ('bun -e "console.log(1)"', "script execution via -e/-c flag"),
            ('bun --eval "console.log(1)"', "script execution via -e/-c flag"),
            ('deno eval "console.log(1)"', "script execution via -e/-c flag"),
            ('deno -e "console.log(1)"', "script execution via -e/-c flag"),
            ('deno --eval "console.log(1)"', "script execution via -e/-c flag"),
            # Command-position wrappers must not shield the interpreter.
            ('sudo bun -e "console.log(1)"', "script execution via -e/-c flag"),
            ('sudo deno eval "console.log(1)"', "script execution via -e/-c flag"),
            # Windows resolves executable names case-insensitively; `_interpreter_family`
            # lowercases the basename, so BUN.EXE / Deno.exe must be classified too.
            ('BUN.EXE -e "console.log(1)"', "script execution via -e/-c flag"),
            ('Deno.exe eval "console.log(1)"', "script execution via -e/-c flag"),
            ('bun << "EOF"\nconsole.log("pwned")\nEOF', "script execution via heredoc"),
            ("deno << 'EOF'\nconsole.log(\"pwned\")\nEOF", "script execution via heredoc"),
            # An unterminated quote fails closed like the other interpreter families.
            ('bun -e "console.log(1)', "command parser limit or malformed executable payload"),
        ],
    )
    def test_inline_eval_classified_like_node_and_python(self, cmd, expected_key):
        dangerous, key, _ = detect_dangerous_command(cmd)
        assert dangerous is True, cmd
        assert key == expected_key, cmd

    @pytest.mark.parametrize(
        "cmd",
        [
            "bun install",
            "bun run build",
            "deno run server.ts",
            "deno test",
            "deno lint",
            # A file literally named eval.ts is a `run` operand, not the eval subcommand.
            "deno run eval.ts",
            # `deno eval` inside quotes is data for echo, not a command position.
            "echo 'use deno eval for that'",
        ],
    )
    def test_everyday_bun_deno_invocations_stay_unflagged(self, cmd):
        assert detect_dangerous_command(cmd) == (False, None, None), cmd

"""Bun and Deno evaluate inline scripts through forms the interpreter flag
tables did not know: ``bun -e/--eval``, and Deno's bare ``eval`` subcommand
(``deno eval "code"``). They must reach the same approval classification as
``node -e`` / ``python -c``. See issue #116012.
"""

from tools.approval import detect_dangerous_command


class TestBunDenoInlineScriptExecution:
    def test_inline_eval_detected(self):
        for cmd in (
            'bun -e "console.log(1)"',
            'bun --eval "console.log(1)"',
            'deno eval "console.log(1)"',
            'deno -e "console.log(1)"',
            'deno --eval "console.log(1)"',
            # Command-position wrappers must not shield the interpreter.
            'sudo bun -e "console.log(1)"',
            'sudo deno eval "console.log(1)"',
        ):
            dangerous, key, _ = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert key == "script execution via -e/-c flag", cmd

    def test_windows_executable_spelling_detected(self):
        # Windows resolves executable names case-insensitively; `_interpreter_family`
        # lowercases the basename, so BUN.EXE / Deno.exe must be classified too.
        for cmd in (
            'BUN.EXE -e "console.log(1)"',
            'Deno.exe eval "console.log(1)"',
        ):
            dangerous, key, _ = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert key == "script execution via -e/-c flag", cmd

    def test_interpreter_heredoc_detected(self):
        for cmd in (
            'bun << "EOF"\nconsole.log("pwned")\nEOF',
            "deno << 'EOF'\nconsole.log(\"pwned\")\nEOF",
        ):
            dangerous, key, _ = detect_dangerous_command(cmd)
            assert dangerous is True, cmd
            assert key == "script execution via heredoc", cmd

    def test_malformed_quoting_fails_closed(self):
        # An unterminated quote on a bun/deno inline script must fail closed like the
        # other interpreter families, not silently skip the family.
        dangerous, key, _ = detect_dangerous_command('bun -e "console.log(1)')
        assert dangerous is True
        assert key == "command parser limit or malformed executable payload"

    def test_plain_bun_deno_invocations_not_flagged(self):
        """Everyday subcommands that run project files must stay safe."""
        for cmd in (
            "bun install",
            "bun run build",
            "deno run server.ts",
            "deno test",
            "deno lint",
            # A file literally named eval.ts is a `run` operand, not the eval subcommand.
            "deno run eval.ts",
        ):
            dangerous, _, _ = detect_dangerous_command(cmd)
            assert dangerous is False, cmd

    def test_quoted_prose_about_deno_eval_not_flagged(self):
        # `deno eval` inside quotes is data for echo, not a command position.
        dangerous, _, _ = detect_dangerous_command("echo 'use deno eval for that'")
        assert dangerous is False

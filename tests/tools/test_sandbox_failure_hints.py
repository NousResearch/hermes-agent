"""Tests for execute_code sandbox failure hints."""

import json


from tools.code_execution_tool import _sandbox_failure_hint, execute_code
from tools.terminal_hints import PAYLOAD_QUOTING_HINT


class TestSandboxFailureHint:
    def test_unavailable_tool_import_lists_available(self):
        err = ("Traceback (most recent call last):\n  File \"script.py\", line 1\n"
               "ImportError: cannot import name 'browser_navigate' from 'hermes_tools'")
        h = _sandbox_failure_hint(err, enabled_tools={"terminal", "read_file"})
        assert "browser_navigate" in h
        assert "read_file" in h and "terminal" in h




    def test_unknown_failure_no_hint(self):
        assert _sandbox_failure_hint("ZeroDivisionError: division by zero") is None

    def test_empty_stderr_no_hint(self):
        assert _sandbox_failure_hint("") is None


class TestLiveSandboxHint:
    def test_bad_import_produces_hint_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = json.loads(execute_code(
            "from hermes_tools import totally_fake_tool\nprint('unreachable')",
            task_id="t-sbhint",
        ))
        assert r["status"] == "error"
        assert "hint" in r
        assert "totally_fake_tool" in r["hint"]


    def test_successful_script_has_no_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = json.loads(execute_code("print('fine')", task_id="t-sbhint"))
        assert r["status"] == "success"
        assert "hint" not in r


class TestPayloadQuotingHint:
    """The execute_code egress must report the SAME hint as the terminal egress —
    assert against the shared PAYLOAD_QUOTING_HINT object, never duplicated prose."""

    def test_gh_issue_body_payload_breaks_literal(self):
        # Realistic: a generated gh issue-creation script whose --body payload
        # carries an em dash, smart quotes, and an apostrophe; the apostrophe
        # closes the single-quoted shell arg and the tail reparses as code.
        err = (
            '  File "<execute_code>", line 4, in <module>\n'
            "    subprocess.run(['gh', 'issue', 'create', '--title', 'Bug', '--body',\n"
            "      'The user\\'s “repro” steps — which we can’t reproduce — fail on macOS.'])\n"
            "      ^\n"
            "SyntaxError: unterminated string literal (detected at line 4)"
        )
        assert _sandbox_failure_hint(err) == PAYLOAD_QUOTING_HINT

    def test_smart_quote_used_as_delimiter(self):
        # Payload's typographic quote lands where Python expects a delimiter.
        err = (
            '  File "<execute_code>", line 2\n'
            "    msg = “don’t let smart quotes delimit this”\n"
            "          ^\n"
            "SyntaxError: invalid character '“' (U+201C)"
        )
        assert _sandbox_failure_hint(err) == PAYLOAD_QUOTING_HINT

    def test_shell_unmatched_quote(self):
        # The payload's apostrophe leaves the shell wrapper quote-unbalanced.
        err = (
            "bash: line 1: unexpected EOF while looking for matching `''\n"
            "bash: line 2: syntax error: unexpected end of file\n"
        )
        assert _sandbox_failure_hint(err) == PAYLOAD_QUOTING_HINT

    def test_hint_is_shared_not_duplicated(self):
        # Same object identity through both import paths => one source of truth.
        from tools import code_execution_tool as cet
        assert cet.PAYLOAD_QUOTING_HINT is PAYLOAD_QUOTING_HINT
        assert _sandbox_failure_hint("SyntaxError: invalid character '“' (U+201C)") is PAYLOAD_QUOTING_HINT

    def test_unrelated_traceback_still_unhinted(self):
        err = ("Traceback (most recent call last):\n"
               '  File "<execute_code>", line 1, in <module>\n'
               "ZeroDivisionError: division by zero")
        assert _sandbox_failure_hint(err) is None

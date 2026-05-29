"""Tests for the execute_code single-tool-call gate.

Motivation: t_4ba269e5 retro found 120/154 execute_code calls were
single-terminal wrappers with no processing logic — a 2× cost amplifier.
The gate rejects these and guides the caller to use the direct tool.

Acceptance criteria (from task body):
1. 1-tool body rejected with guidance message.
2. 3-tool body accepted (gate passes, code would run normally).
3. 1-tool body WITH substantive non-tool logic accepted.
"""

from __future__ import annotations

import pytest

from tools.code_execution_tool import (
    _check_single_tool_call,
    _count_hermes_tool_calls,
    _has_nontrivial_logic,
)

# ---------------------------------------------------------------------------
# _count_hermes_tool_calls
# ---------------------------------------------------------------------------


def test_count_zero_for_pure_python():
    code = "x = 1 + 2\nprint(x)"
    assert _count_hermes_tool_calls(code) == 0


def test_count_one_terminal():
    code = (
        "from hermes_tools import terminal\nresult = terminal('ls -la')\nprint(result)"
    )
    assert _count_hermes_tool_calls(code) == 1


def test_count_one_read_file():
    code = "from hermes_tools import read_file\nr = read_file('/tmp/foo')\nprint(r)"
    assert _count_hermes_tool_calls(code) == 1


def test_count_three_tools():
    code = (
        "from hermes_tools import terminal, read_file, write_file\n"
        "out = terminal('git log')\n"
        "r = read_file('/tmp/a')\n"
        "write_file('/tmp/b', r['content'])\n"
    )
    assert _count_hermes_tool_calls(code) == 3


def test_count_module_attribute_form():
    """hermes_tools.terminal(...) should also count."""
    code = "import hermes_tools\nhermes_tools.terminal('echo hi')"
    assert _count_hermes_tool_calls(code) == 1


def test_count_raises_on_syntax_error():
    with pytest.raises(SyntaxError):
        _count_hermes_tool_calls("def broken(:")


# ---------------------------------------------------------------------------
# _has_nontrivial_logic
# ---------------------------------------------------------------------------


def test_nontrivial_for_loop():
    code = "for i in range(10):\n    print(i)"
    assert _has_nontrivial_logic(code) is True


def test_nontrivial_while_loop():
    code = "while True:\n    break"
    assert _has_nontrivial_logic(code) is True


def test_nontrivial_list_comprehension():
    code = "x = [i for i in range(5)]"
    assert _has_nontrivial_logic(code) is True


def test_nontrivial_try_except():
    code = "try:\n    pass\nexcept Exception:\n    pass"
    assert _has_nontrivial_logic(code) is True


def test_nontrivial_json_parse():
    code = "import json\ndata = json.loads('{}')"
    assert _has_nontrivial_logic(code) is True


def test_nontrivial_re_module():
    code = "import re\nm = re.match(r'foo', 'foobar')"
    assert _has_nontrivial_logic(code) is True


def test_nontrivial_sorted_call():
    code = "xs = sorted([3, 1, 2])"
    assert _has_nontrivial_logic(code) is True


def test_trivial_single_assignment():
    code = "x = 1"
    assert _has_nontrivial_logic(code) is False


def test_trivial_simple_print():
    code = "print('hello')"
    assert _has_nontrivial_logic(code) is False


def test_trivial_one_terminal_call():
    code = "from hermes_tools import terminal\nresult = terminal('ls')\nprint(result)"
    assert _has_nontrivial_logic(code) is False


# ---------------------------------------------------------------------------
# _check_single_tool_call (the gate itself)
# ---------------------------------------------------------------------------


class TestCheckSingleToolCallGate:
    """Acceptance-criteria tests for the execute_code gate."""

    # AC 1: 1-tool body rejected with guidance
    def test_single_terminal_rejected(self):
        code = "from hermes_tools import terminal\nresult = terminal('grep foo bar')\nprint(result)"
        msg = _check_single_tool_call(code)
        assert msg is not None
        assert "terminal()" in msg
        assert "directly" in msg

    def test_single_read_file_rejected(self):
        code = "from hermes_tools import read_file\nr = read_file('/tmp/a')\nprint(r)"
        msg = _check_single_tool_call(code)
        assert msg is not None
        assert "read_file()" in msg

    def test_single_search_files_rejected(self):
        code = "from hermes_tools import search_files\nresult = search_files('pattern')\nprint(result)"
        msg = _check_single_tool_call(code)
        assert msg is not None
        assert "search_files()" in msg

    # AC 2: 3-tool body accepted (gate returns None)
    def test_three_tool_body_accepted(self):
        code = (
            "from hermes_tools import terminal, read_file, write_file\n"
            "out = terminal('git log --oneline -10')\n"
            "r = read_file('/tmp/notes.txt')\n"
            "write_file('/tmp/out.txt', out['output'] + r['content'])\n"
        )
        assert _check_single_tool_call(code) is None

    def test_two_tool_body_accepted(self):
        code = (
            "from hermes_tools import terminal, read_file\n"
            "out = terminal('cat /etc/hosts')\n"
            "r = read_file('/tmp/a')\n"
        )
        assert _check_single_tool_call(code) is None

    # AC 3: 1-tool body WITH substantive non-tool logic accepted
    def test_single_tool_with_loop_accepted(self):
        code = (
            "from hermes_tools import terminal\n"
            "results = []\n"
            "for path in ['/tmp/a', '/tmp/b']:\n"
            "    results.append(terminal(f'ls {path}'))\n"
            "print(results)"
        )
        assert _check_single_tool_call(code) is None

    def test_single_tool_with_regex_accepted(self):
        code = (
            "import re\n"
            "from hermes_tools import terminal\n"
            "out = terminal('cat /etc/hosts')\n"
            "ips = re.findall(r'\\d+\\.\\d+\\.\\d+\\.\\d+', out['output'])\n"
            "print(ips)"
        )
        assert _check_single_tool_call(code) is None

    def test_single_tool_with_json_parsing_accepted(self):
        code = (
            "import json\n"
            "from hermes_tools import terminal\n"
            "out = terminal('cat /tmp/data.json')\n"
            "data = json.loads(out['output'])\n"
            "print(data['key'])"
        )
        assert _check_single_tool_call(code) is None

    def test_single_tool_with_try_except_accepted(self):
        code = (
            "from hermes_tools import read_file\n"
            "try:\n"
            "    r = read_file('/tmp/maybe')\n"
            "    print(r['content'])\n"
            "except Exception as e:\n"
            "    print(f'not found: {e}')\n"
        )
        assert _check_single_tool_call(code) is None

    # Pure Python (no tool calls) — always passes
    def test_pure_python_no_tool_calls_accepted(self):
        code = "x = 2 + 2\nprint(x)"
        assert _check_single_tool_call(code) is None

    # Syntax error → passes (let execution handle it)
    def test_syntax_error_in_code_passes(self):
        code = "def broken(:"
        assert _check_single_tool_call(code) is None

    def test_single_tool_with_list_comprehension_accepted(self):
        code = (
            "from hermes_tools import search_files\n"
            "matches = search_files('TODO', path='/src')\n"
            "paths = [m['path'] for m in matches['matches']]\n"
            "print(paths)"
        )
        assert _check_single_tool_call(code) is None


# ---------------------------------------------------------------------------
# Integration: execute_code function returns guidance string on gate fire
# ---------------------------------------------------------------------------


def test_execute_code_returns_guidance_on_single_tool(monkeypatch):
    """execute_code should return a tool_error JSON when the gate fires."""
    import json

    from tools.code_execution_tool import execute_code

    # Make sandbox available
    monkeypatch.setattr("tools.code_execution_tool.SANDBOX_AVAILABLE", True)

    # Single-tool script — the gate fires before _get_env_config is reached.
    code = "from hermes_tools import terminal\nresult = terminal('ls')\nprint(result)"

    result_str = execute_code(code=code)
    result = json.loads(result_str)
    assert "error" in result
    assert "terminal()" in result["error"]
    assert "directly" in result["error"]

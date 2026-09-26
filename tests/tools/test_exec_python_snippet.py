"""_exec_python_snippet must not emit exec(base64.b64decode(...)) (#122463).

EDR heuristics flag the python -c "import base64;
exec(base64.b64decode('...'))" shape as commodity malware (Metasploit), so
the snippet payload must travel out of band (stdin) instead of embedded as
an exec-of-decode one-liner. The live-equivalence test pins the behavior the
old base64 wrapping protected: backslashes, byte literals, and non-ASCII
must execute byte-identical through the new transport.
"""

import sys
from unittest.mock import MagicMock

from tests.tools.test_file_operations import make_real_subprocess_env
from tools.file_operations import ShellFileOperations

BS = chr(92)  # backslash, built without escape runs so the source stays readable


def _mock_ops():
    env = MagicMock()
    env.cwd = "/tmp/test"
    env.execute.return_value = {"output": "", "returncode": 0}
    return ShellFileOperations(env), env


class TestSnippetEmissionShape:
    def test_command_has_no_exec_b64decode(self):
        ops, env = _mock_ops()
        ops._exec_python_snippet("print('hi')")
        (command,), _kwargs = env.execute.call_args.args, env.execute.call_args.kwargs
        assert "b64decode" not in command
        assert "exec(" not in command

    def test_payload_travels_on_stdin(self):
        ops, env = _mock_ops()
        snippet = "import sys\nprint('hi')\n"
        ops._exec_python_snippet(snippet)
        kwargs = env.execute.call_args.kwargs
        assert kwargs.get("stdin_data") == snippet

    def test_explicit_interpreter_also_pipes_stdin(self):
        ops, env = _mock_ops()
        ops._exec_python_snippet("print(1)", py="python")
        (command,), kwargs = env.execute.call_args.args, env.execute.call_args.kwargs
        assert command.startswith("python ")
        assert kwargs.get("stdin_data") == "print(1)"


class TestSnippetLiveEquivalence:
    def test_backslash_and_unicode_payload_runs_unmangled(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        snippet = (
            "p = 'C:" + BS + BS + "Users" + BS + BS + "x'\n"
            "print(p)\n"
            "print(b'" + BS + "xfe" + BS + "xff'.hex())\n"
            "print('h\u00e9llo \u2713')\n"
        )
        result = ops._exec_python_snippet(snippet, py=sys.executable)
        assert result.exit_code == 0
        assert result.stdout.splitlines() == [
            "C:" + BS + "Users" + BS + "x",
            "feff",
            "h\u00e9llo \u2713",
        ]

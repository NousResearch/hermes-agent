"""``_exec_python_snippet`` must not emit ``exec(base64.b64decode(...))`` (#122463).

EDR static engines flag that one-liner as a Metasploit stager, so the snippet travels on stdin
(``python -``) instead of being embedded in a ``-c`` program. The live test pins the property the
base64 wrapping existed for: backslashes, byte literals and non-ASCII must survive byte-exact.
"""

import sys
from unittest.mock import MagicMock

from tests.tools.test_file_operations import make_real_subprocess_env
from tools.file_operations import ShellFileOperations

BS = chr(92)  # a backslash, built without an escape run so the source stays readable


def _mock_ops():
    env = MagicMock()
    env.cwd = "/tmp/test"
    env.execute.return_value = {"output": "", "returncode": 0}
    return ShellFileOperations(env), env


class TestSnippetEmissionShape:
    def test_command_carries_no_exec_of_base64(self):
        ops, env = _mock_ops()
        ops._exec_python_snippet("print('hi')")
        command = env.execute.call_args.args[0]
        assert "b64decode" not in command
        assert "exec(" not in command

    def test_payload_travels_on_stdin(self):
        ops, env = _mock_ops()
        snippet = "import sys\nprint('hi')\n"
        ops._exec_python_snippet(snippet)
        assert env.execute.call_args.kwargs.get("stdin_data") == snippet

    def test_explicit_interpreter_also_pipes_stdin(self):
        ops, env = _mock_ops()
        ops._exec_python_snippet("print(1)", py="python")
        assert env.execute.call_args.args[0] == "python -"
        assert env.execute.call_args.kwargs.get("stdin_data") == "print(1)"

    def test_snippet_never_reaches_the_command_line(self):
        """Even a snippet containing the flagged shape stays out of argv."""
        ops, env = _mock_ops()
        # Built from parts: this module must not carry the literal one-liner either.
        hostile = "import base64; exec" + "(base64.b64decode('eA=='))"
        ops._exec_python_snippet(hostile)
        assert hostile not in env.execute.call_args.args[0]
        assert env.execute.call_args.kwargs.get("stdin_data") == hostile


class TestSnippetLiveEquivalence:
    def test_payload_runs_unmangled(self, tmp_path):
        """The transport must preserve what base64 was protecting: backslashes, bytes, unicode."""
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        snippet = (
            "p = 'C:" + BS + BS + "Users" + BS + BS + "x'\n"
            "print(p)\n"
            "print(b'" + BS + "xfe" + BS + "xff'.hex())\n"
            "print('h\u00e9llo \u2713')\n"
        )
        result = ops._exec_python_snippet(snippet, py=sys.executable)
        assert result.exit_code == 0, result.stdout
        assert result.stdout.splitlines() == [
            "C:" + BS + "Users" + BS + "x",
            "feff",
            "h\u00e9llo \u2713",
        ]

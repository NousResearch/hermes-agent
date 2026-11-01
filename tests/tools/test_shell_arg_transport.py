"""Non-path values survive local argv and serialized POSIX-script transport."""

import json
import os
import subprocess

import pytest

from tools.environments.local import LocalEnvironment, _find_bash
from tools.file_operations import ShellFileOperations


class _GitBashArgvParser:
    """Model of how Git Bash hands a Windows command line to a native tool.

    Inside the double quotes ``subprocess.list2cmdline`` wraps the ``-c``
    payload in, the MSYS argv parser folds backslashes bash-style: ``\\\\``
    collapses to ``\\`` and ``\\"`` is a literal quote, other ``\\x`` stay
    verbatim. This is the asymmetry the ``translate_path=False``
    compensation exists for (bd258480d3c observed the collapse; #142111
    measured that quote-adjacent runs survive it untouched).
    """

    @staticmethod
    def parse(cmdline):
        argv, cur, i, in_q, started = [], [], 0, False, False
        while i < len(cmdline):
            if not started:
                started, in_q = True, False
            c = cmdline[i]
            if c == '"':
                in_q = not in_q
                i += 1
            elif c == "\\" and in_q:
                nxt = cmdline[i + 1] if i + 1 < len(cmdline) else ""
                if nxt == '"':
                    cur.append('"')
                    i += 2
                elif nxt == "\\":
                    cur.append("\\")
                    i += 2
                else:
                    cur.append(c)
                    i += 1
            elif c in " \t" and not in_q:
                argv.append("".join(cur))
                cur, started = [], False
                i += 1
            else:
                cur.append(c)
                i += 1
        argv.append("".join(cur))
        return argv


def _bash_word_split(command_text):
    """Split a bash command into words: single quotes literal (with the
    ``'"'"'`` splice), double quotes with ``\\\\``/``\\"`` unescaped."""
    args, cur, i, mode = [], [], 0, None
    while i < len(command_text):
        c = command_text[i]
        if mode is None:
            if c == "'":
                mode = "'"
                i += 1
            elif c == '"':
                mode = '"'
                i += 1
            elif c == " ":
                args.append("".join(cur))
                cur = []
                i += 1
            else:
                cur.append(c)
                i += 1
        elif mode == "'":
            if c == "'":
                if command_text[i : i + 5] == "'\"'\"'":
                    cur.append("'")
                    i += 5
                else:
                    mode = None
                    i += 1
            else:
                cur.append(c)
                i += 1
        else:
            if c == '"':
                mode = None
                i += 1
            elif (
                c == "\\" and i + 1 < len(command_text) and command_text[i + 1] in '"\\'
            ):
                cur.append(command_text[i + 1])
                i += 2
            else:
                cur.append(c)
                i += 1
    args.append("".join(cur))
    return args


class _WindowsLocalEnv:
    is_local = True
    cwd = "C:/tmp/x"


def _windows_ops(monkeypatch):
    import tools.environments.local as local_mod

    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    return ShellFileOperations(_WindowsLocalEnv())


def _windows_argv_word(ops, pattern, word_index):
    """Full local-Windows transport, simulated: compensation → single-quoted
    shell word → list2cmdline → Git Bash argv parse → bash word split."""
    quoted = ops._escape_shell_arg(pattern, translate_path=False)
    cmdline = subprocess.list2cmdline(
        ["bash", "--noprofile", "--norc", "-c", f"rg {quoted} C:/src"]
    )
    argv = _GitBashArgvParser.parse(cmdline)
    command_text = argv[argv.index("-c") + 1]
    return _bash_word_split(command_text)[word_index]


def test_windows_compensation_doubles_only_runs_away_from_quotes(monkeypatch):
    """Backslash runs NOT ending at a double quote are doubled (Git Bash
    folds them); a run ending at one rides list2cmdline's MSVCRT escaping
    verbatim instead (#142111: doubling it made `x\\"y` match `x\\` + `y`)."""
    ops = _windows_ops(monkeypatch)
    assert ops._escape_shell_arg("a\\b", translate_path=False) == "'a\\\\b'"
    assert ops._escape_shell_arg("a\\", translate_path=False) == "'a\\\\'"
    assert ops._escape_shell_arg('x\\"y', translate_path=False) == "'x\\\"y'"
    assert ops._escape_shell_arg('x\\\\"y', translate_path=False) == "'x\\\\\"y'"


@pytest.mark.parametrize(
    "pattern",
    [
        'x"y',
        'x\\"y',
        'x\\\\"y',
        "a\\b",
        "a\\\\b",
        "a\\\\\\\\b",
        "a\\",
        '\\"quoted\\"',
        '(?x)x\\"y\\n',
        "it's x\\\"y",
    ],
)
def test_windows_transport_roundtrips_pattern_bytes(monkeypatch, pattern):
    """The pattern rg compiles must be byte-identical to what the user wrote.

    Simulates the whole local-Windows chain on any host: the real
    ``subprocess.list2cmdline`` serializer plus a model of Git Bash's argv
    parse. Red on main's blanket doubling for every quote-adjacent run
    (the #142111 silent zero-match), green with run-aware compensation.
    """
    ops = _windows_ops(monkeypatch)
    assert _windows_argv_word(ops, pattern, word_index=1) == pattern


class _SerializedShell:
    is_local = False

    def __init__(self, bash, cwd, env):
        self.bash, self.cwd, self.env = bash, str(cwd), env

    def execute(self, command, *, cwd=None, timeout=30, **kwargs):
        # Deliver command text on stdin, without Windows command-line parsing.
        script = json.loads(json.dumps({'command': command}))['command']
        result = subprocess.run([self.bash, '--noprofile', '--norc', '-s'],
                                input=script, cwd=cwd or self.cwd, env=self.env,
                                capture_output=True, text=True, encoding='utf-8', timeout=timeout, check=False)
        return {'output': result.stdout, 'returncode': result.returncode}


@pytest.mark.parametrize('serialized', [False, True])
def test_non_path_bytes_survive_real_shell_transport(tmp_path, monkeypatch, serialized):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'store'))
    monkeypatch.setenv('HOME', str(home))
    env = {**os.environ, 'BASH_ENV': '', 'ENV': ''}
    bash = _find_bash()
    backend = (_SerializedShell(bash, tmp_path, env) if serialized
               else LocalEnvironment(cwd=str(tmp_path), timeout=30, env=env))
    ops = ShellFileOperations(backend)
    try:
        for value in ('', 'two words', "quote'and\"quote", '$() ` : ;',
                      'line\nbreak', *('a' + '\\' * length + '.b' for length in (1, 2, 3, 4, 8)),
                      "a\\'b"):
            quoted = ops._escape_shell_arg(value, translate_path=False)
            result = ops._exec(f"printf '%s' {quoted} | od -An -v -tx1")
            assert result.exit_code == 0, (value, result.stdout)
            assert bytes.fromhex(result.stdout) == value.encode('utf-8'), (serialized, value, result.stdout)
    finally:
        if isinstance(backend, LocalEnvironment):
            backend.cleanup()

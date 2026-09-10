"""Regression tests for Windows ripgrep regex argument quoting."""

import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


def test_search_regex_bypasses_path_normalization(monkeypatch):
    """Regex escapes must never pass through path normalization."""
    import tools.environments.local as local_mod

    calls = []
    env = MagicMock()
    env.cwd = "/tmp/test"

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        if command.startswith("test -e "):
            return {"output": "exists", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "yes", "returncode": 0}
        return {"output": "", "returncode": 0}

    env.execute.side_effect = execute
    monkeypatch.setattr(
        local_mod,
        "_bash_safe_path",
        lambda value: value.replace("\\", "/"),
    )

    result = ShellFileOperations(env).search(
        r'AddDataFilePath\("Digital',
        path="/tmp/test",
    )

    assert result.error is None
    rg_calls = [
        (command, kwargs)
        for command, kwargs in calls
        if command.startswith(("{ rg ", "set -o pipefail; { rg "))
    ]
    assert rg_calls
    assert all(
        kwargs.get("stdin_data") == 'AddDataFilePath\\("Digital'
        for _, kwargs in rg_calls
    )


def test_search_glob_bypasses_path_normalization(monkeypatch):
    """Glob syntax is shell data, not a filesystem path to rewrite."""
    import tools.environments.local as local_mod

    commands = []
    env = MagicMock()
    env.cwd = "/tmp/test"

    def execute(command, **kwargs):
        commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "yes", "returncode": 0}
        return {"output": "", "returncode": 0}

    env.execute.side_effect = execute
    monkeypatch.setattr(
        local_mod,
        "_bash_safe_path",
        lambda value: value.replace("\\", "/"),
    )

    result = ShellFileOperations(env).search(
        "needle",
        path="/tmp/test",
        file_glob=r"dir\*.cs",
    )

    assert result.error is None
    assert any("'dir\\*.cs'" in command for command in commands)


def test_search_pattern_reaches_rg_via_heredoc_transport(tmp_path):
    """SDK backends that embed stdin as a heredoc must feed rg, not head."""
    bash = shutil.which("bash")
    if bash is None or shutil.which("rg") is None:
        pytest.skip("bash and ripgrep are required")

    source = tmp_path / "sample.cs"
    source.write_text(
        'unrelated line\nAddDataFilePath("Digital")\n',
        encoding="utf-8",
    )
    env = MagicMock()
    env.cwd = str(tmp_path)

    def execute(command, **kwargs):
        stdin_data = kwargs.get("stdin_data")
        if stdin_data is not None:
            command = BaseEnvironment._embed_stdin_heredoc(command, stdin_data)
        completed = subprocess.run(
            [bash, "-c", command],
            cwd=str(tmp_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=kwargs.get("timeout", 60),
        )
        return {"output": completed.stdout, "returncode": completed.returncode}

    env.execute.side_effect = execute

    result = ShellFileOperations(env).search(
        r'AddDataFilePath\("Digital',
        path=str(tmp_path),
        file_glob="*.cs",
    )

    assert result.error is None
    assert result.total_count == 1


@pytest.mark.parametrize(
    ("pattern", "content", "expected_count", "expects_warning"),
    [
        ("a\nb", "anb\n", 0, True),
        ("a\rb", "arb\n", 0, False),
    ],
    ids=("newline", "carriage-return-no-false-match"),
)
def test_grep_fallback_preserves_control_characters(
    tmp_path,
    monkeypatch,
    pattern,
    content,
    expected_count,
    expects_warning,
):
    """grep must not reinterpret CR/LF controls as letters r/n."""
    if shutil.which("grep") is None:
        pytest.skip("grep is required")

    source = tmp_path / "sample.txt"
    source.write_text(content, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")

    result = ops.search(pattern, path=str(tmp_path), file_glob="*.txt")

    assert result.error is None
    assert result.total_count == expected_count
    if expects_warning:
        assert result.warning is not None
        assert "line-oriented" in result.warning
    else:
        assert result.warning is None


@pytest.mark.windows_only
@pytest.mark.parametrize(
    ("content", "pattern"),
    [
        ('Form1.AddDataFilePath("Digital\\\\Setting")\n', r'AddDataFilePath\("Digital'),
        (r"value = a\nb" + "\n", r"a\\nb"),
        ('value = x"y\n', r'x\"y'),
    ],
    ids=("escaped-metacharacter", "literal-backslash", "escaped-quote"),
)
def test_native_windows_rg_receives_regex_backslashes(tmp_path, content, pattern):
    """Native rg must receive both single and doubled regex backslashes intact."""
    if shutil.which("rg") is None:
        pytest.skip("native Windows ripgrep is not installed")

    source = tmp_path / "sample.cs"
    source.write_text(content, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    result = ops.search(pattern, path=str(tmp_path), file_glob="*.cs")

    assert result.error is None
    assert result.total_count == 1
    assert result.matches[0].path.endswith("sample.cs")


@pytest.mark.parametrize("transport", ["payload", "heredoc", "local"])
@pytest.mark.parametrize(
    ("pattern", "expected_count", "expected_error"),
    [
        ("(?x)alpha\n/beta", 1, False),
        ("(?x)alpha # comment\n/beta", 1, False),
        ("one\ntwo", 2, False),
        ("(?x)absent\n/match", 0, False),
        ("(?x)(alpha\n/beta", 0, True),
        ("(?x)alpha\r/beta", 1, False),
        ("(?x)alpha # comment\r\n/beta", 1, False),
        ("one\ntwo\n", 2, False),
        ("one\ntwo\n\n", 0, False),
        ('(?x)x\\"y\n', 1, False),
        (r"(?x)foo\\bar" + "\n", 1, False),
        ("(?x)alpha # $(touch injected); `touch injected`; '\"\n/beta", 1, False),
    ],
    ids=["extended-mode", "extended-comment", "literal-newline", "no-match", "invalid-regex",
         "carriage-return", "comment-crlf", "trailing-newline", "trailing-newlines-no-match",
         "escaped-quote", "literal-backslash", "shell-metacharacters"],
)
def test_shell_multiline_pattern_preserves_regex_semantics(
    tmp_path, transport, pattern, expected_count, expected_error,
):
    """Line breaks may be regex whitespace/comments, not only matchable characters."""
    bash = shutil.which("bash")
    if bash is None or shutil.which("rg") is None:
        pytest.skip("bash and ripgrep are required")
    source = tmp_path / "sample.txt"
    source.write_text('alpha/beta\nx"y\nfoo\\bar\none\ntwo\n', encoding="utf-8", newline="\n")
    env = MagicMock()
    env.cwd = str(tmp_path)

    def execute(command, **kwargs):
        stdin_data = kwargs.get("stdin_data")
        if transport == "heredoc" and stdin_data is not None:
            command = BaseEnvironment._embed_stdin_heredoc(command, stdin_data)
            stdin_data = None
        completed = subprocess.run(
            [bash, "-c", command], cwd=str(tmp_path), input=stdin_data,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=kwargs.get("timeout", 60),
        )
        return {"output": completed.stdout, "returncode": completed.returncode}

    env.execute.side_effect = execute
    if transport == "local":
        env = LocalEnvironment(cwd=str(tmp_path))
    result = ShellFileOperations(env).search(pattern, path=str(source))

    assert bool(result.error) is expected_error
    assert result.total_count == expected_count
    assert not (tmp_path / "injected").exists()


@pytest.mark.parametrize(
    ("pattern", "content", "hint"),
    [
        ("(?x)ALPHA\n/BETA", "alpha/beta\n", "case-insensitive"),
        ("lookup[key+1]\r", "lookup[key+1]\r\n", "literal match"),
        ("lookup[key+1]\r", "lookup[key+1]\\r\n", None),
        ("lookup[key+1]\nneedle", "lookup[key+1]\\nneedle\n", None),
    ],
    ids=["extended-mode", "fixed-carriage-return", "not-literal-backslash-r", "not-literal-backslash-n"],
)
def test_shell_zero_match_probes_preserve_control_characters(
    tmp_path, monkeypatch, pattern, content, hint,
):
    """Probe patterns must retain the same meaning even when rg uses -F."""
    if shutil.which("bash") is None or shutil.which("rg") is None:
        pytest.skip("bash and ripgrep are required")
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    (tmp_path / "sample.txt").write_bytes(content.encode("utf-8"))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    warning = ops._zero_match_probe(pattern, str(tmp_path), "*.txt")

    if hint is None:
        assert warning is None
    else:
        assert hint in warning
        assert "sample.txt" in warning

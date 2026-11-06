"""Search pattern transport semantics: bytes and multiline regex survival.

Salvaged from closed PR #92267's regression suite after main's
``translate_path=False`` transport (``92686159d1``) superseded its stdin
design. Complements PR #141719's escaped-glob coverage; these cases pin
the regex side: pattern bytes must never pass through the filesystem path
translator, and multiline regex semantics (``(?x)`` whitespace/comments,
CR/LF, trailing newlines) must survive the shell transport unchanged.
"""

import shutil
from unittest.mock import MagicMock

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


def _require_bash_rg():
    if shutil.which("bash") is None or shutil.which("rg") is None:
        pytest.skip("bash and ripgrep are required")


def test_search_regex_bypasses_path_normalization(monkeypatch):
    """Regex escapes must never pass through the filesystem path translator."""
    import tools.environments.local as local_mod

    commands = []
    env = MagicMock()
    env.cwd = "/tmp/test"

    def execute(command, **kwargs):
        commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists", "returncode": 0}
        if "command -v" in command:
            return {"output": "/usr/bin/rg", "returncode": 0}
        return {"output": "", "returncode": 0}

    env.execute.side_effect = execute
    # Simulate the Windows path translator on any host: the bug class was
    # patterns reaching rg with backslashes rewritten to forward slashes.
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
    rg_calls = [c for c in commands if "AddDataFilePath" in c and "rg" in c]
    assert rg_calls, "expected rg invocations carrying the pattern"
    for command in rg_calls:
        # The translator's signature (backslashes rewritten to slashes)
        # must never appear. Exact bytes on POSIX; the argv transport's
        # single backslash compensation is legitimate on Windows.
        assert 'AddDataFilePath/("Digital' not in command
        assert (
            r'AddDataFilePath\("Digital' in command
            or r'AddDataFilePath\\("Digital' in command
        )


@pytest.mark.parametrize(
    ("pattern", "content", "hint"),
    [
        ("(?x)ALPHA\n/BETA", "alpha/beta\n", "case-insensitive"),
        ("lookup[key+1]\r", "lookup[key+1]\r\n", "literal match"),
        ("lookup[key+1]\nneedle", "lookup[key+1]\\nneedle\n", None),
    ],
    ids=["extended-mode", "fixed-carriage-return", "not-literal-backslash-n"],
)
def test_shell_zero_match_probes_preserve_control_characters(
    tmp_path,
    monkeypatch,
    pattern,
    content,
    hint,
):
    """Probe patterns must retain the same meaning even when rg uses -F."""
    _require_bash_rg()
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    (tmp_path / "sample.txt").write_bytes(content.encode("utf-8"))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    warning = ops._zero_match_probe(pattern, str(tmp_path / "sample.txt"), "*.txt")

    if hint is None:
        assert warning is None
    else:
        assert warning is not None
        assert hint in warning


@pytest.mark.parametrize(
    ("pattern", "expected_count"),
    [
        ("(?x)alpha\n/beta", 1),
        ("(?x)alpha # comment\n/beta", 1),
        ("one\ntwo", 2),
        ("(?x)absent\n/match", 0),
        ("(?x)alpha\r/beta", 1),
        ("(?x)alpha # comment\r\n/beta", 1),
        ("one\ntwo\n", 2),
        ("one\ntwo\n\n", 0),
        (r"(?x)foo\\bar" + "\n", 1),
        ("(?x)alpha # $(touch injected); `touch injected`; '\"\n/beta", 1),
    ],
    ids=[
        "extended-mode",
        "extended-comment",
        "literal-newline",
        "no-match",
        "carriage-return",
        "comment-crlf",
        "trailing-newline",
        "trailing-newlines-no-match",
        "literal-backslash",
        "shell-metacharacters",
    ],
)
def test_shell_multiline_pattern_preserves_regex_semantics(
    tmp_path,
    monkeypatch,
    pattern,
    expected_count,
):
    """Line breaks may be regex whitespace/comments, not only matchable characters."""
    _require_bash_rg()
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    (tmp_path / "sample.txt").write_text(
        'alpha/beta\nx"y\nfoo\\bar\none\ntwo\n',
        encoding="utf-8",
        newline="\n",
    )
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    result = ops.search(pattern, path=str(tmp_path / "sample.txt"))

    assert result.error is None
    assert result.total_count == expected_count
    assert not (tmp_path / "injected").exists()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    ("content", "pattern"),
    [
        ('Form1.AddDataFilePath("Digital\\\\Setting")\n', r'AddDataFilePath\("Digital'),
        (r"value = a\nb" + "\n", r"a\\nb"),
    ],
    ids=("escaped-metacharacter", "literal-backslash"),
)
def test_native_windows_rg_receives_regex_backslashes(tmp_path, content, pattern):
    """Native rg must receive both single and doubled regex backslashes intact.

    Note: an escaped-quote case (``x\\"y``) is deliberately absent. On native
    Windows the argv backslash compensation makes ``\\"`` reach rg as a literal
    backslash, so a pattern containing ``\\"`` silently misses even though Rust
    regex treats ``\\"`` as a plain quote. Reported separately; adding it here
    would pin a known bug as expected behaviour.
    """
    if shutil.which("rg") is None:
        pytest.skip("native Windows ripgrep is not installed")

    (tmp_path / "sample.cs").write_text(content, encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    result = ops.search(pattern, path=str(tmp_path), file_glob="*.cs")

    assert result.error is None
    assert result.total_count == 1
    assert result.matches[0].path.endswith("sample.cs")

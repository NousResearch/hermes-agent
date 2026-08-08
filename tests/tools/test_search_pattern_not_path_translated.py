"""A search regex is not a path — the Windows path translator must not touch it.

``_escape_shell_arg`` rewrites every backslash to ``/`` on Windows
(``_bash_safe_path``), which is what ``C:\\Users\\x`` needs. The content-search
builders used it for the regex and the glob too, so on Windows ``\\d+`` reached
the engine as ``/d+`` and ``\\bword\\b`` as ``/bword/b`` — a silent zero-match
result for any pattern using a regex escape.

The translator is monkeypatched here so the Windows behaviour is exercised on
every platform; off Windows the real one is a no-op.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.file_operations import ShellFileOperations


@pytest.fixture()
def windows_translator(monkeypatch):
    """Force the Windows backslash rewrite regardless of the host platform."""
    import tools.environments.local as local_env

    monkeypatch.setattr(
        local_env, "_bash_safe_path", lambda p: p.replace("\\", "/") if p else p,
    )


def _fops_capturing(commands):
    env = MagicMock()
    env.cwd = "/work"

    def _execute(cmd, *args, **kwargs):
        commands.append(cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
        return {"output": "", "returncode": 1}

    env.execute = _execute
    return ShellFileOperations(env)


@pytest.mark.parametrize(
    "pattern",
    [r"\d+", r"\bword\b", r"foo\.bar", r"\s+end", r"a\\b"],
)
def test_regex_reaches_the_engine_unmodified(windows_translator, pattern):
    commands: list[str] = []
    fops = _fops_capturing(commands)
    fops._has_command = lambda cmd: cmd == "grep"

    fops._search_content(pattern, "/work", None, 20, 0, "content", 0)

    search_cmds = [c for c in commands if "grep -rnHE" in c]
    assert search_cmds, f"no grep command was issued (commands={commands!r})"
    assert f"'{pattern}'" in search_cmds[0], (
        f"the regex was rewritten before reaching grep: {search_cmds[0]!r}"
    )


def test_glob_filter_keeps_its_backslashes(windows_translator):
    commands: list[str] = []
    fops = _fops_capturing(commands)
    fops._has_command = lambda cmd: cmd == "grep"

    fops._search_content("needle", "/work", r"src\*.py", 20, 0, "content", 0)

    search_cmds = [c for c in commands if "grep -rnHE" in c]
    assert search_cmds
    assert r"'src\*.py'" in search_cmds[0], (
        f"the glob was rewritten before reaching grep: {search_cmds[0]!r}"
    )


def test_the_path_argument_is_still_translated(windows_translator):
    """The rewrite must keep working where it belongs — the search root."""
    commands: list[str] = []
    fops = _fops_capturing(commands)
    fops._has_command = lambda cmd: cmd == "grep"

    fops._search_content("needle", r"C:\work\src", None, 20, 0, "content", 0)

    search_cmds = [c for c in commands if "grep -rnHE" in c]
    assert search_cmds
    assert "'C:/work/src'" in search_cmds[0], (
        f"the search root lost its path translation: {search_cmds[0]!r}"
    )


def test_escape_helpers_split_path_from_literal(windows_translator):
    """The two quoters differ exactly in whether they translate separators."""
    fops = _fops_capturing([])

    assert fops._escape_shell_literal(r"\d+") == r"'\d+'"
    assert fops._escape_shell_arg(r"C:\work") == "'C:/work'"
    # Single-quote escaping is unchanged on both.
    assert fops._escape_shell_literal("it's") == "'it'\"'\"'s'"

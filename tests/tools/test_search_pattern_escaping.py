"""Regression tests for regex/glob pattern escaping (#92260).

On native Windows, ``_escape_shell_arg()`` ran match expressions through
the Git Bash path translator (``_bash_safe_path``), rewriting backslashes
to forward slashes: ``syncBalance\\(`` became ``syncBalance/(`` (rg
"unclosed group" failure) and ``currency\\.`` became ``currency/.``
(silent false-zero results). Match expressions are not paths and must be
quoted without path translation.
"""

import json

from tools.file_tools import search_tool


def _make_project(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "wallet.kt").write_text(
        "fun syncBalance(amount: Int) {\n"
        "    val rate = currency.value\n"
        "}\n",
        encoding="utf-8",
    )
    return d


def test_escaped_paren_pattern_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = _make_project(tmp_path)
    r = json.loads(search_tool(r"syncBalance\(", path=str(d), task_id="t-esc1"))
    assert r.get("error") is None
    assert "unclosed group" not in json.dumps(r)
    assert r["total_count"] == 1


def test_escaped_dot_pattern_is_not_false_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = _make_project(tmp_path)
    r = json.loads(search_tool(r"currency\.", path=str(d), task_id="t-esc2"))
    assert r.get("error") is None
    assert r["total_count"] >= 1


def test_file_name_glob_with_brackets_matches(tmp_path, monkeypatch):
    # find -name / rg -g glob expressions are match args too, not paths.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = _make_project(tmp_path)
    r = json.loads(search_tool("wallet[.]kt", target="files", path=str(d), task_id="t-esc3"))
    assert any(f.endswith("wallet.kt") for f in r.get("files", []))


def test_escape_match_arg_preserves_backslashes():
    from tools.file_operations import ShellFileOperations

    env = ShellFileOperations.__new__(ShellFileOperations)
    quoted = env._escape_match_arg(r"syncBalance\( \. [a-z]")
    assert quoted == r"'syncBalance\( \. [a-z]'"


def test_escape_shell_arg_still_translates_windows_paths(monkeypatch):
    from tools.file_operations import ShellFileOperations

    monkeypatch.setattr("sys.platform", "win32")
    env = ShellFileOperations.__new__(ShellFileOperations)
    quoted = env._escape_shell_arg("C:\\Users\\x\\file.py")
    assert "\\" not in quoted
    assert quoted.startswith("'") and quoted.endswith("'")

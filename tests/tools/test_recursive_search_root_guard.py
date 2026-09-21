"""Recursive grep/rg/find must not use $HOME, /, /tmp, or /private/tmp as the search root."""

from __future__ import annotations

import json
from pathlib import Path

from tools.terminal_tool_guards import recursive_search_root_block
from tools.terminal_tool import _pre_exec_block


def _error(blocked: str | None) -> dict:
    assert blocked is not None
    data = json.loads(blocked)
    assert data["exit_code"] == 1
    assert data["status"] == "blocked"
    assert "Blocked: recursive search of" in data["error"]
    assert "seeded repo path" in data["error"]
    return data


def test_grep_r_home_blocked(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    blocked = recursive_search_root_block(
        f"grep -rln llm-access {home} --include=*.md",
        cwd=str(tmp_path / "proj"),
        home=home,
    )
    data = _error(blocked)
    assert str(home.resolve()) in data["error"]


def test_grep_r_dollar_home_blocked(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    blocked = recursive_search_root_block(
        "grep -rln x $HOME",
        cwd=str(tmp_path / "proj"),
        home=home,
    )
    _error(blocked)


def test_grep_r_quoted_dollar_home_blocked(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    blocked = recursive_search_root_block(
        'grep -rln x "$HOME"',
        cwd=str(tmp_path / "proj"),
        home=home,
    )
    _error(blocked)


def test_find_quoted_tmp_blocked(tmp_path):
    blocked = recursive_search_root_block('find "/tmp" -name "*.md"', cwd=str(tmp_path))
    _error(blocked)


def test_rg_quoted_private_tmp_blocked(tmp_path):
    blocked = recursive_search_root_block('rg llm-access "/private/tmp"', cwd=str(tmp_path))
    _error(blocked)


def test_grep_r_after_other_short_flags_blocked(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    blocked = recursive_search_root_block(
        "grep -n -r $HOME",
        cwd=str(tmp_path / "proj"),
        home=home,
    )
    _error(blocked)


def test_grep_r_after_long_option_blocked(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    blocked = recursive_search_root_block(
        "grep --include=*.md -rln x $HOME",
        cwd=str(tmp_path / "proj"),
        home=home,
    )
    _error(blocked)


def test_find_tmp_blocked(tmp_path):
    blocked = recursive_search_root_block("find /tmp -name '*.md'", cwd=str(tmp_path))
    data = _error(blocked)
    assert "/tmp" in data["error"] or "/private/tmp" in data["error"]


def test_find_slash_blocked(tmp_path):
    blocked = recursive_search_root_block("find / -name llm-access", cwd=str(tmp_path))
    _error(blocked)


def test_rg_private_tmp_blocked(tmp_path):
    blocked = recursive_search_root_block(
        "rg llm-access /private/tmp",
        cwd=str(tmp_path),
    )
    _error(blocked)


def test_grep_r_under_project_dir_allowed(tmp_path):
    proj = tmp_path / "repo"
    proj.mkdir()
    (proj / "a.md").write_text("hello\n", encoding="utf-8")
    blocked = recursive_search_root_block(
        f"grep -r foo {proj}",
        cwd=str(tmp_path),
        home=tmp_path / "home",
    )
    assert blocked is None


def test_grep_without_recursive_flag_allowed_even_on_home_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    target = home / "notes.md"
    target.write_text("x\n", encoding="utf-8")
    blocked = recursive_search_root_block(
        f"grep llm-access {target}",
        cwd=str(tmp_path),
        home=home,
    )
    assert blocked is None


def test_pre_exec_block_raises_rejected_for_home_grep(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    from tools.terminal_tool import _Rejected

    try:
        _pre_exec_block(
            f"grep -r x {home}",
            env=None, env_type="local", cwd=str(tmp_path),
            workdir=None, session_key="t",
        )
    except _Rejected as exc:
        payload = json.loads(exc.args[0])
        assert payload["exit_code"] == 1
        assert "Blocked: recursive search of" in payload["error"]
        return
    raise AssertionError("expected _Rejected")

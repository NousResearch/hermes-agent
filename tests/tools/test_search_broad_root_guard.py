"""Regression tests: search_files refuses whole-disk roots before any walk.

A content/file search rooted at a multi-TB tree (``/``, the user home, or an
operator-listed root such as a model-storage mount) still *succeeds* — after rg
has read GiBs from the disk, starving every other process on it. Measured on a
96%-full SATA SSD: one ``rg --files --hidden`` over the workspace read 935 MiB
in 60 s.

Contract after the fix:
- A root that IS, or CONTAINS, a broad root (``/``, ``$HOME``, any entry of
  ``HERMES_SEARCH_BROAD_ROOTS`` / ``file_search.broad_roots``) is refused before
  any shell runs, with a guardrail error telling
  the caller to narrow ``path``.
- The same applies to every part of a multi-path string.
- A task-root-sized search below a broad root still runs and returns matches.
"""

import json

import pytest

from tools import file_tools
from tools.file_tools import search_tool


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    home = tmp_path / "home" / "user"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    big = tmp_path / "disk" / "workspace"
    task = big / "oss" / "repo"
    task.mkdir(parents=True)
    (task / "a.py").write_text("NEEDLE_TOKEN = 1\n")
    monkeypatch.setenv("HERMES_SEARCH_BROAD_ROOTS", str(big))
    return tmp_path


@pytest.fixture
def no_shell(monkeypatch):
    """Fail loudly if a refused search reaches the file-ops/shell layer."""
    def _boom(*_a, **_k):
        raise AssertionError("refused search must not reach the shell")
    monkeypatch.setattr(file_tools, "_get_file_ops", _boom)


def _refused(raw: str) -> dict:
    result = json.loads(raw)
    assert "error" in result, result
    assert "Narrow `path`" in result["error"]
    assert "narrow" in result["error"].lower()
    return result


@pytest.mark.parametrize("rel", ["disk/workspace", "disk", "home/user", "home"])
def test_broad_root_or_its_ancestor_is_refused(tree, no_shell, rel):
    _refused(search_tool("NEEDLE_TOKEN", path=str(tree / rel), task_id="t-broad"))


def test_filesystem_root_is_refused(tree, no_shell):
    _refused(search_tool("NEEDLE_TOKEN", target="files", path="/", task_id="t-broad"))


def test_trailing_slash_and_dotdot_do_not_bypass(tree, no_shell):
    sneaky = str(tree / "disk" / "workspace" / "oss" / "..") + "/"
    _refused(search_tool("NEEDLE_TOKEN", path=sneaky, task_id="t-broad"))


def test_multi_path_string_with_a_broad_part_is_refused(tree, no_shell):
    task = tree / "disk" / "workspace" / "oss" / "repo"
    _refused(search_tool("NEEDLE_TOKEN", path=f"{task}, {tree / 'disk'}", task_id="t-broad"))


def test_task_root_below_broad_root_still_searches(tree):
    task = tree / "disk" / "workspace" / "oss" / "repo"
    result = json.loads(search_tool("NEEDLE_TOKEN", path=str(task), task_id="t-broad"))
    assert "error" not in result, result
    assert result["total_count"] == 1


def test_config_list_is_honoured(tree, no_shell, monkeypatch):
    monkeypatch.delenv("HERMES_SEARCH_BROAD_ROOTS")
    other = tree / "srv"
    other.mkdir()
    from tools import search_root_guard
    monkeypatch.setattr(search_root_guard, "_config_roots", lambda: [str(other)])
    _refused(search_tool("NEEDLE_TOKEN", path=str(other), task_id="t-broad"))

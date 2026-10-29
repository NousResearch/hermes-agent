"""A search root named like engine stderr keeps its results.

The search lanes tell engine diagnostics from results by their ``rg: ``/``grep: ``/
``find: `` prefix, and every result line starts with the root as given. A relative
root literally named ``rg: data`` therefore turned all of its own results into
"diagnostics": ``total_count: 0``, no warning, no error. These live runs use real
rg/find on a temp tree.
"""

import os

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.platforms("linux", "macos")


def _ops(tree):
    return ShellFileOperations(LocalEnvironment(cwd=str(tree)), cwd=str(tree))


def _tree(tmp_path, root_name):
    (tmp_path / root_name).mkdir()
    (tmp_path / root_name / "report.txt").write_text("needle\n")
    return tmp_path


@pytest.mark.parametrize("root_name", ["rg: data", "grep: data"])
@pytest.mark.parametrize("target", ["files", "content"])
def test_rg_root_named_like_a_diagnostic_keeps_its_results(tmp_path, root_name, target):
    ops = _ops(_tree(tmp_path, root_name))
    if not ops._has_command("rg"):
        pytest.skip("needs ripgrep")

    pattern = "*.txt" if target == "files" else "needle"
    result = ops.search(pattern, path=root_name, target=target)

    assert result.error is None
    found = result.files or [m.path for m in result.matches]
    assert [os.path.normpath(p) for p in found] == [f"{root_name}/report.txt"]
    assert result.total_count == 1


def test_find_root_named_like_a_diagnostic_keeps_its_results(tmp_path, monkeypatch):
    ops = _ops(_tree(tmp_path, "find: data"))
    real_has = ops._has_command
    monkeypatch.setattr(ops, "_has_command", lambda cmd: False if cmd == "rg" else real_has(cmd))

    result = ops.search("*.txt", path="find: data", target="files")

    assert result.error is None
    assert [os.path.normpath(p) for p in result.files] == ["find: data/report.txt"]
    assert result.total_count == 1


def test_pruned_grep_root_named_like_a_diagnostic_keeps_its_results(tmp_path, monkeypatch):
    # A root that is (or sits under) a dot-named directory sends no-rg content
    # search down the pruned find|grep lane, which passes the root to find on its own.
    (tmp_path / "grep: data" / ".cfg").mkdir(parents=True)
    (tmp_path / "grep: data" / ".cfg" / "report.txt").write_text("needle\n")
    ops = _ops(tmp_path)
    real_has = ops._has_command
    monkeypatch.setattr(ops, "_has_command", lambda cmd: False if cmd == "rg" else real_has(cmd))

    result = ops.search("needle", path="grep: data/.cfg", target="content")

    assert result.error is None
    assert [os.path.normpath(m.path) for m in result.matches] == ["grep: data/.cfg/report.txt"]
    assert result.total_count == 1

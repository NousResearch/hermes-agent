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


@pytest.mark.parametrize(("engine", "root_name"), [
    ("rg", "-dash"), ("rg", " data"),
    # A dot-named root sends no-rg content search down the pruned find|grep lane.
    ("no-rg", "-dash/.cfg"), ("no-rg", " data/.cfg"),
])
def test_content_search_under_a_dash_or_space_led_root_finds_its_matches(tmp_path, monkeypatch, engine, root_name):
    # Unanchored, a ``-`` root parsed as an option (rg: an error; find: a silent 0) and a
    # whitespace-led root's result lines were dropped as stderr: 0 results, no error.
    (tmp_path / root_name).mkdir(parents=True)
    (tmp_path / root_name / "report.txt").write_text("needle\n")
    ops = _ops(tmp_path)
    real_has = ops._has_command
    if engine == "rg" and not real_has("rg"):
        pytest.skip("needs ripgrep")
    if engine == "no-rg":
        monkeypatch.setattr(ops, "_has_command", lambda cmd: False if cmd == "rg" else real_has(cmd))

    result = ops.search("needle", path=root_name, target="content")

    assert result.error is None
    assert [os.path.normpath(m.path) for m in result.matches] == [f"{root_name}/report.txt"]
    assert result.total_count == 1


def test_file_search_under_a_dash_led_root_lists_its_matching_directories(tmp_path):
    # rg --files lists files only; matching directories come from a find supplement,
    # which read an unanchored ``-a`` root as an option and listed nothing.
    (tmp_path / "-a" / "subdir").mkdir(parents=True)
    ops = _ops(tmp_path)
    if not ops._has_command("rg") or not ops._has_command("find"):
        pytest.skip("needs ripgrep and find")

    result = ops.search("*sub*", path="-a", target="files")

    assert result.error is None
    assert [os.path.normpath(p) for p in result.files] == ["-a/subdir"]


def test_zero_match_hint_under_a_dash_led_root_stays_inside_it(tmp_path):
    # The 0-match probe passed the raw root, so rg read ``-a`` as a flag, searched
    # the whole cwd and the hint named a file outside the root.
    (tmp_path / "-a").mkdir()
    (tmp_path / "-a" / "inside.txt").write_text("NEEDLE\n")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "outside.txt").write_text("NEEDLE\n")
    ops = _ops(tmp_path)
    if not ops._has_command("rg"):
        pytest.skip("needs ripgrep")

    result = ops.search("needle", path="-a", target="content")

    assert result.total_count == 0
    assert "inside.txt" in (result.warning or "")
    assert "outside.txt" not in (result.warning or "")

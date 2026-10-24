"""Native Windows search transport contracts; related upstream PR #92267."""

import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(("target", "engine"), [("content", "rg"), ("files", "rg"), ("content", "grep"), ("files", "find")])
def test_windows_search_keeps_escaped_glob_and_native_root(tmp_path, monkeypatch, target, engine):
    if shutil.which("rg") is None:
        pytest.skip("native Windows ripgrep is required")
    root = tmp_path / "root with spaces"
    root.mkdir()
    (root / "sample[1].txt").write_text("needle\n", encoding="utf-8", newline="\n")
    (root / "sample1.txt").write_text("needle\n", encoding="utf-8", newline="\n")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))

    if engine != "rg":
        monkeypatch.setattr(ops, "_resolve_command", lambda command: None)
        monkeypatch.setattr(ops, "_has_command", lambda command: command == engine)

    if target == "content":
        result = ops.search("needle", path=str(root), file_glob=r"sample\[1\].txt")
        paths = [match.path for match in result.matches]
    else:
        result = ops.search(r"sample\[1\].txt", path=str(root), target="files")
        paths = result.files

    assert result.error is None
    assert result.total_count == 1
    assert len(paths) == 1
    assert paths[0].endswith("sample[1].txt")


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    ("pattern", "content"),
    [
        (r"\d+", "value 42\n"),
        (r"call\(", "call()\n"),
        (r"a\\nb", "a\\nb\n"),
        ("alpha\nbeta", "alpha\nbeta\n"),
        ("(?x)alpha # comment\n/beta", "alpha/beta\n"),
    ],
    ids=["digit-class", "escaped-paren", "literal-backslash-n", "actual-newline", "extended-comment"],
)
def test_windows_regex_controls_keep_native_semantics(tmp_path, pattern, content):
    if shutil.which("rg") is None:
        pytest.skip("native Windows ripgrep is required")
    (tmp_path / "sample.txt").write_text(content, encoding="utf-8", newline="\n")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    result = ops.search(pattern, path=str(tmp_path), file_glob="*.txt")

    assert result.error is None
    assert result.total_count >= 1

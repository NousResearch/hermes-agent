import json

import pytest

from tools import file_tools as ft
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.fixture
def ops(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    backend = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": backend)
    return backend


@pytest.mark.parametrize("engine", ["rg", "grep", "grep_pruned"])
@pytest.mark.parametrize("mode", ["content", "files_only", "count"])
def test_literal_search_dispatch_preserves_exact_text(ops, tmp_path, monkeypatch, engine, mode):
    if engine == "rg" and not ops._has_command("rg"):
        pytest.skip("ripgrep unavailable")
    if engine.startswith("grep"):
        original = ops._has_command
        monkeypatch.setattr(ops, "_has_command", lambda command: False if command == "rg" else original(command))
    root = tmp_path / (".hidden" if engine == "grep_pruned" else "visible")
    root.mkdir()
    (root / "sample.txt").write_text("-x[1].a\n-x1za\n-x[1].a\n", encoding="utf-8")
    out = json.loads(registry.dispatch("search_files", {
        "path": str(root), "pattern": "-x[1].a", "literal": True, "output_mode": mode,
        "limit": 20, "context": 0}, task_id=engine+mode))
    assert not out.get("error"), out
    assert out["total_count"] == (1 if mode == "files_only" else 2), out


def test_filename_regex_hint_and_literal_rejection(ops, tmp_path):
    (tmp_path / "sample.py").write_text("print(1)\n", encoding="utf-8")
    def search(pattern, **args):
        return json.loads(registry.dispatch("search_files", {
            "path": str(tmp_path), "target": "files", "pattern": pattern, **args}, task_id=pattern+str(args)))
    assert "glob" in search(r".*\.py$")["warning"]
    assert search("*.py")["total_count"] == 1
    assert "glob" in search("*.py", literal=True)["error"]
    assert "warning" not in search("[ab]*.txt")
    for name in ["source.bin", "moved.bin"]:
        (tmp_path / name).write_text("data\n", encoding="utf-8")
    missed = search("source.bin|moved.bin")
    assert missed["total_count"] == 0
    assert missed["warning"] == "No filenames matched the glob pattern provided."
    assert "target=content" not in missed["warning"]
    assert "literal=true" not in missed["warning"]
    assert search("source.bin")["total_count"] == 1
    assert search("moved.bin")["total_count"] == 1


@pytest.fixture(params=["rg", "grep", "grep_pruned"])
def search_engine(request):
    return request.param


@pytest.fixture
def search_root(ops, tmp_path, monkeypatch, search_engine):
    engine = search_engine
    if engine == "rg" and not ops._has_command("rg"):
        pytest.skip("ripgrep unavailable")
    if engine.startswith("grep"):
        original = ops._has_command
        monkeypatch.setattr(ops, "_has_command", lambda command: False if command == "rg" else original(command))
    root = tmp_path / (".regex-tests" if engine == "grep_pruned" else "regex-tests")
    root.mkdir()
    return root


@pytest.mark.parametrize("literal", [None, False], ids=["omitted", "false"])
@pytest.mark.parametrize("mode", ["content", "files_only", "count"])
@pytest.mark.parametrize("pattern,text", [
    (r"^(alpha|beta)$", "alpha\nbeta\nalphabet\n"),
    (r"^item[0-9]+$", "item1\nitem42\nitemX\n"),
    (r"^literal\.\*$", "literal.*\nliteralXYZ\nliteral.*\n"),
])
def test_regex_dispatch_retains_semantics(search_root, literal, mode, pattern, text):
    (search_root / "sample.txt").write_text(text, encoding="utf-8")
    args = {"path": str(search_root), "pattern": pattern, "output_mode": mode}
    if literal is not None:
        args["literal"] = literal
    out = json.loads(registry.dispatch("search_files", args, task_id="regex-contract"))
    assert not out.get("error"), out
    assert out["total_count"] == (1 if mode == "files_only" else 2), out


@pytest.mark.parametrize("pattern", [
    'quote " and \'single\' [x].*',
    r"C:\folder\name[1].txt",
    "東京 café [7].*",
    r"literal\ntext",
])
def test_literal_special_text_is_preserved(search_root, pattern):
    (search_root / "sample.txt").write_text(pattern + "\ndecoy\n", encoding="utf-8")
    out = json.loads(registry.dispatch("search_files", {
        "path": str(search_root), "pattern": pattern, "literal": True,
    }, task_id="literal-special-text"))
    assert not out.get("error"), out
    assert out["total_count"] == 1, out
    assert out["matches"][0]["content"] == pattern
    assert "multiline" not in out.get("warning", "")


def test_regex_and_literal_modes_survive_multi_path_recovery(search_root):
    roots = [search_root / "one", search_root / "two"]
    for root in roots:
        root.mkdir()
        (root / "sample.txt").write_text("item7\nitem[0-9]+\n", encoding="utf-8")
    for literal in [False, True]:
        out = json.loads(registry.dispatch("search_files", {
            "path": ",".join(str(root) for root in roots), "pattern": "item[0-9]+",
            "literal": literal,
        }, task_id=f"multi-path-{literal}"))
        assert not out.get("error"), out
        assert out["total_count"] == 2, out
        assert {match["content"] for match in out["matches"]} == {"item[0-9]+" if literal else "item7"}


@pytest.mark.parametrize("literal", [None, False], ids=["omitted", "false"])
def test_invalid_regex_is_not_reported_as_zero_matches(search_root, literal):
    (search_root / "sample.txt").write_text("[\n", encoding="utf-8")
    args = {"path": str(search_root), "pattern": "["}
    if literal is not None:
        args["literal"] = literal
    out = json.loads(registry.dispatch("search_files", args, task_id="invalid-regex"))
    assert out.get("error"), out


def test_invalid_regex_text_matches_in_literal_mode(search_root):
    (search_root / "sample.txt").write_text("[\n", encoding="utf-8")
    out = json.loads(registry.dispatch("search_files", {
        "path": str(search_root), "pattern": "[", "literal": True,
    }, task_id="literal-bracket"))
    assert not out.get("error"), out
    assert out["total_count"] == 1, out


def test_filename_hint_does_not_change_successful_special_names_or_globs(ops, tmp_path):
    for name in ["report$.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text("data\n", encoding="utf-8")
    for pattern, expected in [("report$.txt", 1), ("[ab].txt", 2)]:
        out = json.loads(registry.dispatch("search_files", {
            "path": str(tmp_path), "target": "files", "pattern": pattern,
        }, task_id=pattern))
        assert not out.get("error"), out
        assert out["total_count"] == expected, out
        assert "filename globs" not in out.get("warning", "")
    out = json.loads(registry.dispatch("search_files", {
        "path": str(tmp_path / "missing"), "target": "files", "pattern": "^report$",
    }, task_id="filename-error"))
    assert out.get("error"), out
    assert "filename globs" not in out.get("warning", "")


@pytest.mark.platforms("posix")
def test_filename_hint_preserves_existing_pipe_filename(ops, tmp_path):
    # A pipe is a valid POSIX filename character but cannot be created on Windows.
    name = "source.bin|moved.bin"
    (tmp_path / name).write_text("data\n", encoding="utf-8")
    out = json.loads(registry.dispatch("search_files", {
        "path": str(tmp_path), "target": "files", "pattern": name,
    }, task_id="pipe-filename"))
    assert not out.get("error"), out
    assert out["total_count"] == 1, out
    assert not out.get("warning"), out


@pytest.mark.parametrize("mode", ["content", "files_only", "count"])
@pytest.mark.parametrize("literal", [False, True])
def test_valid_empty_search_is_not_an_error(search_root, mode, literal):
    (search_root / "sample.txt").write_text("present\n", encoding="utf-8")
    out = json.loads(registry.dispatch("search_files", {
        "path": str(search_root), "pattern": "absent", "literal": literal,
        "output_mode": mode,
    }, task_id="valid-empty-search"))
    assert not out.get("error"), out
    assert out["total_count"] == 0, out

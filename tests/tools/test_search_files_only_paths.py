"""files_only search returns every matching path, including paths with spaces.

Regression: the stdout/stderr shape filter treated a files_only line with whitespace
("My Project/app.py", "Application Support", "Google Drive") as diagnostic prose and dropped it,
while content and count modes returned the same file.
"""

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_operations_common import ExecuteResult
from tools.file_operations_search import _parse_search_output


@pytest.mark.parametrize("native", ["0", "1"], ids=["shell", "native"])
def test_files_only_keeps_paths_with_spaces(tmp_path, monkeypatch, native):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", native)
    spaced = tmp_path / "My Project" / "app.py"
    spaced.parent.mkdir()
    spaced.write_text("# TODO: ship\n", encoding="utf-8")
    (tmp_path / "plain.py").write_text("# TODO: test\n", encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    # Relative root: the tmp prefix ("pytest-0") contains a ``-<digit>`` that happens to satisfy
    # the content-line shape and would mask the bug for absolute paths.
    files = ops.search("TODO", path=".", output_mode="files_only")
    content = ops.search("TODO", path=".")

    assert not files.error, files.error
    assert sorted(files.files) == sorted({m.path for m in content.matches}), files.to_dict()
    assert any("My Project" in f for f in files.files)


@pytest.mark.parametrize("stdout, exit_code", [
    # rg exit 2: a per-file error beside real matches (verbatim rg 14 output, error first).
    ("rg: ./broken.py: No such file or directory (os error 2)\n./My Project/app.py\n./plain.py\n", 2),
    # Interrupted run (exit 130): partial engine output, then the executor's marker line.
    ("./My Project/app.py\n./plain.py\n\n[Command interrupted]", 130),
], ids=["partial-error", "interrupted"])
def test_files_only_returns_exactly_the_engine_paths(stdout, exit_code):
    """The paths a files_only search reports are the engine's path lines, spaces included, and
    nothing else: its diagnostics and the executor's interrupt marker are not files."""
    result = _parse_search_output(ExecuteResult(stdout=stdout, exit_code=exit_code), "files_only",
                                  limit=50, offset=0, context=0)
    assert result.error is None, result.error
    assert result.files == ["./My Project/app.py", "./plain.py"]


@pytest.mark.parametrize("native", ["0", "1"], ids=["shell", "native"])
def test_files_only_fatal_engine_error_is_not_read_as_paths(tmp_path, monkeypatch, native):
    """A fatal engine error prints several lines — rg's caret block and help prose — and no
    paths; files_only must surface it as a failed search, never as file names."""
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", native)
    (tmp_path / "plain.py").write_text("ab\n", encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    result = ops.search("(?<=a)b", path=".", output_mode="files_only")

    assert result.files == [], result.to_dict()
    assert result.error and result.error.startswith("Search failed"), result.to_dict()

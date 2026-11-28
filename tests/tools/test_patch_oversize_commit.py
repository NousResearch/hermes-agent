"""A stale oversized snapshot must never be deleted at commit."""
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.patch_parser import _commit_resolved, _resolve_operations, parse_v4a_patch


def test_commit_refuses_changed_oversized_file(tmp_path, monkeypatch):
    from tools import file_operations

    monkeypatch.setattr(file_operations, "_RAW_READ_MAX_BYTES", 64)
    path = tmp_path / "large.txt"
    path.write_text("a" * 1024)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    patch = "\n".join(["*" * 3 + " Begin Patch", "*" * 3 + f" Delete File: {path}", "*" * 3 + " End Patch", ""])
    parsed, error = parse_v4a_patch(patch)
    assert error is None
    resolved = _resolve_operations(parsed, ops)
    assert not resolved.errors
    path.write_text("b" * 1024)
    result = _commit_resolved(resolved.commits, ops)
    assert not result.success
    assert path.read_text() == "b" * 1024

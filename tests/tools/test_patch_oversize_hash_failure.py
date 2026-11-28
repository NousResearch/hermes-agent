"""A failed digest must not authorize deleting an oversized file."""
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


def test_hash_unavailable_refuses_deletion(tmp_path, monkeypatch):
    from tools import file_operations

    monkeypatch.setattr(file_operations, "_RAW_READ_MAX_BYTES", 64)
    path = tmp_path / "large.txt"
    path.write_text("a" * 1024)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    monkeypatch.setattr(ops, "file_sha256", lambda path: None)
    patch = "\n".join(["*" * 3 + " Begin Patch", "*" * 3 + f" Delete File: {path}", "*" * 3 + " End Patch", ""])
    parsed, error = parse_v4a_patch(patch)
    assert error is None
    result = apply_v4a_operations(parsed, ops)
    assert not result.success and result.error and "fingerprint" in result.error
    assert path.exists()

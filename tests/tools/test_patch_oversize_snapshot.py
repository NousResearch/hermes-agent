"""Oversized delete snapshots must detect same-size changes before any mutation."""
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.patch_parser import apply_v4a_operations, parse_v4a_patch


def test_oversized_delete_refuses_same_size_drift(tmp_path, monkeypatch):
    from tools import file_operations
    import tools.patch_parser as parser

    monkeypatch.setattr(file_operations, "_RAW_READ_MAX_BYTES", 64)
    path = tmp_path / "large.txt"
    path.write_text("a" * 1024)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    patch, error = parse_v4a_patch(f"*** Begin Patch\n*** Delete File: {path}\n*** End Patch\n")
    assert error is None
    original = parser._snapshot_drift

    def mutate_then_check(snaps, file_ops):
        path.write_text("b" * 1024)
        return original(snaps, file_ops)

    monkeypatch.setattr(parser, "_snapshot_drift", mutate_then_check)
    result = apply_v4a_operations(patch, ops)
    assert not result.success
    assert "changed after" in result.error
    assert path.read_text() == "b" * 1024

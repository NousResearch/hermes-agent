"""The oversized deletion must validate at the delete boundary."""
import hashlib
import sys

import pytest

from tools import file_operations
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.patch_parser import _commit_resolved, _resolve_operations, parse_v4a_patch


def test_same_size_replacement_at_delete_boundary_survives(tmp_path, monkeypatch):
    monkeypatch.setattr(file_operations, '_RAW_READ_MAX_BYTES', 64)
    path = tmp_path / 'data.txt'
    path.write_text('a' * 1024)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    patch, error = parse_v4a_patch(f'*** Begin Patch\n*** Delete File: {path}\n*** End Patch\n')
    assert error is None
    resolved = _resolve_operations(patch, ops)
    assert not resolved.errors
    delete = ops.delete_file

    def replace_before_delete(*args, **kwargs):
        path.write_text('b' * 1024)
        return delete(*args, **kwargs)

    monkeypatch.setattr(ops, 'delete_file', replace_before_delete)
    result = _commit_resolved(resolved.commits, ops)
    assert not result.success
    assert path.read_text() == 'b' * 1024


@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require elevated privileges on Windows")
def test_delete_oversized_symlink_preserves_target(tmp_path, monkeypatch):
    monkeypatch.setattr(file_operations, '_RAW_READ_MAX_BYTES', 64)
    target = tmp_path / 'large-target'
    target.write_text('a' * 1024)
    link = tmp_path / 'link'
    link.symlink_to(target)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    star = '*' * 3
    patch, error = parse_v4a_patch(f'{star} Begin Patch\n{star} Delete File: {link}\n{star} End Patch\n')
    assert error is None
    from tools.patch_parser import apply_v4a_operations
    result = apply_v4a_operations(patch, ops)
    assert result.success, result.error
    assert not link.is_symlink()
    assert target.read_text() == 'a' * 1024


@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require elevated privileges on Windows")
def test_digest_bound_delete_matches_and_rejects_changed_target(tmp_path):
    target = tmp_path / 'target'
    target.write_bytes(b'bytes')
    digest = hashlib.sha256(b'bytes').hexdigest()
    link = tmp_path / 'link'
    link.symlink_to(target)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    assert ops.delete_file(str(link), expected_sha256='a' * 64).error
    assert link.is_symlink() and target.read_bytes() == b'bytes'
    assert not ops.delete_file(str(target), expected_sha256=digest).error
    assert not target.exists()

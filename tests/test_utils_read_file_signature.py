"""Focused tests for read_file_with_signature symlink-loop fallback."""
from __future__ import annotations

from pathlib import Path

import pytest

from utils import read_file_with_signature


def test_read_file_with_signature_returns_none_on_oserror(tmp_path, monkeypatch):
    """OSError from Path.resolve (missing file, permission) → None, not crash."""
    p = tmp_path / "absent.txt"
    result = read_file_with_signature(p)
    assert result is None


def test_read_file_with_signature_returns_none_on_runtime_error(tmp_path, monkeypatch):
    """RuntimeError from Path.resolve (symlink loop) → None, not crash."""
    p = tmp_path / "loop.txt"

    def _raising_resolve(*_args, **_kwargs):
        raise RuntimeError("Symlink loop detected")

    monkeypatch.setattr(Path, "resolve", _raising_resolve)

    result = read_file_with_signature(p)
    assert result is None, (
        "read_file_with_signature must return None on a symlink-loop "
        "RuntimeError, not propagate the exception"
    )


def test_read_file_with_signature_still_works_for_normal_files(tmp_path):
    """Sanity: a readable regular file returns (bytes, signature)."""
    p = tmp_path / "normal.txt"
    p.write_bytes(b"hello world")
    result = read_file_with_signature(p)
    assert result is not None
    data, sig = result
    assert data == b"hello world"
    resolved, mtime_ns, size, digest = sig
    assert isinstance(resolved, str)
    assert size == len(b"hello world")
    assert len(digest) == 64  # SHA-256 hex

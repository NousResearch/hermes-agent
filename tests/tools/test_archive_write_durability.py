"""Regression for #156164 item 4: an archive the stub names must be durable and byte-exact.

A pruned original the store has not seen yet is the ONLY copy of that tool result, so the file
has to reach the disk before the stub pointing at it is committed, and it has to hold the exact
bytes the stub promises. ``errors="replace"`` silently stored a ``?`` for an undecodable surrogate
and compared sizes against the substituted bytes, so a corrupt archive passed as lossless.
"""
import os

import pytest

from tools.tool_result_storage import _write_to_spillover


def test_written_archive_is_fsynced_before_the_stub_can_name_it(tmp_path, monkeypatch):
    """Every archive write reaches the platform's fsync, not just the page cache."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    synced = []
    real_fsync = os.fsync

    def spy(fd):
        synced.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    path = _write_to_spillover("payload that must survive a crash", "durable.txt")

    assert path is not None
    assert len(synced) == 1, "the archive write must fsync exactly once before being returned"


def test_an_undecodable_surrogate_is_not_archived_as_a_lossy_copy(tmp_path, monkeypatch):
    """A lone surrogate cannot round-trip, so the archive is refused instead of storing ``?``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    path = _write_to_spillover("before\ud800after", "surrogate.txt")

    assert path is None, "an inexact archive must fail closed, not promise bytes it cannot hold"
    assert not (tmp_path / "cache" / "spillover" / "surrogate.txt").exists()


def test_a_normal_archive_holds_the_exact_bytes_it_was_handed(tmp_path, monkeypatch):
    """The happy path still stores the input byte for byte (including multi-byte text)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    content = "中文 and emoji \N{ROCKET} plus a lone tab\there\n"

    path = _write_to_spillover(content, "exact.txt")

    assert path is not None
    assert (tmp_path / "cache" / "spillover" / "exact.txt").read_bytes() == content.encode("utf-8")
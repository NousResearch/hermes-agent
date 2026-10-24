"""Atomic text writes can opt out of translation without changing existing callers."""

import os

from utils import atomic_write_bytes, atomic_write_text


def test_atomic_text_write_preserves_explicit_line_endings(tmp_path):
    target = tmp_path / "portable.txt"
    supplied = "LF café\nCRLF\r\nCR\rEnd\n"
    atomic_write_text(target, supplied, newline="")
    assert target.read_bytes() == supplied.encode("utf-8")


def test_atomic_writers_keep_default_text_and_binary_behavior(tmp_path):
    target = tmp_path / "native.txt"
    atomic_write_text(target, "First\nLast\n")
    assert target.read_bytes() == f"First{os.linesep}Last{os.linesep}".encode()
    payload = b"First\nSecond\r\n\x00\xff"
    atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload

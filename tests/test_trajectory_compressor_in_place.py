"""Regression tests for in-place (input == output) truncation guard.

See #84688 — compressing to the same path as the input truncates/overwrites
the source JSONL. main() must refuse output paths that resolve to the input.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trajectory_compressor import _reject_in_place_output, main


def _write_sample_jsonl(path: Path, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i, "role": "user"}) + "\n")


# ---------------------------------------------------------------------------
# _reject_in_place_output
# ---------------------------------------------------------------------------

def test_reject_same_lexical_path(tmp_path):
    src = tmp_path / "trajectories.jsonl"
    _write_sample_jsonl(src)
    assert _reject_in_place_output(src, tmp_path / "trajectories.jsonl") is True


def test_reject_symlink_alias(tmp_path):
    src = tmp_path / "trajectories.jsonl"
    _write_sample_jsonl(src)
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(src)
    # A symlink to the input is the same file — must be rejected.
    assert _reject_in_place_output(src, alias) is True


def test_reject_hardlink_alias(tmp_path):
    src = tmp_path / "trajectories.jsonl"
    _write_sample_jsonl(src)
    hardlink = tmp_path / "hardlink.jsonl"
    hardlink.hardlink_to(src)
    # A hardlink is the *same inode* as the input — resolve() alone would miss
    # it (two different path strings), but samefile() must reject it.
    assert _reject_in_place_output(src, hardlink) is True


def test_accept_distinct_output(tmp_path):
    src = tmp_path / "trajectories.jsonl"
    _write_sample_jsonl(src)
    out = tmp_path / "compressed.jsonl"
    assert _reject_in_place_output(src, out) is False


# ---------------------------------------------------------------------------
# main() file-mode guard
# ---------------------------------------------------------------------------

def test_main_refuses_in_place_file_output(tmp_path, capsys, monkeypatch):
    src = tmp_path / "trajectories.jsonl"
    _write_sample_jsonl(src)
    original = src.read_text(encoding="utf-8")

    # If the guard failed, the compressor would be invoked and (via the
    # mocked write) truncate the source. Guarding must prevent any call and
    # must surface a non-zero exit so a scripted pipeline sees the refusal.
    fake_compressor = MagicMock()
    with patch("trajectory_compressor.TrajectoryCompressor", return_value=fake_compressor):
        with pytest.raises(SystemExit) as excinfo:
            main(input=str(src), output=str(src))

    assert excinfo.value.code == 1
    fake_compressor.process_directory.assert_not_called()
    assert src.read_text(encoding="utf-8") == original
    assert "refusing to overwrite the source dataset" in capsys.readouterr().out


def test_main_allows_distinct_file_output(tmp_path, monkeypatch):
    src = tmp_path / "trajectories.jsonl"
    _write_sample_jsonl(src)
    out = tmp_path / "compressed.jsonl"

    fake_compressor = MagicMock()
    # Simulate a successful (trivial) directory compress so main proceeds.
    with patch("trajectory_compressor.TrajectoryCompressor", return_value=fake_compressor):
        main(input=str(src), output=str(out))

    assert fake_compressor.process_directory.called


def test_main_refuses_in_place_directory_output(tmp_path, capsys):
    # Directory mode: output path pointing back at the input directory must be
    # refused with a non-zero exit (second call site of the guard).
    src = tmp_path / "runs"
    src.mkdir()
    _write_sample_jsonl(src / "a.jsonl")

    fake_compressor = MagicMock()
    with patch("trajectory_compressor.TrajectoryCompressor", return_value=fake_compressor):
        with pytest.raises(SystemExit) as excinfo:
            main(input=str(src), output=str(src))

    assert excinfo.value.code == 1
    fake_compressor.process_directory.assert_not_called()
    assert "refusing to overwrite the source dataset" in capsys.readouterr().out

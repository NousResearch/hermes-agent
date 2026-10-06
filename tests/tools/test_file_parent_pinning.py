"""Managed local writes keep their validated ancestor when a path is swapped."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

from tools.environments.local import LocalEnvironment, _find_bash
from tools.file_operations import ShellFileOperations


def _local_env(cwd: Path) -> LocalEnvironment:
    """A real local-shell backend without profile/session startup side effects."""
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.cwd = str(cwd)

    def execute(command: str, **kwargs):
        completed = subprocess.run(
            [_find_bash(), "-c", command],
            cwd=kwargs.get("cwd", cwd),
            input=kwargs.get("stdin_data"),
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=kwargs.get("timeout"),
            check=False,
        )
        return {"output": completed.stdout, "returncode": completed.returncode}

    env.execute = execute
    return env


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["write", "replace", "delete", "move"])
def test_managed_mutation_pins_parent_across_validation_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A validated ancestor must not be re-walked after an atomic symlink swap."""
    safe_root = tmp_path / "managed"
    original = safe_root / "original"
    outside = tmp_path / "outside"
    original.mkdir(parents=True)
    outside.mkdir()
    route = safe_root / "route"
    route.symlink_to(original, target_is_directory=True)

    if operation == "move":
        (original / "source.txt").write_text("ORIGINAL SOURCE\n", encoding="utf-8")
        (outside / "source.txt").write_text("OUTSIDE SOURCE\n", encoding="utf-8")
        target = route / "source.txt"
    else:
        (original / "target.txt").write_text("old value\n", encoding="utf-8")
        (outside / "target.txt").write_text("old value\noutside marker\n", encoding="utf-8")
        target = route / "target.txt"

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(safe_root)), cwd=str(safe_root))

    # Pause after the operation's final managed-root validation.  Move validates
    # both endpoints, so its barrier is the second successful check.
    import tools.file_operations as file_operations

    real_check = file_operations.get_write_denied_error
    validated = threading.Event()
    resume = threading.Event()
    calls = 0
    stop_after = 2 if operation == "move" else 1

    def paused_check(path: str, *, verb: str = "Write", entry: bool = False):
        nonlocal calls
        error = real_check(path, verb=verb, entry=entry)
        if error is None:
            calls += 1
            if calls == stop_after:
                validated.set()
                assert resume.wait(10), "test did not release the validation barrier"
        return error

    monkeypatch.setattr(file_operations, "get_write_denied_error", paused_check)
    outcome: dict[str, object] = {}

    def mutate() -> None:
        try:
            if operation == "write":
                outcome["result"] = ops.write_file(str(target), "new value\n")
            elif operation == "replace":
                outcome["result"] = ops.patch_replace(str(target), "old value", "new value")
            elif operation == "delete":
                outcome["result"] = ops.delete_file(str(target))
            else:
                outcome["result"] = ops.move_file(
                    str(target), str(route / "destination.txt")
                )
        except BaseException as exc:  # surfaced in the main test thread
            outcome["exception"] = exc

    worker = threading.Thread(target=mutate, name=f"file-{operation}-swap")
    worker.start()
    assert validated.wait(10), "mutation never reached the validation barrier"

    replacement = safe_root / "replacement"
    replacement.symlink_to(outside, target_is_directory=True)
    os.replace(replacement, route)  # atomic ancestor-symlink swap: original -> outside
    resume.set()
    worker.join(10)
    assert not worker.is_alive(), "mutation did not finish after releasing the barrier"
    assert "exception" not in outcome, outcome.get("exception")

    result = outcome["result"]
    assert getattr(result, "error", None) is None, getattr(result, "error", None)
    if operation in {"write", "replace"}:
        assert (original / "target.txt").read_text(encoding="utf-8") == "new value\n"
        assert (outside / "target.txt").read_text(encoding="utf-8") == (
            "old value\noutside marker\n"
        )
    elif operation == "delete":
        assert not (original / "target.txt").exists()
        assert (outside / "target.txt").read_text(encoding="utf-8") == (
            "old value\noutside marker\n"
        )
    else:
        assert not (original / "source.txt").exists()
        assert (original / "destination.txt").read_text(encoding="utf-8") == "ORIGINAL SOURCE\n"
        assert (outside / "source.txt").read_text(encoding="utf-8") == "OUTSIDE SOURCE\n"
        assert not (outside / "destination.txt").exists()


def _managed_ops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ShellFileOperations, Path]:
    safe_root = tmp_path / "managed"
    safe_root.mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    return ShellFileOperations(LocalEnvironment(cwd=str(safe_root)), cwd=str(safe_root)), safe_root


@pytest.mark.platforms("posix")
def test_managed_write_into_missing_directory_still_creates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent that does not exist yet cannot be pinned: the write falls back to the shell path
    (which folds in ``mkdir -p``) and returns a result instead of raising FileNotFoundError."""
    ops, safe_root = _managed_ops(tmp_path, monkeypatch)
    target = safe_root / "newdir" / "deep" / "f.txt"

    result = ops.write_file(str(target), "hello\n")

    assert result.error is None, result.error
    assert result.dirs_created is True
    assert target.read_text(encoding="utf-8") == "hello\n"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["write", "replace"])
def test_managed_pinned_write_keeps_lint_and_real_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """The pinned path swaps only the transport (fd-relative write and re-read): the lint tier
    still runs and ``verified`` comes from re-reading the bytes, not from assuming them."""
    ops, safe_root = _managed_ops(tmp_path, monkeypatch)
    target = safe_root / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    pinned_writes = []
    real_write_at = ShellFileOperations._write_at

    def spy_write_at(parent_fd: int, name: str, data: bytes) -> None:
        pinned_writes.append(name)
        real_write_at(parent_fd, name, data)

    monkeypatch.setattr(ShellFileOperations, "_write_at", staticmethod(spy_write_at))

    if operation == "write":
        result = ops.write_file(str(target), "def broken(:\n")
    else:
        result = ops.patch_replace(str(target), "x = 1", "def broken(:")

    assert pinned_writes == ["mod.py"], "the edit did not take the pinned path"
    assert result.error is None, result.error
    assert result.lint and result.lint["status"] == "error", result.lint
    assert "SyntaxError" in result.lint["output"]
    if operation == "write":
        assert result.verified is True


@pytest.mark.platforms("posix")
def test_managed_pinned_write_reports_a_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bytes that did not land as intended fail verification instead of reporting verified=True."""
    ops, safe_root = _managed_ops(tmp_path, monkeypatch)
    target = safe_root / "notes.txt"
    target.write_text("old\n", encoding="utf-8")
    real_write_at = ShellFileOperations._write_at

    def lossy_write_at(parent_fd: int, name: str, data: bytes) -> None:
        real_write_at(parent_fd, name, data[:-1])  # drop the last byte

    monkeypatch.setattr(ShellFileOperations, "_write_at", staticmethod(lossy_write_at))

    result = ops.write_file(str(target), "new\n")

    assert result.error and "Post-write verification failed" in result.error

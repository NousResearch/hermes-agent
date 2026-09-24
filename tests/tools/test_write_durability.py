"""Durable atomic writes must really fsync wherever the platform allows it.

Two ways this silently degrades into "no fsync at all", both on the platform
the original truncation report came from (Windows + Git Bash/MSYS):
  * ``sync FILE`` is unsupported there (measured: exits 1),
  * ``os.fsync()`` on a read-only handle raises EBADF (errno 9).

The helper chains the fallbacks with ``2>/dev/null && return 0``, so a broken
python fsync doesn't fail the write — it just skips durability. These tests
execute the durability calls on the host platform and fail when the fsync
never lands (the flag assertions cover POSIX CI, where the read-only form is
both correct for files and required for directories).
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from tools.checkpoint_manager import _fsync_file, _fsync_parent_dir
from tools.environments.local import _find_bash
from tools.file_operations import ShellFileOperations


def _generated_atomic_write_script(tmp_path: Path) -> str:
    """Capture the exact shell script ``_atomic_write`` would run."""
    return _generated_atomic_write_script_for(tmp_path / "a.txt")


def _generated_atomic_write_script_for(target: Path) -> str:
    env = MagicMock()
    seen = {}

    def side_effect(command, stdin_data=None, **kwargs):
        seen["script"] = command
        return {"output": "", "returncode": 0}

    env.execute.side_effect = side_effect
    ShellFileOperations(env)._atomic_write(str(target), "hello\n")
    return seen["script"]


def _fsync_one_liner(script: str) -> str:
    match = re.search(r'python3 -c "([^"]+)"', script)
    assert match, f"no python3 fsync one-liner in the generated script: {script[:400]}"
    return match.group(1)


def _shell_path(path: Path) -> str:
    """Native path -> MSYS form (/c/Users/...), which is the form the local
    shell and the generated script carry on Windows."""
    posix = path.as_posix()
    if os.name == "nt" and len(posix) > 1 and posix[1] == ":":
        return "/" + posix[0].lower() + posix[2:]
    return posix


class TestShellWriteFsync:
    def test_helper_opens_a_handle_it_can_actually_fsync(self, tmp_path):
        one_liner = _fsync_one_liner(_generated_atomic_write_script(tmp_path))
        if os.name == "nt":
            # FlushFileBuffers needs write access; O_RDONLY raised EBADF and
            # the 2>/dev/null chain swallowed it -> durability was a no-op.
            assert "O_RDWR" in one_liner
        else:
            # POSIX: O_RDONLY is the only form that opens a directory (the
            # `_fsync "$d"` call) and it fsyncs regular files too.
            assert "O_RDONLY" in one_liner

    def test_inline_fsync_call_succeeds_on_this_platform(self, tmp_path):
        """Run the script's own fsync one-liner — a non-zero exit here means
        the write path fell through to `sync`, which does nothing for a
        single file on Windows."""
        one_liner = _fsync_one_liner(_generated_atomic_write_script(tmp_path))
        target = tmp_path / "probe.txt"
        target.write_text("data\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", one_liner, str(target)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            "durable-write fsync is a silent no-op on this platform "
            f"(exit {proc.returncode}): {proc.stderr.strip()}"
        )


class TestGeneratedScriptDurability:
    """End-to-end: run the real generated script through the local shell under
    `bash -x` and read the trace of the `_fsync` helper.

    This is the failure the review called out: the helper chains its fallbacks
    with `2>/dev/null && return 0`, so a python fsync that dies (wrong path
    form, read-only handle) is invisible — the write still reports success and
    durability is silently skipped. The trace shows exactly which branch ran:
    a successful fsync returns immediately, a broken one falls through to
    `command -v python` and the `sync` tail.
    """

    def test_temp_file_fsync_takes_the_python_branch_and_succeeds(self, tmp_path):
        target = tmp_path / "out.txt"
        script = _generated_atomic_write_script_for(target)
        script_file = tmp_path / "atomic.sh"
        script_file.write_text(script, encoding="utf-8", newline="\n")

        proc = subprocess.run(
            [_find_bash(), "-x", _shell_path(script_file)],
            input="hello\n",
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"script failed: {proc.stderr}"
        assert target.read_text(encoding="utf-8") == "hello\n"

        trace = proc.stderr.splitlines()
        start = next(
            i for i, line in enumerate(trace) if line.startswith("+ _fsync ")
        )
        block = []
        for line in trace[start:]:
            if line.startswith("+ _fsync ") and block:
                break
            block.append(line)

        py_line = next((ln for ln in block if "python3 -c" in ln), None)
        assert py_line, f"fsync helper never called python3: {block}"
        # No fallback line may appear: reaching `command -v python` or the
        # `sync` tail means the python fsync exited non-zero (silent no-op).
        assert not any(
            re.fullmatch(r"\+ command -v python\b", ln)
            or re.fullmatch(r"\+ sync\b.*", ln)
            for ln in block
        ), f"fsync fell through to the no-op fallback: {block}"
        if os.name == "nt":
            # MSYS hands the shell /c/... paths; python resolves those as
            # literal paths and fails. The fsync call must get the native form.
            assert " C:/" in py_line, f"fsync got a non-native path: {py_line}"


class TestPythonSideFsync:
    def test_fsync_file_reaches_os_fsync(self, tmp_path, monkeypatch):
        target = tmp_path / "ledger.json"
        target.write_text("{}", encoding="utf-8")
        outcomes = []
        real_fsync = os.fsync

        def spy(fd):
            try:
                real_fsync(fd)
            except OSError as exc:  # pragma: no cover - platform dependent
                outcomes.append(exc)
                raise
            outcomes.append(None)

        monkeypatch.setattr(os, "fsync", spy)
        _fsync_file(target)
        assert outcomes and outcomes[0] is None, (
            f"_fsync_file skipped durability on this platform: {outcomes}"
        )

    def test_fsync_parent_dir_is_best_effort(self, tmp_path, monkeypatch):
        outcomes = []
        real_fsync = os.fsync

        def spy(fd):
            try:
                real_fsync(fd)
            except OSError as exc:
                outcomes.append(exc)
                raise
            outcomes.append(None)

        monkeypatch.setattr(os, "fsync", spy)
        _fsync_parent_dir(tmp_path / "ledger.json")  # must never raise
        if os.name != "nt":
            assert outcomes and outcomes[0] is None

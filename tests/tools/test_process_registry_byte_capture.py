"""Byte consumers retain separate bounded streams without changing terminal text output."""
import shlex
import sys

import pytest

from tools.process_registry import ProcessRegistry


@pytest.mark.platforms("posix")
def test_native_capture_preserves_binary_streams_and_terminal_text(tmp_path):
    registry = ProcessRegistry()
    code = "import os; os.write(1, b'out\\x00\\xff'); os.write(2, b'err\\x00\\xfe')"
    session = registry.spawn_local(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}", cwd=str(tmp_path),
        max_output_bytes=64, shell_interactive=False,
    )
    try:
        assert session._completion_event.wait(15)
        assert session.exit_code == 0
        assert session.byte_capture.read("stdout", 0, 64) == b"out\x00\xff"
        assert session.byte_capture.read("stderr", 0, 64).endswith(b"err\x00\xfe")
        assert "out\x00" in session.output_buffer
        assert "err\x00" in session.output_buffer
    finally:
        registry.kill_process(session.id)


@pytest.mark.platforms("posix")
def test_native_capture_keeps_bounded_prefix_and_marks_overflow(tmp_path):
    registry = ProcessRegistry()
    code = "import os; os.write(1, b'x' * 100000); os.write(2, b'y' * 100000)"
    session = registry.spawn_local(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}", cwd=str(tmp_path),
        max_output_bytes=1024, shell_interactive=False,
    )
    try:
        assert session._completion_event.wait(15)
        assert session.byte_capture.read("stdout", 0, 2048) == b"x" * 1024
        assert session.byte_capture.truncated["stdout"]
        assert session.byte_capture.truncated["stderr"]
        assert len(session.byte_capture.read("stderr", 0, 2048)) == 1024
        assert session.exit_code == 0
    finally:
        registry.kill_process(session.id)

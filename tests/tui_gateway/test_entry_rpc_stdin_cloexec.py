"""Regression coverage for the stdio gateway's child-process boundary."""

import os
import sys

import pytest

from tui_gateway import entry


@pytest.mark.skipif(os.name != "posix", reason="POSIX close-on-exec semantics")
def test_rpc_stdin_is_closed_before_a_bare_child_executes(monkeypatch):
    """A child with no explicit stdin must not be able to consume RPC bytes."""
    original_stdin = os.dup(0)
    original_inheritable = os.get_inheritable(0)
    reader, writer = os.pipe()
    try:
        os.dup2(reader, 0)
        os.write(writer, b'{"method":"prompt.submit"}\n')
        monkeypatch.setattr(entry.sys, "stdin", type("RpcStdin", (), {"fileno": lambda _: 0})())

        entry._close_rpc_stdin_on_exec()

        output_reader, output_writer = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.dup2(output_writer, 1)
            os.close(output_reader)
            os.close(output_writer)
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    "-c",
                    "import os; "
                    "\ntry: os.read(0, 1)\n"
                    "except OSError: print('stdin-closed')\n"
                    "else: print('stdin-inherited')",
                ],
            )

        os.close(output_writer)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert os.read(output_reader, 100) == b"stdin-closed\n"
        os.close(output_reader)
    finally:
        os.dup2(original_stdin, 0, inheritable=original_inheritable)
        os.close(original_stdin)
        os.close(reader)
        os.close(writer)

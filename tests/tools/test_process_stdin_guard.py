"""process(action="write"/"submit") cannot bypass the terminal command guard.

A background process can be an interactive shell or interpreter, so its stdin is a second command
channel: start a harmless ``bash`` in the background, then feed it the command ``terminal()`` would
have refused. Each line the process would execute now goes through the same guard, reassembled
across writes and with shell line editing applied. Salvages the per-write check of community PR
#22557 and closes its split-write gap.
"""

import time
from unittest.mock import MagicMock

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession

HARDLINE = "rm -rf $HOME"


@pytest.fixture
def shell():
    registry = ProcessRegistry()
    stdin = MagicMock()
    session = ProcessSession(id="proc_stdin", command="bash", task_id="t1", started_at=time.time())
    session.process = MagicMock(stdin=stdin)
    registry._running[session.id] = session
    return registry, session, stdin


def _written(stdin):
    return "".join(call.args[0] for call in stdin.write.call_args_list)


def test_hardline_line_is_refused_before_reaching_the_process(shell):
    registry, session, stdin = shell
    result = registry.submit_stdin(session.id, HARDLINE)
    assert result["status"] in ("blocked", "pending_approval")
    stdin.write.assert_not_called()


def test_a_command_split_across_writes_is_judged_as_one_line(shell):
    registry, session, stdin = shell
    assert registry.write_stdin(session.id, "rm -rf ")["status"] == "ok"
    result = registry.write_stdin(session.id, "$HOME\n")
    assert result["status"] in ("blocked", "pending_approval")
    assert "\n" not in _written(stdin)
    # "$HOME" never reached the shell: Enter now runs only the harmless fragment it holds.
    assert registry.write_stdin(session.id, "\n")["status"] == "ok"
    assert _written(stdin) == "rm -rf \n"
    # Fragments forwarded without a line end are judged together once Enter arrives.
    assert registry.write_stdin(session.id, "rm -rf ")["status"] == "ok"
    assert registry.write_stdin(session.id, "$HOME")["status"] == "ok"
    assert registry.write_stdin(session.id, "\r")["status"] in ("blocked", "pending_approval")
    assert _written(stdin) == "rm -rf \nrm -rf $HOME"


def test_line_editing_keys_are_applied_before_judging(shell):
    registry, session, stdin = shell
    assert registry.write_stdin(session.id, "rm -rf $HOMEX\x7f\n")["status"] in ("blocked", "pending_approval")
    stdin.write.assert_not_called()
    # ^U kills the pending text, so what runs is only what follows it.
    assert registry.write_stdin(session.id, "rm -rf $HO")["status"] == "ok"
    assert registry.write_stdin(session.id, "\x15echo hello\n")["status"] == "ok"
    assert _written(stdin).endswith("\x15echo hello\n")


def test_ordinary_interactive_input_is_forwarded(shell):
    registry, session, stdin = shell
    for data in ("y\n", "ls -la\n", "print('hi')\n", "\x03", "q"):
        assert registry.write_stdin(session.id, data)["status"] == "ok", data
    assert _written(stdin) == "y\nls -la\nprint('hi')\n\x03q"

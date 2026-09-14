"""Regression coverage for t_df72a8c6: a delegate_task child bypassing the Kanban CLI mutation guard by
unsetting HERMES_DELEGATED_CHILD_CONTEXT before shelling out.

hermes_cli/kanban_db.py::_assert_not_delegated_child_mutation and the CLI's own fast-fail both key off
that env var, which lives in the SUBPROCESS's environment — a shell command can `unset` it (or run under
`env -u HERMES_DELEGATED_CHILD_CONTEXT`) before ``hermes kanban complete ...`` ever execs, so neither guard
ever sees it set. The fix in tools/approval.py's _floor_block (and the execute_code mirror in
tools/code_execution_tool.py) checks agent.delegation_context.is_delegated_child_context() — a ContextVar
in the PARENT process the child's shell text cannot reach — before the subprocess is ever spawned.
"""
from __future__ import annotations

import pytest

from tools import approval as approval_module
from tools.approval_floors import _delegated_child_kanban_cli_block_result
from tools.kanban_cli_mutation_guard import contains_denied_kanban_mutation


class TestContainsDeniedKanbanMutation:
    @pytest.mark.parametrize("command", [
        "hermes kanban complete t_4989b28e --summary done",
        "hermes kanban block t_4989b28e --reason nope",
        "hermes kanban request-review t_4989b28e",
        "hermes kanban boards rm victim --delete",
        "unset HERMES_DELEGATED_CHILD_CONTEXT; hermes kanban complete t_4989b28e",
        "env -u HERMES_DELEGATED_CHILD_CONTEXT hermes kanban complete t_4989b28e",
        "HERMES_DELEGATED_CHILD_CONTEXT= hermes kanban complete t_4989b28e",
        "hermes -p some-profile kanban complete t_4989b28e",
        "hermes kanban --board beta complete t_4989b28e",
        '["hermes", "kanban", "complete", "t_4989b28e"]',
    ])
    def test_flags_mutating_verbs(self, command):
        assert contains_denied_kanban_mutation(command)

    @pytest.mark.parametrize("command", [
        "hermes kanban show t_4989b28e",
        "hermes kanban list",
        "hermes kanban boards list",
        "hermes kanban --help",
        "",
        None,
    ])
    def test_leaves_read_only_and_unrelated_alone(self, command):
        assert not contains_denied_kanban_mutation(command)

    def test_known_limitation_quoted_prose_is_not_distinguished_from_a_real_command(self):
        """Documented best-effort gap: unlike cron.lifecycle_guard (which re-scans shlex-tokenized
        segments and can tell a quoted argument from command position), this guard is a plain
        regex/string scan and cannot distinguish `echo "hermes kanban complete t_x"` (prose) from an
        actually-executed command. It fails toward MORE refusals here, not fewer — an over-broad match
        blocks something harmless; it never lets a real mutation through, which is the actual risk this
        guard exists to close."""
        assert contains_denied_kanban_mutation('echo "hermes kanban complete t_x"')


def test_floor_blocks_delegated_child_even_after_env_unset(monkeypatch):
    """The exact reported bypass: env flag unset, ContextVar still marks the child."""
    from agent.delegation_context import delegated_child_context

    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    command = "unset HERMES_DELEGATED_CHILD_CONTEXT; hermes kanban complete t_4989b28e --summary done"

    with delegated_child_context():
        result = approval_module._floor_block(command)

    assert result is not None
    assert result["approved"] is False
    assert "delegate_task child" in result["message"]
    assert "cannot mutate Kanban tasks via the CLI" in result["message"]


def test_floor_allows_the_same_command_outside_a_delegated_child(monkeypatch):
    """No regression for Dima's own interactive `hermes kanban complete` — floor is a no-op outside
    a delegate_task child."""
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    command = "hermes kanban complete t_4989b28e --summary done"

    result = approval_module._floor_block(command)

    assert result is None


def test_floor_allows_read_only_kanban_commands_inside_a_delegated_child():
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        result = approval_module._floor_block("hermes kanban show t_4989b28e")

    assert result is None


def test_block_message_tells_the_child_not_to_retry_via_env_manipulation():
    message = _delegated_child_kanban_cli_block_result()["message"]
    assert "unset" in message.lower() or "environment manipulation" in message.lower()

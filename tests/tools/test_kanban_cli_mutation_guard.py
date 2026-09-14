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

    @pytest.mark.parametrize("command", [
        "/usr/local/bin/hermes kanban complete t_4989b28e",
        "/home/dima/.local/bin/hermes kanban boards rm victim --delete",
        "./hermes kanban complete t_4989b28e",
        r"C:\Users\dima\AppData\hermes.exe kanban complete t_4989b28e",
        "hermes.exe kanban complete t_4989b28e",
        "hermes.EXE kanban boards rm victim",
    ])
    def test_flags_path_qualified_and_windows_invocations(self, command):
        """A child that evades the floor by spelling out an absolute path or the `.exe` suffix reaches a
        CLI whose own env-var check is already unset — this guard is the last line, so it must match."""
        assert contains_denied_kanban_mutation(command)

    @pytest.mark.parametrize("command", [
        "myhermes kanban complete t_4989b28e",
        "non-hermes kanban complete t_4989b28e",
        "pkg.hermes kanban complete t_4989b28e",
    ])
    def test_does_not_match_a_different_binary_whose_name_merely_contains_hermes(self, command):
        assert not contains_denied_kanban_mutation(command)

    @pytest.mark.parametrize("command", [
        "$(which hermes) kanban complete t_4989b28e",
        "`which hermes` kanban boards rm victim --delete",
        "$(command -v hermes) kanban complete t_4989b28e",
        "$(type -p hermes) kanban request-review t_4989b28e",
        "$(which hermes) kanban set-model t_4989b28e some-model",
    ])
    def test_flags_command_substitution_spelling_of_the_binary(self, command):
        """Command substitution hides the name from a plain string scan — the token before `kanban`
        is `)` or a backtick, never `hermes` — while the shell still resolves it and runs the real
        command. `$(which hermes)` is an ordinary way to spell "find it on PATH", no more exotic than
        the absolute path this guard already blocks."""
        assert contains_denied_kanban_mutation(command)

    def test_known_limitation_variable_indirection_is_out_of_reach(self):
        """`${HERMES_BIN} kanban complete t_x` is NOT caught, and cannot be by a static scan: the
        binary's name lives in a variable's VALUE, which this guard never sees, and the variable may
        be named anything at all (`${X}`). Substitution of a command whose text names `hermes` IS
        caught (test above) because the name is present in the text; an env-var reference is a
        different class. Pinned as a known gap so a future reader does not mistake it for coverage —
        the durable boundary for it is the ContextVar check in the delegated child's own process,
        which does not depend on reading shell text at all."""
        assert not contains_denied_kanban_mutation("${HERMES_BIN} kanban complete t_4989b28e")

    def test_flags_set_model_which_was_absent_from_both_denylists(self):
        """`set-model` repoints a task's model override — a mutation. It reached the parser without
        being added to either denied set, so a delegated child could run it past both the CLI's
        env-var check and this guard. tests/tools/test_kanban_denied_actions.py now fails on any
        future verb added the same way."""
        assert contains_denied_kanban_mutation("hermes kanban set-model t_4989b28e some-model")

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

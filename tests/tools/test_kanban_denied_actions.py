"""The denied-action sets must cover every mutating verb the Kanban parser actually exposes.

Regression for the drift these tests exist to make impossible: `set-model` was added to
`hermes_cli/kanban_parser.py` and to NEITHER denylist, so a delegate_task child could repoint a
task's model override past both the CLI's env-var check and the pre-spawn floor guard. The two sets
had been duplicated across `hermes_cli/kanban.py` and `tools/kanban_cli_mutation_guard.py` behind a
"keep these two lists in sync" comment; a comment cannot fail a build, so the gap survived review
twice.

These assert a relationship between the parser and the denylists rather than freezing either one's
contents: adding a read-only verb is silently fine, adding a mutating verb fails here until it is
classified. Nothing below reads source text — the parser is built and introspected for real.
"""
from __future__ import annotations

import argparse

from hermes_cli.kanban_denied_actions import DENIED_BOARD_ACTIONS, DENIED_KANBAN_ACTIONS
from hermes_cli.kanban_parser import build_parser

# Verbs that only READ board state — each verified against its parser help text: `show`/`list`/`runs`/
# `stats`/`assignees`/`context`/`log`/`attachments`/`notify-list` print, `diagnostics`/`diag` lists,
# `tail`/`watch` stream events. `boards` is the group itself (its children are classified separately).
# Everything the parser exposes is either here or denied; a new verb in neither set fails the coverage
# test below, which is the point — the failure is the prompt to classify it deliberately rather than
# defaulting it open.
READ_ONLY_ACTIONS = frozenset({
    "show", "list", "ls", "runs", "diagnostics", "diag", "attachments", "boards", "help",
    "assignees", "context", "log", "notify-list", "stats", "tail", "watch",
})
READ_ONLY_BOARD_ACTIONS = frozenset({
    "list", "ls", "show", "current", "export",
})


def _parser_actions() -> tuple[frozenset[str], frozenset[str]]:
    """Every `kanban <verb>` and `kanban boards <verb>` the real parser accepts, aliases included.

    Builds the actual parser tree (aliases only exist once argparse has registered them), so a verb
    added to `_SPECS` shows up here without this test naming it."""
    root = argparse.ArgumentParser(prog="hermes")
    kanban_parser = build_parser(root.add_subparsers(dest="command"))

    actions: set[str] = set()
    board_actions: set[str] = set()

    def _choices(parser, dest: str) -> dict:
        for action in parser._actions:
            choices = getattr(action, "choices", None)
            if getattr(action, "dest", None) == dest and choices:
                return dict(choices)
        return {}

    task_choices = _choices(kanban_parser, "kanban_action")
    actions.update(task_choices)
    boards_parser = task_choices.get("boards")
    if boards_parser is not None:
        board_actions.update(_choices(boards_parser, "boards_action"))

    return frozenset(actions), frozenset(board_actions)


def test_every_parser_task_verb_is_classified_as_denied_or_read_only():
    actions, _ = _parser_actions()
    assert actions, "parser exposed no kanban verbs — introspection broke, not a real pass"

    unclassified = actions - DENIED_KANBAN_ACTIONS - READ_ONLY_ACTIONS
    assert not unclassified, (
        f"kanban verbs in neither denylist nor the read-only set: {sorted(unclassified)}. "
        "A mutating verb must be added to DENIED_KANBAN_ACTIONS or a delegate_task child can reach it."
    )


def test_every_parser_board_verb_is_classified_as_denied_or_read_only():
    _, board_actions = _parser_actions()
    assert board_actions, "parser exposed no `kanban boards` verbs — introspection broke"

    unclassified = board_actions - DENIED_BOARD_ACTIONS - READ_ONLY_BOARD_ACTIONS
    assert not unclassified, (
        f"`kanban boards` verbs in neither denylist nor the read-only set: {sorted(unclassified)}."
    )


def test_denylists_name_only_verbs_the_parser_actually_exposes():
    """The other direction: a denied verb the parser never offers is dead weight that reads as
    coverage. `hold`/`resume` are the one sanctioned exception — deliberately listed ahead of the
    PR that adds them, where denying a not-yet-exposed verb is inert."""
    actions, board_actions = _parser_actions()

    assert not (DENIED_KANBAN_ACTIONS - actions)
    assert not (DENIED_BOARD_ACTIONS - board_actions - {"hold", "resume"})


def test_the_cli_check_and_the_floor_guard_share_one_definition():
    """Both trust boundaries must read the same sets — importing the same names is the mechanism
    that replaced the 'keep these two lists in sync' comment."""
    from hermes_cli import kanban as cli_kanban
    from tools import kanban_cli_mutation_guard as guard

    assert cli_kanban.DENIED_KANBAN_ACTIONS is DENIED_KANBAN_ACTIONS
    assert cli_kanban.DENIED_BOARD_ACTIONS is DENIED_BOARD_ACTIONS
    assert guard.DENIED_KANBAN_ACTIONS is DENIED_KANBAN_ACTIONS
    assert guard.DENIED_BOARD_ACTIONS is DENIED_BOARD_ACTIONS

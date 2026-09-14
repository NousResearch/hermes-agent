"""Floor guard: block delegate_task children from mutating Kanban via the CLI.

``hermes_cli/kanban_db.py::_assert_not_delegated_child_mutation`` is the durable trust boundary, but it
only ever sees ``HERMES_DELEGATED_CHILD_CONTEXT`` in the SPAWNED subprocess's own environment. A child can
defeat that by running e.g. ``unset HERMES_DELEGATED_CHILD_CONTEXT; hermes kanban complete t_x ...`` in one
shell invocation — unsetting an inherited env var before exec is not something any downstream process can
prevent (see t_df72a8c6 / t_4989b28e).

This module's check is meant to run BEFORE the subprocess is spawned, inside the delegated child's own
Python process, against ``agent.delegation_context.is_delegated_child_context()`` (a ContextVar) rather
than the env var — the child's shell text can shape what its subprocess's environment looks like, but it
cannot reach back into the parent process and clear a ContextVar it doesn't have a handle to. Same
placement pattern as ``cron.lifecycle_guard`` being consulted from ``tools/code_execution_tool.py`` before
a gateway-lifecycle command reaches a real subprocess.
"""
from __future__ import annotations

import re
from typing import Optional

# Mirrors hermes_cli/kanban.py's _DELEGATED_CHILD_DENIED_ACTIONS / _DELEGATED_CHILD_DENIED_BOARD_ACTIONS.
# Duplicated rather than imported: hermes_cli.kanban pulls in the full CLI arg-parser machinery, which
# would be a heavy, layering-inverted import from a tools/ guard module. Keep these two lists in sync.
_DENIED_KANBAN_ACTIONS = frozenset({
    "init", "create", "swarm", "assign", "reclaim", "reassign", "link", "unlink",
    "claim", "comment", "attach", "attach-rm", "complete", "edit", "block",
    "schedule", "unblock", "promote", "archive", "dispatch", "daemon", "repair",
    "heartbeat", "notify-subscribe", "notify-unsubscribe", "specify", "decompose",
    "request-review", "request-changes", "reopen-review", "gc",
})
_DENIED_BOARD_ACTIONS = frozenset({
    "create", "new", "rm", "remove", "delete", "switch", "use", "rename",
    "set-default-workdir", "import",
})

_ACTIONS_ALT = "|".join(sorted(_DENIED_KANBAN_ACTIONS))
_BOARD_ACTIONS_ALT = "|".join(sorted(_DENIED_BOARD_ACTIONS))

# Python argv-list punctuation (`subprocess.run(["hermes", "kanban", "complete", ...])`) separates
# exec'd words with brackets/commas/quotes; stripped only for the punctuation-stripped re-scan, never
# from raw text. Mirrors cron/lifecycle_guard.py's constant of the same name/purpose (that one omits
# quotes because it re-scans shlex-tokenized segments instead; this guard is a plain string scan).
_ARGV_LIST_PUNCTUATION = re.compile(r"[\[\],\"']+")

# Anchored the same way as cron/lifecycle_guard.py's _GATEWAY_LIFECYCLE_PATTERN: the lookbehind keeps
# `hermes` from matching as a path component or word tail, while every real command position (text start,
# whitespace, `;`/`&`/`|`, `$(`, backtick) still matches. Flags before AND after `kanban` (`-p profile`,
# `--board x`) are allowed so a routed/scoped call is still caught.
_FLAG_RUN = r"(?:\s+(?:-{1,2}\S+(?:[ =]\S+)?))*"
_KANBAN_MUTATION_PATTERN = re.compile(
    r"(?i)(?:(?<![/\w.\-])hermes)\b"
    + _FLAG_RUN
    + r"\s+kanban" + _FLAG_RUN
    + r"\s+"
    r"(?:boards" + _FLAG_RUN + r"\s+(?:" + _BOARD_ACTIONS_ALT + r")\b"
    r"|(?:" + _ACTIONS_ALT + r")\b)"
)


def contains_denied_kanban_mutation(command: Optional[str]) -> bool:
    """True if *command* invokes a Kanban-mutating CLI verb: ``hermes kanban <verb>`` or
    ``hermes kanban boards <verb>`` for one of the actions ``hermes_cli/kanban.py`` denies to delegated
    children. Read-only verbs (``show``, ``list``, ``boards list``, ...) never match.

    Also tries a punctuation-stripped variant so a Python argv list
    (``subprocess.run(["hermes", "kanban", "complete", tid])``) is caught even though its tokens are
    joined by commas/brackets rather than whitespace — same normalization
    ``cron.lifecycle_guard._ARGV_LIST_PUNCTUATION`` applies for the analogous gateway-lifecycle case."""
    if not command or "kanban" not in command.lower():
        return False
    if _KANBAN_MUTATION_PATTERN.search(command):
        return True
    stripped = _ARGV_LIST_PUNCTUATION.sub(" ", command)
    return stripped != command and bool(_KANBAN_MUTATION_PATTERN.search(stripped))

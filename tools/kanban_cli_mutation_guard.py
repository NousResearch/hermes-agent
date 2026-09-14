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

from hermes_cli.kanban_denied_actions import DENIED_BOARD_ACTIONS, DENIED_KANBAN_ACTIONS

_ACTIONS_ALT = "|".join(sorted(DENIED_KANBAN_ACTIONS))
_BOARD_ACTIONS_ALT = "|".join(sorted(DENIED_BOARD_ACTIONS))

# Python argv-list punctuation (`subprocess.run(["hermes", "kanban", "complete", ...])`) separates
# exec'd words with brackets/commas/quotes; stripped only for the punctuation-stripped re-scan, never
# from raw text. Mirrors cron/lifecycle_guard.py's constant of the same name/purpose (that one omits
# quotes because it re-scans shlex-tokenized segments instead; this guard is a plain string scan).
_ARGV_LIST_PUNCTUATION = re.compile(r"[\[\],\"']+")

# Command substitution hides the binary's name from a plain string scan: in `$(which hermes) kanban
# complete t_x` the token immediately before `kanban` is `)`, so the anchor below never sees `hermes`
# at all. A shell resolves the substitution to the real path before exec, so the command runs; the
# guard must therefore scan the substitution's own text too. Collapsing the delimiters to whitespace
# reduces that case to one the main pattern already handles (`which hermes kanban complete t_x`) —
# the same reduction strategy as the argv-list pass, rather than enumerating substitution spellings
# inside the main pattern. Known gap by construction: `${HERMES_BIN} kanban complete t_x` cannot be
# caught this way (the name lives in the variable's VALUE, never in the text, and the variable may
# be named anything) — the ContextVar check in the child's own process is the boundary that does not
# depend on reading shell text. Pinned in tests/tools/test_kanban_cli_mutation_guard.py.
_SHELL_SUBSTITUTION_PUNCTUATION = re.compile(r"(?:\$\(|\$\{|[()`}])+")

# Anchored like cron/lifecycle_guard.py's _GATEWAY_LIFECYCLE_PATTERN, with two deliberate widenings for
# this guard's threat model (a child actively trying to evade it, rather than a job incidentally naming a
# command):
#   * `/` is NOT excluded by the lookbehind, so a path-qualified invocation
#     (`/usr/local/bin/hermes kanban complete t_x`) still matches. Excluding it would let any child evade
#     the floor by spelling out an absolute path — the CLI's own env-var check is already unset by then,
#     so this guard is the last line.
#   * An optional `.exe` suffix is consumed, so `hermes.exe kanban complete t_x` matches on Windows.
# `\w`, `.` and `-` stay excluded so a word tail (`myhermes`), a dotted attribute (`pkg.hermes`) or a
# hyphenated name (`non-hermes`) does not match. Flags before AND after `kanban` (`-p profile`,
# `--board x`) are allowed so a routed/scoped call is still caught.
_FLAG_RUN = r"(?:\s+(?:-{1,2}\S+(?:[ =]\S+)?))*"
_KANBAN_MUTATION_PATTERN = re.compile(
    r"(?i)(?:(?<![\w.\-])hermes(?:\.exe)?)\b"
    + _FLAG_RUN
    + r"\s+kanban" + _FLAG_RUN
    + r"\s+"
    r"(?:boards" + _FLAG_RUN + r"\s+(?:" + _BOARD_ACTIONS_ALT + r")\b"
    r"|(?:" + _ACTIONS_ALT + r")\b)"
)

# Each normalization reduces one syntactic disguise to plain command text. Applied independently
# (not chained) so one rewrite cannot mask a match the other would have found; a normalization that
# changes nothing is skipped rather than re-scanned.
_NORMALIZATIONS = (_ARGV_LIST_PUNCTUATION, _SHELL_SUBSTITUTION_PUNCTUATION)


def contains_denied_kanban_mutation(command: Optional[str]) -> bool:
    """True if *command* invokes a Kanban-mutating CLI verb: ``hermes kanban <verb>`` or
    ``hermes kanban boards <verb>`` for one of the actions ``hermes_cli/kanban.py`` denies to delegated
    children. Read-only verbs (``show``, ``list``, ``boards list``, ...) never match.

    Scans the raw text first, then each normalized variant: argv-list punctuation stripped (so
    ``subprocess.run(["hermes", "kanban", "complete", tid])`` is caught even though its tokens are joined
    by commas/brackets) and shell-substitution delimiters stripped (so ``$(which hermes) kanban complete
    t_x`` is caught even though `hermes` never sits adjacent to `kanban`). Same normalization strategy
    ``cron.lifecycle_guard`` applies for the analogous gateway-lifecycle case."""
    if not command or "kanban" not in command.lower():
        return False
    if _KANBAN_MUTATION_PATTERN.search(command):
        return True
    for normalization in _NORMALIZATIONS:
        normalized = normalization.sub(" ", command)
        if normalized != command and _KANBAN_MUTATION_PATTERN.search(normalized):
            return True
    return False

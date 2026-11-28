"""Judge one command against Hermes's own approval rules — the public seam for plugins.

A plugin that runs a host's commands on that host (Hermes Tether) must judge each one
with THAT machine's Hermes rules before running it: the hardline floors, the
dangerous-command findings, the user's permanent allowlist and saved approvals, and the
approval mode. Until this seam it reached into private internals
(``tools.approval._floor_block`` and friends) and walked the call stack of Hermes's own
gate; plugins import only this module now.

``judge_command`` composes exactly the checks Hermes's own command gate composes
(``check_all_command_guards`` minus the human prompt, which the plugin's own channel
replaces):

- the unconditional floors first, always — hardline catastrophic commands, deletion of
  the runtime this very process boots from, the ``sudo -S`` stdin guard and the user's
  own ``approvals.deny`` rules — so a ``block`` is final: nobody can approve it, not
  even the caller's gate;
- ``gate_approved`` when the caller's own Hermes gate already approved exactly this
  command;
- the approval mode and the permanent ``command_allowlist``;
- the dangerous-command findings, minus the keys the host's human approved and minus
  permanently approved keys.

Every check reads the same stores and config the session's own gate reads, so the
verdict is the machine's answer, not the plugin's opinion. If a rule check cannot run,
the command is never waved through: the plain detectors
(``tools.approval_detection.detect_hardline_command`` /
``detect_dangerous_command``, public for a plugin that wants the pattern layer alone)
still block hardline commands and ask on flagged ones.

The composed internals stay private on purpose; nothing here prompts a human.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

from tools import approval_context
from tools.approval import (
    _NO_REVIEWER_BYPASS, _command_findings, _floor_block, _is_permanently_approved, _scan_incomplete_block)
from tools.approval_detection import hardline_scan_incomplete
from tools.approval_floors import _command_matches_permanent_allowlist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandJudgement:
    """One command judged against Hermes's own approval rules.

    ``verdict`` is ``"block"`` (a floor hit: final, nobody can approve it), ``"ask"``
    (the findings still need a human yes from the caller's own channel) or ``"run"``.
    ``reason`` carries the floor's message on ``block``; ``pattern_keys`` and
    ``description`` carry the findings still needing consent on ``ask``. A ``run``
    judgement carries none of them.
    """

    verdict: str
    reason: str | None = None
    pattern_keys: tuple[str, ...] = ()
    description: str | None = None


def judge_command(command: str, *,
                  approved_keys: Iterable[str] = (),
                  gate_approved: bool = False) -> CommandJudgement:
    """Judge *command* the way Hermes's own command gate would, without prompting anyone.

    ``approved_keys`` are the pattern keys a human already approved for exactly this
    command on the caller's channel; ``gate_approved`` says the caller's own Hermes gate
    approved it. Returns a :class:`CommandJudgement` whose ``verdict`` is ``"block"``
    (the command may not run, with the floor's message in ``reason``), ``"ask"`` (the
    caller must get a human yes for the findings in ``pattern_keys`` / ``description``
    before it runs) or ``"run"``.
    """
    granted = {str(key) for key in approved_keys or ()}
    try:
        blocked = _floor_block(command, sudo_guard=True)
        if blocked is not None:
            return CommandJudgement(
                "block", reason=blocked.get("message") or "blocked by Hermes's command rules")
        if gate_approved:
            return CommandJudgement("run")
        # As the gate: a command the hardline scan could not read to the end is for a human only, so with nobody
        # reviewing (mode off) it is refused, and the permanent allowlist never covers it.
        unread = hardline_scan_incomplete(command)
        if approval_context._get_approval_mode() == "off":
            if unread:
                return CommandJudgement("block", reason=_scan_incomplete_block(_NO_REVIEWER_BYPASS)["message"])
            return CommandJudgement("run")
        if not unread and _command_matches_permanent_allowlist(command):
            return CommandJudgement("run")
        pending = [(key, text) for key, text in _command_findings(command)
                   if key not in granted and not _is_permanently_approved(key)]
    # A rule check that cannot run must not wave the command through. A partial answer would drop the floors
    # that did not run (the user's deny rules, the runtime self-delete floor) and the content rules, so the
    # command is refused; Hermes's own gate fails the call in the same situation.
    except Exception:
        logger.warning("judge_command: a rule check failed; the command is refused", exc_info=True)
        return CommandJudgement("block", reason="Hermes's command rules could not be checked, so the command was refused")
    if not pending:
        return CommandJudgement("run")
    return CommandJudgement(
        "ask", pattern_keys=tuple(key for key, _ in pending),
        description="; ".join(text for _, text in pending))

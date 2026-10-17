"""What an unattended approve-mode context may still refuse. Pure classification, no approval state.

``approvals.single_query_mode: approve`` (and ``cron_mode`` / ``unattended_mode``) auto-approve every
dangerous-pattern and Tirith finding, because nobody is present to answer a prompt. Below the
hardline floor, that made both detectors advisory in exactly the runs with nobody watching. An agent's
test script then sent ``X="rm -rf /home"; $X`` through the real terminal tool, and the
command ran. The hardline floor now resolves that spelling, but approve mode still passes some
commands, and this module refuses them:

* The program is unreadable. A command word that is a bare ``$VAR`` / ``${VAR}`` whose value the
  command does not fix takes its program, and possibly its arguments too (``$X`` word-splits), from
  the environment at run time. No detector can say what it runs, so it is refused whatever its
  arguments are. "Fixes" means an assignment that always runs in the same shell before the use
  (``PY=python3; $PY -c ...``) or a literal ``for p in a b`` loop. A prefix-only ``X=v cmd``, a
  conditional/subshell/pipeline assignment, ``X=$(cat f)`` (command output), and any builtin that
  writes the name with a value not in the text do not fix it: ``read``/``mapfile``/``printf -v``/
  ``wait -p``/``getopts``/``let`` however spelled (``IFS= read X``, ``builtin read``, ``printf -vX``,
  a bare ``read`` setting REPLY, ``R=read; $R X``), ``eval``/``source``, a ``trap`` handler or a
  ``declare -n`` nameref. Shell payloads (``bash -c '...'``,
  a heredoc fed to a shell, an ``eval``'s arguments) are checked the same way.
* The inspection was incomplete. Any bound the resolver hits (too many value combinations, loop
  words, or shell payloads nested past the depth limit) is refused rather than treated as clean.
* A command substitution in command position (``$(which rm) -rf x``) is
  refused when its arguments look destructive: a short recursive flag group, ``--recursive``, a
  raw-device ``of=/dev/...``, or an operand that would be hardline under ``rm``.
* Two independent scanners say "destroys data". Tirith returns ``block`` AND the dangerous-pattern
  class is ``delete in root path``, whatever Tirith's rule is (the minimum rule).
  The other destructive classes (recursive delete, find -delete, dd, mkfs) are refused with a
  Tirith blast-radius block (``blast_*``). A Tirith ``analysis_incomplete`` block fires on most
  loops and ``$(...)``, so pairing that with a plain ``rm -r build/`` stays advisory.

A refusal is not a hardline block. yolo and ``approvals.mode: off`` still bypass it, and an
interactive user can still approve the same command. Only the unattended auto-approve is withdrawn.
"""

import re

from tools.approval_detection_assignments import opaque_command_leaders, uninspectable_reasons

# Dangerous-pattern descriptions that destroy data (not merely "risky"): the classes where an
# unattended auto-approve has no recovery path short of a backup restore.
DESTRUCTIVE_DESCRIPTIONS = frozenset({
    "delete in root path", "recursive delete", "recursive delete (long flag)",
    "recursive delete (flags after operands)", "xargs with rm", "find -exec/-execdir rm", "find -delete",
    "format filesystem", "disk copy", "write to block device",
})
# Refused with ANY Tirith block, not only a blast-radius one.
_ANY_BLOCK_DESCRIPTIONS = frozenset({"delete in root path"})
# Tirith's blast-radius rule family (tirith 0.4.x rule ids are ``blast_<what>``).
_TIRITH_BLAST_RULE_PREFIX = "blast_"
# A bare variable reference, optionally double-quoted: `$X`, `${X}`, `"${c[@]}"`.
_VARIABLE_LEADER_RE = re.compile(r'"?\$(?:\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]}]*\])?(?::?[-=?+][^}]*)?\}'
                                 r'|[A-Za-z_][A-Za-z0-9_]*)"?')
# A variable reference anywhere in a word (`/bin/$X`), and a command substitution (`$(...)`, backtick).
_VARIABLE_REFERENCE_RE = re.compile(r"\$(?:\{|[A-Za-z_])")
_SUBSTITUTION_RE = re.compile(r"\$\(|`")
# Destructive-looking argv for an unknown program: a short-option group holding r/R (`-rf`, `-R`,
# `-fr`), --recursive, or a dd-style raw-device output. `--version` / `-c` / `-m` do not match.
_DESTRUCTIVE_ARGS_RE = re.compile(r'(?:^|\s)(?:-[A-Za-z]*[rR][A-Za-z]*|--recursive)(?=\s|$)|\bof=/dev/')


def _opaque_leader_refusal(command: str) -> str | None:
    from tools.approval_detection import _shell_command_segment, detect_hardline_command
    incomplete = uninspectable_reasons(command)
    if incomplete:
        return (f"the command could not be fully inspected ({incomplete[0]}), so what it runs "
                "cannot be verified")
    for resolved, _, end, word in opaque_command_leaders(command):
        if _VARIABLE_LEADER_RE.fullmatch(word) or (_VARIABLE_REFERENCE_RE.search(word)
                                                   and not _SUBSTITUTION_RE.search(word)):
            return (f"command word {word} is a shell variable whose value this command never fixes "
                    "(unassigned, or assigned only conditionally, temporarily, in a subshell, from "
                    "command output, or overwritten by read/printf -v/eval/trap and the like), so "
                    "the program it runs cannot be inspected")
        arguments = _shell_command_segment(resolved, end)
        if arguments and (_DESTRUCTIVE_ARGS_RE.search(arguments) or detect_hardline_command(f"rm {arguments}")[0]):
            return (f"command word {word} is an unresolved command substitution and its arguments "
                    f"({arguments[:80]}) look destructive, so what it runs cannot be inspected")
    return None


def tirith_block_rules(tirith_result: dict | None) -> list[str]:
    """Every rule id in a Tirith ``block`` verdict (empty for allow/warn)."""
    if not tirith_result or tirith_result.get("action") != "block":
        return []
    return [str(f.get("rule_id")) for f in tirith_result.get("findings") or []] or ["block"]


def tirith_blast_rules(tirith_result: dict | None) -> list[str]:
    """The blast-radius rule ids in a Tirith ``block`` verdict (empty for allow/warn)."""
    return [rule for rule in tirith_block_rules(tirith_result) if rule.startswith(_TIRITH_BLAST_RULE_PREFIX)]


def unattended_approve_refusal(command: str, *, dangerous_description: str | None,
                               tirith_result: dict | None) -> str | None:
    """Why an unattended approve-mode context must NOT auto-approve *command*, or None if it may."""
    rules = (tirith_block_rules(tirith_result) if dangerous_description in _ANY_BLOCK_DESCRIPTIONS
             else tirith_blast_rules(tirith_result) if dangerous_description in DESTRUCTIVE_DESCRIPTIONS
             else [])
    if rules:
        return (f"Tirith blocked it ({', '.join(rules)}) and it matches the destructive pattern "
                f"'{dangerous_description}'")
    return _opaque_leader_refusal(command)

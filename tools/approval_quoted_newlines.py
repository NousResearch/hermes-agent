"""Quote-aware newline masking for :mod:`tools.approval_detection` (detection-only, no approval state)."""

from tools.approval_parser_budget import _scan_shell


def _mask_quoted_newlines(command: str) -> str:
    """Replace raw newlines inside single/double quotes with a space (detection-only).
    A quoted newline is DATA to the shell, yet the flat ``_CMDPOS`` class treats every raw ``\\n``
    as a command start, so multi-line quoted arguments (commit messages, heredoc text) tripped the
    hardline blocklist when a data line began with e.g. ``sudo reboot``. Quote tracking mirrors
    ``_iter_shell_command_starts``. Unquoted newlines pass through and ``_mark_command_starts``
    still re-inserts newlines at genuine starts; an unclosed quote absorbs following newlines
    exactly as the shell would, so masking them cannot hide a runnable command."""
    if "\n" not in command and "\r" not in command:
        return command
    return _mask_quoted_newlines_span(command, 0, len(command))


def _mask_quoted_newlines_span(command: str, start: int, end: int) -> str:
    """``_mask_quoted_newlines`` over ``command[start:end]``. A ``$(...)`` / backtick substitution
    inside double quotes is EXECUTABLE, not data: its body is re-scanned with a fresh quote state so a
    newline that separates commands inside it survives as a command boundary. Masking it as quoted
    data turned ``"$(grep x f\nreboot)"`` into ``... f reboot)``, which no later stage can tell from an
    operand; the pre-fix scanner only caught it by refusing the whole command as malformed."""
    out: list[str] = []
    for kind, i, j, quote in _scan_shell(command, start, end, subst="q"):
        if kind == "subst":
            # j is the index just past the closer; keep the opener and closer, recurse into the body.
            body_start = i + (2 if command.startswith("$(", i) else 1)
            body_end = j - 1
            out.append(command[i:body_start])
            out.append(_mask_quoted_newlines_span(command, body_start, body_end))
            out.append(command[body_end:j])
        elif quote and kind == "char" and command[i] in "\r\n":  # a CR too: PowerShell rules treat it as a break
            out.append(" ")
        else:
            out.append(command[i:j])
    return "".join(out)

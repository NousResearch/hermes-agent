"""Docker/compose lifecycle commands: which parts of a command are prose rather than executed.

``docker restart/stop/kill`` quoted inside ``echo``, ``git commit -m`` or a kanban comment is text,
not a container lifecycle command. The shell-scanning helpers live in the ``approval_detection``
facade; they are late-imported (``_facade()``) so this sibling never forms a module-level cycle.
"""

import os
import re


def _facade():
    from tools import approval_detection

    return approval_detection


_DOCKER_LIFECYCLE_DESCRIPTIONS = frozenset({
    "docker compose restart/stop/kill/down (container lifecycle)",
    "docker restart/stop/kill (container lifecycle)",
})
# Commands whose arguments and stdin are data, never executed, so docker lifecycle words inside
# them are prose (`echo`, `git commit -m`, `grep PATTERN`, `hermes kanban create --body`). Every
# other command's segment is still scanned: xargs, find -exec, ssh, watch and interpreters DO run
# their arguments, so an allowlist of prose commands (not a denylist of executors) keeps them flagged.
_DOCKER_LIFECYCLE_PROSE_COMMANDS = frozenset({
    "cat", "echo", "egrep", "fgrep", "git", "grep", "hermes", "printf", "rg",
})
# Prose commands that also ship executor-shaped subcommands are prose only for the listed
# subcommand directly after the name. git runs shell through `-c alias.x='!cmd'`, core.sshCommand,
# etc.; hermes stores or runs its arguments under `config set`, `cron`, `plugins`, ... and only the
# kanban text-carrying subcommands write board text. Global flags before the subcommand fail closed.
_DOCKER_LIFECYCLE_PROSE_SUBCOMMANDS = {
    "git": re.compile(r'\S*git\s+(?:commit|grep|log|show|tag|notes)\b'),
    "hermes": re.compile(
        r'\S*hermes\s+kanban\s+(?:create|comment|complete|block|edit|request-review|request-changes)(?=\s|$)'
    ),
}
# Pipe consumers that only display or count their stdin. Prose fed to anything else (`ssh host`,
# `tee file`, `xargs`, an interpreter) may be executed or persisted, so the prose stays scanned.
_DOCKER_LIFECYCLE_PROSE_SINKS = frozenset({"cut", "head", "less", "more", "sort", "tail", "uniq", "wc"})
_DOCKER_LIFECYCLE_SAFE_WRITE_TARGETS = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout"})


def _command_word_name(word: str) -> str:
    return os.path.basename(_facade()._deobfuscate_shell_word_for_detection(word)).lower()


def _segment_writes_file(segment: str) -> bool:
    """Whether an unquoted output redirection in *segment* writes somewhere other than an fd."""
    for kind, i, _, quote in _facade()._scan_shell(segment, subst="uq", brace=True):
        if kind != "char" or quote is not None or segment[i] != ">":
            continue
        target_at = i + 1 + (segment[i + 1:i + 2] in (">", "|"))
        if segment.startswith("&", target_at):
            target_at += 1
            if re.match(r"\s*(?:[0-9]+|-)(?![^\s;&|<>()])", segment[target_at:]):
                continue  # `>&2`, `2>&1`, `>&-`: fd duplication, nothing written to a path.
        target = _facade()._deobfuscate_shell_word_for_detection(_facade()._read_shell_word(segment, target_at)[2])
        if target not in _DOCKER_LIFECYCLE_SAFE_WRITE_TARGETS:
            return True  # a file, a `>(...)` process substitution, or an unreadable target
    return False


def _docker_lifecycle_prose_starts(command: str) -> set[int]:
    """Return the command-word offsets whose segment is pure prose for lifecycle matching.

    A segment is prose when its command is in ``_DOCKER_LIFECYCLE_PROSE_COMMANDS`` (passing the
    subcommand gate where one exists), it redirects output to no file, and everything downstream in
    its pipeline is itself prose or a display-only sink. Data that is written to a file or piped
    into another command may be persisted (an approval allowlist in config.yaml, a script run
    later) or executed (``| ssh host``), so it keeps the lifecycle scan. Fails closed: a pipe
    with no readable consumer is not prose.
    """
    words = [(start, _command_word_name(word)) for start, _, word in _facade()._iter_shell_command_word_spans(command)]
    verdicts: dict[tuple[int, bool], bool] = {}

    def is_prose(index: int, consumer: bool) -> bool:
        start, name = words[index]
        key = (start, consumer)
        if key in verdicts:
            return verdicts[key]
        verdicts[key] = False  # cycle guard: unreachable for well-formed input, fails closed
        end = _facade()._shell_command_segment_end(command, start, redirect_aware=True)
        gate = _DOCKER_LIFECYCLE_PROSE_SUBCOMMANDS.get(name)
        allowed = (consumer and name in _DOCKER_LIFECYCLE_PROSE_SINKS) or (
            name in _DOCKER_LIFECYCLE_PROSE_COMMANDS
            and (gate is None or gate.match(command[start:end].lower()))
        )
        if not allowed or _segment_writes_file(command[start:end]):
            return False
        if command.startswith("|", end) and not command.startswith("||", end):
            consumer_start = _facade()._skip_shell_whitespace(command, end + 1 + command.startswith("|&", end))
            # Wrapper chain only (`sudo tee f` -> sudo, tee); its last word is the real consumer.
            chain_end = _facade()._shell_command_segment_end(command, consumer_start)
            chain = [i for i, (s, _) in enumerate(words) if consumer_start <= s < chain_end]
            if not chain or not is_prose(chain[-1], consumer=True):
                return False
        verdicts[key] = True
        return True

    return {words[i][0] for i in range(len(words)) if is_prose(i, consumer=False)}


def _drop_prose_heredoc_bodies(command: str) -> str:
    """Remove here-doc bodies fed to a prose command, keeping the header and following lines.

    The command-start scanner treats every body line as an executable position, so documentation
    fed to ``cat``/``hermes`` reads as commands. Only bodies whose owning command is in
    ``_DOCKER_LIFECYCLE_PROSE_COMMANDS`` are dropped: ``ssh host <<EOF`` or ``python3 - <<EOF`` bodies
    are code and stay visible. An unquoted delimiter expands ``$(...)``/backticks in the body, so such
    a body stays visible too. Unterminated here-docs drop to the end (the shell would still be
    waiting for the delimiter). Bodies are removed rather than blanked so huge documents do not turn
    into long whitespace runs the command-start scanner re-walks from every line.
    """
    parts: list[str] = []
    cursor = 0
    while cursor < len(command):
        operator = next((
            index
            for kind, index, _, quote in _facade()._scan_shell(command, cursor)
            if kind == "char" and quote is None and command.startswith("<<", index)
            and (index == 0 or command[index - 1] != "<")
            and not command.startswith("<<<", index)
        ), None)
        if operator is None:
            break
        strip_tabs = command.startswith("<<-", operator)
        _, delimiter_end, delimiter_word = _facade()._read_shell_word(command, operator + 2 + strip_tabs)
        header_end = command.find("\n", delimiter_end)
        delimiter = _facade()._deobfuscate_shell_word_for_detection(delimiter_word)
        if header_end < 0 or not delimiter:
            break
        body_start = header_end + 1
        line_start = body_start
        body_end = len(command)
        while line_start < len(command):
            line_end = command.find("\n", line_start)
            if line_end < 0:
                line_end = len(command)
            line = command[line_start:line_end]
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                body_end = line_end + (line_end < len(command))
                break
            line_start = line_end + 1
        body = command[body_start:body_end]
        expands = not any(ch in delimiter_word for ch in "'\"\\") and ("$(" in body or "`" in body)
        owners = [start for start, _, _ in _facade()._iter_shell_command_word_spans(command[cursor:operator])]
        parts.append(command[cursor:body_start])
        # The header line carries the owner's redirections and pipe (`cat <<'EOF' > run.sh`).
        if expands or not owners or owners[-1] not in _docker_lifecycle_prose_starts(command[cursor:header_end]):
            parts.append(body)
        cursor = body_end
    parts.append(command[cursor:])
    return "".join(parts)


def _docker_lifecycle_executable_text(command: str) -> str:
    """Join the command segments that are not led by a prose command, for lifecycle matching.

    Nested ``$(...)``/backtick commands are their own command starts, so ``echo $(docker stop x)``
    is still scanned even though its ``echo`` segment is skipped. Segments are joined with NUL so a
    pattern cannot match across two of them.
    """
    command = _drop_prose_heredoc_bodies(command)
    prose = _docker_lifecycle_prose_starts(command)
    return "\0".join(
        _facade()._shell_command_segment(command, start).lower()
        for start, _, _ in _facade()._iter_shell_command_word_spans(command)
        if start not in prose
    )


def _docker_lifecycle_requires_raw_scan(command: str, variant: str) -> bool:
    """Keep shell-carrier code subject to the raw lifecycle scan.

    Shell-carrier arguments are executable code, and extracted ``bash -c`` variants must inherit
    that fact from the source. Policy writes need no key list here: a prose segment that writes a
    file or feeds another command is not prose (``_docker_lifecycle_prose_starts``), and
    ``hermes config set`` is outside the hermes prose subcommands.
    """
    return any(_facade()._contains_shell_carrier(candidate) for candidate in (command, variant))

"""Current-PID self-kill grammar for :mod:`tools.approval_detection`.

Name-based patterns cover ``pkill hermes`` and ``kill $(pgrep ...)``, but no static pattern
can know the runtime PID. Detection finds the command-position ``kill`` words and hands their
deobfuscated argv here; these helpers decide whether that argv sends a terminating signal to
a literal PID. Signal 0 and the list/help/version modes send nothing and stay approval-free;
for kill(1) a long option never does (BSD kill skips it and sends TERM). Pure argv parsing: no
shell scanning, config or approval state.
"""

import os
import re

_PID_RE = re.compile(r"\+?(\d+)")
_LIST_MODE_PREFIXES = ("-l", "-L")
_NON_SENDING_OPTIONS = frozenset({"-l", "-L", "-h", "-V"})
_SIGNAL_OPTIONS = frozenset({"-s", "-n"})
# Options whose operands are values, not PIDs: procps-ng/util-linux `-q <value>`.
_OPTION_OPERAND_COUNTS = {"-q": 1}


def kill_runs_shell_builtin(executable: str, wrappers) -> bool:
    """Whether bash runs this ``kill`` as its builtin rather than exec'ing kill(1).

    *wrappers* are the deobfuscated wrapper words before it in its command. A ``command`` or
    ``builtin`` right before it keeps the builtin; every other wrapper (sudo, env, exec, nohup,
    nice, time, ...) execs a program. Bash's ``time`` keyword would keep the builtin too, but
    ``time`` is the keyword only in some positions (``FOO=1 time`` is time(1)), so it is read
    like the other wrappers.
    """
    return "/" not in executable and (not wrappers or os.path.basename(wrappers[-1]).lower() in ("command", "builtin"))


def _literal_pid(word: str) -> str | None:
    """A literal PID operand normalized the way kill(1) reads it (leading zeroes, ``+``)."""
    match = _PID_RE.fullmatch(word)
    # Normalize as a string: a hostile operand can exceed Python's int-conversion digit limit.
    return (match.group(1).lstrip("0") or "0") if match else None


def _kill_option(arg: str, following: list[str]) -> tuple[str | None, int] | None:
    """Parse one kill option -> ``(signal or None, words consumed)``; ``None`` means this
    invocation sends no signal at all."""
    if arg.startswith("--"):
        # Only bash's builtin gets here, and it has no GNU long options: it errors before a PID.
        return None
    if arg in _NON_SENDING_OPTIONS or arg.startswith(_LIST_MODE_PREFIXES):
        return None
    if arg in _SIGNAL_OPTIONS:
        return (following[0].strip(), 2) if following else None
    if re.fullmatch(r"-[sn].+", arg):
        return arg[2:], 1
    if arg in _OPTION_OPERAND_COUNTS:
        return None, 1 + _OPTION_OPERAND_COUNTS[arg]
    # Traditional compact signal forms: -9, -TERM, -0.
    return arg[1:], 1


def kill_argv_signals_pid(argv: list[str], pid: str, *, shell_builtin: bool) -> bool:
    """Whether ``kill`` with this deobfuscated *argv* sends a non-zero signal to literal *pid*."""
    signal, pids, options, index, term = None, set(), True, 0, False
    while index < len(argv):
        arg = argv[index].strip()
        if options and arg == "--":
            options, index = False, index + 1
            continue
        if options and arg.startswith("--") and not shell_builtin:
            # BSD kill(1) (macOS) skips a `--`-prefixed word and sends the default TERM, so whatever
            # GNU kill makes of a long option (`--signal=0`, `--list`), TERM counts. A separate value
            # (`--signal 0`) is then an operand.
            term, index = True, index + 1
            continue
        if options and arg.startswith("-") and len(arg) > 1:
            parsed = _kill_option(arg, argv[index + 1:index + 2])
            if parsed is None:
                return False
            if parsed[0] is not None:
                # Bash's builtin takes one signal: later words (`--help`, `-0`) are operands.
                signal, options = parsed[0], not shell_builtin
            index += parsed[1]
            continue
        literal = _literal_pid(arg)
        if literal is not None:
            pids.add(literal)
            # Options end at the first PID (bash's builtin and kill(1) builds that do not permute):
            # `kill PID -0` or `/bin/kill PID -l` has already signalled PID.
            options = False
        index += 1
    number = (signal or "TERM").strip().upper().removeprefix("SIG").removeprefix("+")
    return pid in pids and (term or not (number and not number.strip("0")))

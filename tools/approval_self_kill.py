"""Current-PID self-kill grammar for :mod:`tools.approval_detection`.

Name-based patterns cover ``pkill hermes`` and ``kill $(pgrep ...)``, but no static pattern
can know the runtime PID. Detection finds the command-position ``kill`` words and hands their
deobfuscated argv here; these helpers decide whether that argv sends a terminating signal to
a literal PID. Signal 0 and the list/table/help/version modes send nothing and stay
approval-free. Pure argv parsing: no shell scanning, config or approval state.
"""

import os
import re

_PID_RE = re.compile(r"\+?(\d+)")
_LIST_MODE_PREFIXES = ("-l", "-L", "--list=")
_NON_SENDING_OPTIONS = frozenset({"-l", "--list", "-L", "--table", "-h", "--help", "-V", "--version"})
_SIGNAL_OPTIONS = frozenset({"-s", "-n"})
# procps-ng options whose operands are values, not PIDs: `--queue <value>`, `--timeout <ms> <signal>`.
_OPTION_OPERAND_COUNTS = {"-q": 1, "--queue": 1, "--timeout": 2}


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


def _kill_option(arg: str, following: list[str], shell_builtin: bool) -> tuple[str | None, int] | None:
    """Parse one kill option -> ``(signal or None, words consumed)``; ``None`` means this
    invocation sends no signal at all."""
    if shell_builtin and arg.startswith("--"):
        # Bash's builtin has no GNU long options: it errors before reaching a PID, while
        # procps kill(1) accepts --signal/--queue/--timeout and their equals forms.
        return None
    if arg in _NON_SENDING_OPTIONS or arg.startswith(_LIST_MODE_PREFIXES):
        return None
    if arg in _SIGNAL_OPTIONS:
        return (following[0].strip(), 2) if following else None
    if arg == "--signal":
        # A separate value is not trusted as the signal: a kill(1) without long options skips the
        # option and sends TERM, so the option counts as an unknown signal and VALUE as an operand.
        return arg, 1
    if arg.startswith("--signal="):
        return arg.split("=", 1)[1], 1
    if re.fullmatch(r"-[sn].+", arg):
        return arg[2:], 1
    if arg in _OPTION_OPERAND_COUNTS:
        return None, 1 + _OPTION_OPERAND_COUNTS[arg]
    if arg.startswith("--queue="):
        return None, 1
    # Traditional compact signal forms: -9, -TERM, -0.
    return arg[1:], 1


def kill_argv_signals_pid(argv: list[str], pid: str, *, shell_builtin: bool) -> bool:
    """Whether ``kill`` with this deobfuscated *argv* sends a non-zero signal to literal *pid*."""
    signal, pids, options, index = None, set(), True, 0
    while index < len(argv):
        arg = argv[index].strip()
        if options and arg == "--":
            options, index = False, index + 1
            continue
        if options and arg.startswith("-") and len(arg) > 1:
            parsed = _kill_option(arg, argv[index + 1:index + 2], shell_builtin)
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
    return pid in pids and not (number and not number.strip("0"))

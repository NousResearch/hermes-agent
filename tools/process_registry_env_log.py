"""Sandbox background jobs for ProcessRegistry's non-local backends: the launch wrapper that
writes the pid, log and exit files, and the incremental reads the poller makes of them."""

import re
import shlex
from typing import Any, NamedTuple

_HEADER = re.compile(r"([0-9]{1,20}) ([0-9]{1,20}) (?:(run)|exit:([0-9]{0,9}))")


def env_job_paths(temp_dir: str, session_id: str) -> tuple[str, str, str]:
    """The log, pid and exit files of one sandbox background job."""
    stem = f"{temp_dir}/hermes_bg_{session_id}"
    return f"{stem}.log", f"{stem}.pid", f"{stem}.exit"


#: ``$0`` of the supervisor shell: names the job in ``ps`` and marks the launch exec.
LAUNCH_NAME = "hermes-bg"


def env_launch_command(temp_dir: str, command: str, log_path: str, pid_path: str, exit_path: str) -> str:
    """Shell command that starts *command* in the background and prints the job's PID.

    ``set -m`` (bash job control, which needs no tty and no ``setsid``/``pkill``
    in the image) puts the job in a new process group led by a supervisor shell,
    so the printed PID is also the group id: a kill signals the supervisor, the
    command and every ordinary descendant together. The supervisor traps TERM, so
    it outlives a group SIGTERM until the command returns, then records the
    command's exit code (143 after a kill) in *exit_path*; the poller's liveness
    check on that PID keeps reporting the job while its command runs. A
    descendant that leaves the group (``setsid``, a daemon's double fork) is
    outside it."""
    q = shlex.quote
    supervisor = (
        'trap : TERM; printf \'%s\\n\' "$$" > "$2"; bash -lc "$1"; rc=$?; printf \'%s\\n\' "$rc" > "$3"')
    return (
        f"mkdir -p {q(temp_dir)} && {{ "
        f"set -m; nohup bash -c {q(supervisor)} {LAUNCH_NAME} {q(command)} {q(pid_path)} {q(exit_path)} "
        f"> {q(log_path)} 2>&1 < /dev/null & "
        f"for _hermes_wait in {{1..100}}; do test -s {q(pid_path)} && break; sleep 0.01; done; "
        f"cat {q(pid_path)}; }}")


class LogDelta(NamedTuple):
    """One parsed ``log_delta_command`` reply. ``exit_code`` is ``None`` while the
    process runs; once it has exited it is the recorded code, or -1 when the
    wrapper recorded none."""
    size: int
    offset: int
    exit_code: int | None
    delta: str


def _alive_probe(quoted_pid_path: str, group: bool) -> str:
    """``kill -0`` on the job's supervisor, or on its whole process group (``-<pgid>``) once
    a kill has begun, so a member that outlives the supervisor keeps the job running."""
    sign = "-" if group else ""
    return f'kill -0 "{sign}$(cat {quoted_pid_path} 2>/dev/null)" 2>/dev/null'  # no "--": dash rejects it


def log_delta_command(quoted_log_path: str, offset: int, quoted_pid_path: str, quoted_exit_path: str,
                      *, group: bool = False) -> str:
    """Shell command that reads only the log bytes written since ``offset``
    (``cat``-ing the whole file every poll re-sends all output over docker/SSH)
    and reports the process status in the same exec (sandboxes rate-limit exec
    starts, so one poll must cost one).

    Prints one header line ``"<size> <offset> <status>"`` then the bytes in
    [offset, size). The status is ``run`` while ``_alive_probe`` succeeds, else ``exit:<code>`` (``exit:`` with no digits when the exit file
    is missing or unreadable). It sits in the header line, ahead of the delta, so
    log content cannot forge it.

    The status is taken BEFORE the size: the wrapper shell writes the exit file
    only after the command it wraps has returned, so once the wrapper is gone its
    output is in the log, and the delta in that reply is the last of it.
    The size is read before the tail and the tail cut at that same size, so a
    growing file never sends a byte twice; a file that shrank was rotated or
    truncated, so the offset drops to 0 and the reader starts over.

    While the process runs, the window end is pulled back to a UTF-8 character
    boundary (the backend decodes each ``execute()`` result on its own, so a
    straddling multibyte char would become U+FFFD and break watch patterns at the
    seam): up to 3 trailing continuation bytes are held for the next poll and the
    header reports the trimmed size. After an exit no poll follows, so nothing is
    held back and a truncated trailing sequence is delivered (decoded as U+FFFD)."""
    return (
        f'if {_alive_probe(quoted_pid_path, group)}; then T=run; '
        f"else T=\"exit:$(cat {quoted_exit_path} 2>/dev/null | tr -dc '0-9')\"; fi; "
        f"O={offset}; "
        f"S=$({{ wc -c < {quoted_log_path}; }} 2>/dev/null | tr -dc '0-9'); "
        f"S=${{S:-0}}; "
        f'if [ "$S" -lt "$O" ]; then O=0; fi; '
        f'case "$T" in exit:*) ;; *) '
        # Scan back up to 3 continuation bytes (octal 200-277) to the lead byte; if
        # the lead's declared length (3xx=2, 34x-35x=3, 36x-37x=4) exceeds the bytes
        # present, trim to before it. Complete sequences and ASCII tails untouched.
        f'N=0; P=$S; while [ "$P" -gt "$O" ] && [ "$N" -lt 3 ]; do '
        f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
        f'case "$B" in 2[0-7][0-7]) P=$((P-1)); N=$((N+1));; *) break;; esac; done; '
        f'if [ "$N" -gt 0 ] || [ "$P" -eq "$S" ]; then '
        f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
        f'case "$B" in 3[0-3][0-7]) L=2;; 3[4-5][0-7]) L=3;; 3[6-7][0-7]) L=4;; *) L=1;; esac; '
        f'if [ "$L" -gt $((N+1)) ]; then S=$((P-1)); fi; fi; '
        f"esac; "
        f'echo "$S $O $T"; '
        f'if [ "$S" -gt "$O" ]; then '
        f"tail -c +$((O+1)) {quoted_log_path} 2>/dev/null | head -c $((S-O)); fi"
    )


def parse_log_delta_reply(raw: str) -> LogDelta | None:
    """Parse a ``log_delta_command`` reply. ``None`` when it has no usable header
    (command failed, shell missing a tool): the caller drops that reply rather
    than act on a half-read value."""
    header, _, delta = raw.partition("\n")
    # ASCII digits only (``\d`` and ``str.isdigit`` accept other scripts), bounded
    # so ``int()`` cannot hit the interpreter's digit limit on a mangled reply.
    match = _HEADER.fullmatch(header.strip())
    if match is None:
        return None
    size, offset, running, code = match.groups()
    exit_code = None if running else int(code) if code else -1
    return LogDelta(int(size), int(offset), exit_code, delta)


def fallback_exit_code(env: Any, quoted_pid_path: str, quoted_exit_path: str, group: bool = False) -> int | None:
    """Separate liveness and exit-code reads for a reply the combined command
    could not deliver, so a backend that mangles it still sees the exit.
    ``None`` while the process looks alive, else the recorded code (-1 if none)."""
    check = env.execute(
        f"{_alive_probe(quoted_pid_path, group)}; echo $?", timeout=5)
    check_output = check.get("output", "").strip()
    if not check_output or check_output.splitlines()[-1].strip() == "0":
        return None
    exit_str = env.execute(f"cat {quoted_exit_path} 2>/dev/null", timeout=5).get("output", "").strip()
    try:
        return int(exit_str.splitlines()[-1].strip())
    except (ValueError, IndexError):
        return -1

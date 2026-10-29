"""Why a background process has no writable stdin, phrased as the fix the agent can act on."""
import importlib.util
import sys

_NON_INTERACTIVE = "Pass input non-interactively instead (arguments, a here-doc, or `printf ... | cmd`)."


def _pty_available() -> bool:
    return importlib.util.find_spec("winpty" if sys.platform == "win32" else "ptyprocess") is not None


def _no_pty_reason(session) -> str:
    """Why a pty=true relaunch of this local pipe-mode session would not help, or "" when it would.

    ``spawn_local`` records a pty=true request that fell back to pipe mode, so the advice never
    repeats a request that already failed: an installed PTY library is not proof a PTY spawns."""
    if session.pty_fallback:
        return f"pty=true was requested but {session.pty_fallback}, so it runs without stdin"
    if session.output_log:
        return "background processes here run in an isolated worker that cannot give them a pseudo-terminal"
    if not _pty_available():
        return "the PTY library (ptyprocess/pywinpty) is not installed"
    return ""


def already_exited_error(session) -> str:
    """An exited pipe-mode process had no stdin; if it read stdin it got EOF at once. Say so, or the agent
    retries blind."""
    if session.process is None or session.process.stdin is not None:
        return "Process has already finished"
    if reason := _no_pty_reason(session):
        return (f"Process has already finished. It ran without stdin ({reason}), so if it read stdin, it "
                f"got end-of-file at once. {_NON_INTERACTIVE}")
    return ("Process has already finished. It was started without stdin (terminal background=true without "
            "pty=true), so if it read stdin, it got end-of-file at once. Relaunch it with "
            "terminal(background=true, pty=true) to send input.")


def stdin_unavailable_error(session) -> str:
    """Error for process_manage write/submit on a session without a stdin pipe or PTY.

    A local pipe-mode spawn connects stdin to /dev/null by design, so only a PTY relaunch can take
    input, and only where ``spawn_local`` can give one; a non-local backend has no live stdin."""
    if session.process is None:
        return ("Process stdin not available: it runs on a non-local terminal backend (or was recovered "
                f"after a restart), which has no live stdin. {_NON_INTERACTIVE}")
    if reason := _no_pty_reason(session):
        return f"This process has no stdin: {reason}. Relaunching with pty=true would not help. {_NON_INTERACTIVE}"
    return ("This process was started without stdin (terminal background=true without pty=true). "
            "Relaunch it with terminal(background=true, pty=true) to send input.")

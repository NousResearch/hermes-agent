"""Namespace ownership for Docker exec commands, independent of the client PID."""

import logging
import shlex
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DockerForeground:
    env: object
    directory: str = field(init=False)
    adopted: bool = False

    def __post_init__(self):
        self.directory = f"{self.env.get_temp_dir().rstrip('/')}/.hermes-foreground-{uuid.uuid4().hex}"

    def wrap(self, command: str) -> str:
        q = shlex.quote
        child = (f"printf '%s\\n' $$ > {q(self.directory + '/pid')}; "
                 f"test ! -e {q(self.directory + '/cancel')} || exit 130; "
                 f"exec bash -c {q(command)}")
        # Bash job control gives the child its own namespace-local process group, even in images
        # without util-linux/setsid. The client shell waits so its exit code is the command's.
        return (f"(umask 077; mkdir -p {q(self.directory)}) || exit $?; "
                f"set -m; bash -c {q(child)} & p=$!; wait \"$p\" 2>/dev/null")

    def control(self, command: str) -> dict:
        proc = self.env._run_bash_untracked(command, timeout=10)
        try:
            stdout, _ = proc.communicate(timeout=10)
            return {"output": stdout or "", "returncode": proc.returncode}
        except BaseException:
            proc.kill()
            proc.wait()
            raise

    def pid(self) -> int:
        result = self.control(f"cat {shlex.quote(self.directory + '/pid')}")
        if result["returncode"] != 0:
            raise RuntimeError("Docker foreground PID is not available")
        return int(result["output"].strip())

    def kill(self) -> None:
        q = shlex.quote
        # The cancel fence also covers an interrupt before the namespace child has published its PID.
        result = self.control(
            f"mkdir -p {q(self.directory)} && touch {q(self.directory + '/cancel')} || exit 1; "
            f"p=$(cat {q(self.directory + '/pid')} 2>/dev/null); "
            'if [ -n "$p" ]; then '
            'kill -KILL -- -"$p" 2>/dev/null || :; '
            'for attempt in {1..40}; do alive=; '
            'for f in /proc/[0-9]*/stat; do '
            'read -r s < "$f" 2>/dev/null || continue; s=${s##*) }; set -- $s; '
            '[ "$3" = "$p" ] && [ "$1" != Z ] && alive=1; done; '
            '[ -z "$alive" ] && exit 0; sleep 0.05; done; exit 1; fi')
        if result["returncode"] != 0:
            raise RuntimeError("Docker foreground process group termination could not be verified")

    def cleanup(self) -> None:
        try:
            self.control(f"rm -rf -- {shlex.quote(self.directory)}")
        except Exception:
            # A late container loss must not turn a completed command into a retry.
            logger.warning("Could not remove Docker foreground ownership record", exc_info=True)


def adopt_foreground(registry, record: DockerForeground, proc, *, command, cwd,
                     task_id, owner_task_id, session_key, output_so_far):
    """Keep draining the original exec client, but manage the owned namespace group on kill."""
    session = registry._new_session(command, task_id, owner_task_id, session_key, cwd,
                                    env_ref=record.env, pid_scope="sandbox")
    session.pid = record.pid()
    session.process = proc
    session.host_start_time = registry._safe_host_start_time(proc.pid)
    session.notify_on_complete = True
    session.append_output(output_so_far)

    def read(session):
        try:
            registry._reader_loop(session)
        finally:
            record.cleanup()
            record.env._foreground_commands.pop(id(proc), None)

    record.adopted = True
    try:
        registry._track_started(session, read, f"proc-reader-{session.id}")
    except BaseException:
        record.adopted = False
        raise
    return session

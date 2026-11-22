"""Namespace ownership for Docker exec commands, independent of the client PID."""

import math
import shlex
import uuid
from dataclasses import dataclass, field

@dataclass
class DockerForeground:
    env: object
    directory: str = field(init=False)
    adopted: bool = False
    term_grace: float = 2.0

    def __post_init__(self):
        self.directory = f"{self.env.get_temp_dir().rstrip('/')}/.hermes-foreground-{uuid.uuid4().hex}"

    def wrap(self, command: str) -> str:
        q = shlex.quote
        child = ("read -r s < /proc/$$/stat || exit 1; s=${s##*) }; set -- $s; "
                 f"printf '%s %s\\n' $$ \"${{20}}\" > {q(self.directory + '/pid')}; "
                 f"test ! -e {q(self.directory + '/cancel')} || exit 130; "
                 f"exec bash -c {q(command)}")
        # Bash job control gives the child its own namespace-local process group, even in images
        # without util-linux/setsid. The client shell waits so its exit code is the command's.
        return (f"(umask 077; mkdir -p {q(self.directory)}) || exit $?; "
                f"set -m; bash -c {q(child)} & p=$!; wait \"$p\" 2>/dev/null; rc=$?; "
                f"rm -rf -- {q(self.directory)} 2>/dev/null; exit \"$rc\"")

    def control(self, command: str, *, timeout: float = 10) -> dict:
        proc = self.env._run_bash_untracked(command, timeout=timeout)
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            return {"output": stdout or "", "returncode": proc.returncode}
        except BaseException:
            proc.kill()
            proc.wait()
            raise

    def pid(self) -> int:
        result = self.control(f"cat {shlex.quote(self.directory + '/pid')}")
        if result["returncode"] != 0:
            raise RuntimeError("Docker foreground PID is not available")
        return int(result["output"].split()[0])

    def kill(self, *, force: bool = False) -> None:
        q = shlex.quote
        steps = math.ceil(self.term_grace / 0.05)
        escalation = (f'wait_group {steps} && exit 0; '
                      'kill -KILL -- -"$p" 2>/dev/null || :; wait_group 40'
                      if steps else 'exit 0')
        # The cancel fence also covers an interrupt before the namespace child has published its PID.
        result = self.control(
            f"mkdir -p {q(self.directory)} && touch {q(self.directory + '/cancel')} || exit 1; "
            f"read -r p born < {q(self.directory + '/pid')} 2>/dev/null || :; "
            'if [ -n "$p" ]; then '
            'if read -r s < "/proc/$p/stat" 2>/dev/null; then '
            's=${s##*) }; set -- $s; [ "${20}" = "$born" ] || exit 0; fi; '
            'alive_group() { '
            'for f in /proc/[0-9]*/stat; do '
            'read -r s < "$f" 2>/dev/null || continue; s=${s##*) }; set -- $s; '
            '[ "$3" = "$p" ] && [ "$1" != Z ] && return 0; done; return 1; }; '
            'wait_group() { local attempt; for ((attempt=0; attempt<$1; attempt++)); do '
            'alive_group || return 0; sleep 0.05; done; ! alive_group; }; '
            f'kill -{"KILL" if force else "TERM"} -- -"$p" 2>/dev/null || :; '
            f'{"wait_group 40" if force else escalation}; else exit 0; fi',
            timeout=10 if force else max(10, self.term_grace + 5))
        if result["returncode"] != 0:
            raise RuntimeError("Docker foreground process group termination could not be verified")

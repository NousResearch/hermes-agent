"""A long-lived process INSIDE a terminal backend with both stdio pipes open.

``BaseEnvironment.execute`` is request/response: it captures output and returns. The desktop that
follows the terminal backend needs three things that are not: the RFB byte stream between the
Desktop pane and the sandbox's Xvnc, cua-driver's MCP-over-stdio server, and agent-browser
invocations whose daemon must outlive the call. All three are "spawn argv in the sandbox, keep
stdin/stdout as pipes". Each spawn-per-call backend has a local argv prefix that gives exactly that
(``docker exec -i``, ``ssh``, ``apptainer exec``); this module is the one place that knows it.

SDK backends (modal, daytona, vercel) have no argv prefix: ``exec_prefix`` returns None there and
the desktop stays on the gateway host (or refuses, per ``bot_desktop.placement``).
"""
from __future__ import annotations

import shlex
import subprocess
from typing import Optional, Sequence

from tools.environments.base import BaseEnvironment


def exec_prefix(env: BaseEnvironment, *, user: Optional[str] = None, interactive: bool = True) -> Optional[list[str]]:
    """Local argv that runs its remainder inside ``env``, or None for backends without one.

    ``user`` selects the sandbox-side account (docker only; ssh runs as the configured login, apptainer as
    the caller). ``interactive`` keeps stdin open (``docker exec -i``); ssh and apptainer always do.
    """
    from tools.environments.docker import DockerEnvironment
    from tools.environments.singularity import SingularityEnvironment
    from tools.environments.ssh import SSHEnvironment

    if isinstance(env, DockerEnvironment):
        container = getattr(env, "_container_id", None)
        if not container:
            return None
        argv = [env._docker_exe, "exec"]
        if interactive:
            argv.append("-i")
        if user:
            argv += ["-u", user]
        return argv + [container]
    if isinstance(env, SSHEnvironment):
        return env._build_ssh_command()
    if isinstance(env, SingularityEnvironment):
        if not getattr(env, "_instance_started", False):
            return None
        return [env.executable, "exec", f"instance://{env.instance_id}"]
    return None


def supports_streams(env: BaseEnvironment) -> bool:
    return exec_prefix(env, interactive=True) is not None


def remote_argv(prefix: Sequence[str], argv: Sequence[str], *, env: Optional[dict] = None) -> list[str]:
    """``prefix`` + a ``bash -c`` that exports ``env`` and execs ``argv``. Everything is one shell word so
    ssh (which joins its arguments with spaces and re-parses them remotely) and docker (which does not)
    agree on what runs."""
    exports = " ".join(f"export {k}={shlex.quote(v)};" for k, v in (env or {}).items())
    script = f"{exports} exec {' '.join(shlex.quote(a) for a in argv)}"
    return [*prefix, "bash", "-c", script]


def open_stream(env: BaseEnvironment, argv: Sequence[str], *, child_env: Optional[dict] = None,
                user: Optional[str] = None, stderr=subprocess.DEVNULL) -> subprocess.Popen:
    """Spawn ``argv`` inside ``env`` with stdin and stdout as pipes (bytes). Raises ``RuntimeError`` for a
    backend that cannot host a stream."""
    prefix = exec_prefix(env, user=user, interactive=True)
    if prefix is None:
        raise RuntimeError(f"{type(env).__name__} cannot host a long-lived stdio stream")
    return subprocess.Popen(  # windows-footgun: ok — the prefix is a Linux sandbox's client (docker/ssh/apptainer)
        remote_argv(prefix, argv, env=child_env), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
        close_fds=True)


def run_in(env: BaseEnvironment, argv: Sequence[str], *, child_env: Optional[dict] = None, user: Optional[str] = None,
           timeout: float = 30.0, stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
    """One short command inside ``env`` with captured bytes output (probes: ``command -v``, ``cat env``)."""
    prefix = exec_prefix(env, user=user, interactive=stdin is not None)
    if prefix is None:
        raise RuntimeError(f"{type(env).__name__} cannot exec argv")
    return subprocess.run(  # windows-footgun: ok — Linux sandbox client
        remote_argv(prefix, argv, env=child_env), input=stdin, capture_output=True, timeout=timeout, check=False)

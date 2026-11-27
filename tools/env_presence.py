"""env_presence — name-only environment-variable presence checks.

Returns PRESENT/ABSENT for each requested variable name, never a value.
Supports three targets:

- the local NiPoGi process environment (``target=None`` or ``"local"``);
- a named Docker container's environment (``target="docker:<name>"``);
- a systemd unit's configured Environment= metadata (``target="systemd:<unit>"``,
  read via ``systemctl show`` — already in the read-only allowlist).

No value is ever read into this tool's return value — only whether the
name appears as a key in the target's environment.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from tools.read_only_command_guard import run_read_only_guard
from tools.secret_redaction import redact_text


class EnvPresenceError(Exception):
    pass


@dataclass
class EnvPresenceResult:
    target: str
    presence: dict  # name -> "PRESENT" | "ABSENT"
    error: str | None = None


def _run_guarded(command: str, timeout: int = 10) -> str:
    """Run a command through the read-only guard, then execute it. Never
    used for arbitrary caller-supplied commands — only the two fixed
    shapes this module itself constructs (docker inspect / systemctl
    show), so there is no injectable surface here beyond the target name,
    which is validated separately (see _validate_target_name)."""
    guard = run_read_only_guard(command)
    if not guard.allowed:
        raise EnvPresenceError(f"internal command construction was rejected by the read-only guard: {guard.reason}")
    proc = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout,  # noqa: S602 — guarded above
    )
    if proc.returncode != 0:
        # Never trust that a failing subprocess's stderr is itself
        # secret-free (a mistyped target, a wrapped error message, etc.
        # could echo back something sensitive) — redact same as any other
        # output before it can reach a caller.
        raise EnvPresenceError(redact_text(proc.stderr.strip()) or f"command exited {proc.returncode}")
    return proc.stdout


_SAFE_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)


def _validate_target_name(name: str, kind: str) -> None:
    """Docker container names and systemd unit names share a narrow safe
    character set; reject anything else outright rather than trying to
    shell-quote it — this tool never needs a name containing spaces,
    quotes, or shell metacharacters, so refusing them is strictly safer
    than attempting to escape them correctly."""
    if not name or any(ch not in _SAFE_NAME_CHARS for ch in name):
        raise EnvPresenceError(f"'{name}' is not a valid {kind} name (unsafe characters)")


def _local_env_names() -> set:
    import os

    return set(os.environ.keys())


def _docker_env_names(container: str) -> set:
    _validate_target_name(container, "container")
    output = _run_guarded(f"docker inspect {container}")
    import json

    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise EnvPresenceError(f"docker inspect output was not valid JSON: {exc}") from exc
    if not data:
        raise EnvPresenceError(f"container '{container}' not found")
    env_list = data[0].get("Config", {}).get("Env", [])
    return {entry.split("=", 1)[0] for entry in env_list if "=" in entry}


def _systemd_env_names(unit: str) -> set:
    _validate_target_name(unit, "unit")
    output = _run_guarded(f"systemctl show {unit} --property=Environment")
    # Output shape: "Environment=FOO=bar BAZ=qux" (space-separated,
    # values may themselves be absent if the unit sets none).
    names = set()
    for line in output.splitlines():
        if not line.startswith("Environment="):
            continue
        rest = line[len("Environment="):]
        for pair in rest.split():
            if "=" in pair:
                names.add(pair.split("=", 1)[0])
    return names


def env_presence(names: list, target: str | None = None) -> EnvPresenceResult:
    """Check presence (never value) of each name in ``names`` against
    ``target``.

    ``target`` is one of:
      - None / "local"        — this process's own environment
      - "docker:<container>"  — the named container's configured env
      - "systemd:<unit>"      — the named systemd unit's Environment=
    """
    target = target or "local"
    try:
        if target == "local":
            available = _local_env_names()
        elif target.startswith("docker:"):
            available = _docker_env_names(target[len("docker:"):])
        elif target.startswith("systemd:"):
            available = _systemd_env_names(target[len("systemd:"):])
        else:
            return EnvPresenceResult(target=target, presence={}, error=f"unknown target scheme: '{target}'")
    except EnvPresenceError as exc:
        return EnvPresenceResult(target=target, presence={}, error=str(exc))

    presence = {name: ("PRESENT" if name in available else "ABSENT") for name in names}
    return EnvPresenceResult(target=target, presence=presence)

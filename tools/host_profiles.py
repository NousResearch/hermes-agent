"""Minimum host-profile abstraction so a Rob operator tool call can target
"nipogi" (local — Rob already runs on NiPoGi) or, later, "workstation"
(Nicolas's dev machine) without ``terminal.backend`` being the only
selector (that config key is a single GLOBAL setting shared by every
tool call, which cannot express "inspect NiPoGi and the workstation in
the same session").

This deliberately does NOT reimplement ``tools/environments/ssh.py``'s
``SSHEnvironment`` — that class already exists, is already wired into
``terminal_tool.py``, and already owns the harder lifecycle concerns
(connection setup, remote-home detection, bulk file sync, cleanup) that
this first Rob slice has no need to duplicate. What's genuinely missing
is per-call host SELECTION, not a new transport — so this module adds
only that: a small profile registry plus a direct, narrow SSH invocation
for the remote case, using the same target model (host/user, existing
key-based auth) ``SSHEnvironment`` already assumes, rather than importing
its private, lifecycle-heavy internals for a single bounded command.

The "workstation" profile is intentionally left disabled/documented, per
the P0/P1 spec's own instruction ("If the workstation profile cannot be
safely configured now, implement the abstraction and leave the concrete
remote profile disabled/documented") — enabling it is a follow-up
mutation (adding real connection details), not something this module
invents a default for.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


@dataclass
class HostProfile:
    name: str
    enabled: bool
    disabled_reason: str = ""
    _ssh_target: str | None = field(default=None, repr=False)  # "user@host" when remote

    def run(self, command: str, timeout: int) -> subprocess.CompletedProcess:
        if not self.enabled:
            raise RuntimeError(f"host profile '{self.name}' is disabled: {self.disabled_reason}")
        if self._ssh_target is None:
            return subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout,  # noqa: S602 — caller already ran the read-only guard
            )
        return subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=6",
                self._ssh_target,
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )


_PROFILES = {
    "nipogi": HostProfile(name="nipogi", enabled=True),
    "local": HostProfile(name="local", enabled=True),  # alias — Rob's own current host is NiPoGi
    "workstation": HostProfile(
        name="workstation",
        enabled=False,
        disabled_reason=(
            "no SSH target configured yet — set ROB_HOST_WORKSTATION_TARGET "
            "(e.g. 'nicolas@100.x.x.x') to enable, once Nicolas's dev machine "
            "has an authorized key for this operation and the target has been "
            "reviewed. Not enabled by default."
        ),
    ),
}


def get_host_profile(name: str | None) -> HostProfile:
    """Resolve a profile by name; ``None``/unrecognized both fall back to
    'nipogi', matching where Rob's own process already runs — never
    silently resolves to an unconfigured remote target."""
    if not name:
        return _PROFILES["nipogi"]
    profile = _PROFILES.get(name)
    if profile is None:
        return HostProfile(name=name, enabled=False, disabled_reason=f"unknown host profile '{name}'")
    return profile


def register_workstation_profile(ssh_target: str) -> None:
    """Explicit, separate activation step for the workstation profile —
    never called automatically. A caller (an operator script, never Rob
    itself) invokes this once real connection details are ready."""
    _PROFILES["workstation"] = HostProfile(name="workstation", enabled=True, _ssh_target=ssh_target)

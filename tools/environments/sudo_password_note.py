"""One-time log note when a sudo password is sent to a terminal backend other than the local one.

Every backend that runs a command with ``sudo`` receives the password (on stdin or through an
askpass helper), so programs running in that backend can read it there. For the local backend
that is this machine; for container, cloud sandbox, SSH and plugin backends it is another
environment, where the password should be one that is only valid in that environment.
``note_sudo_password_sent`` logs that advice once per process per backend class.
"""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

_noted_backend_types: set[type] = set()
_noted_lock = threading.Lock()

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def backend_label(env: object) -> str:
    """The backend name the user configured: a plugin backend's registered name (stamped as
    ``_hermes_backend_name`` when it is created), else one built from the environment class:
    ``DockerEnvironment`` -> ``docker``, ``VercelSandboxEnvironment`` -> ``vercel_sandbox``,
    ``SSHEnvironment`` -> ``ssh``."""
    stamped = getattr(env, "_hermes_backend_name", None)
    if isinstance(stamped, str) and stamped.strip():
        return stamped.strip()
    name = type(env).__name__
    if name.endswith("Environment") and name != "Environment":
        name = name[: -len("Environment")]
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def note_sudo_password_sent(env: object, sudo_stdin: str | None) -> None:
    """Log a WARNING the first time this process sends a sudo password to a non-local backend
    class. No-op for the local backend, when no password is sent (``sudo_stdin`` is None), or
    when the password is empty. The message never contains the password."""
    if sudo_stdin is None or not sudo_stdin.strip("\n") or getattr(env, "is_local", False):
        return
    backend_type = type(env)
    with _noted_lock:
        if backend_type in _noted_backend_types:
            return
        _noted_backend_types.add(backend_type)
    logger.warning(
        "The sudo password is sent to the %s terminal backend, where programs running in it can "
        "read it. Use a password that is only valid there, or passwordless sudo in that "
        "environment's image, rather than the password of the machine running Hermes.",
        backend_label(env),
    )


def _reset_noted_backends() -> None:
    """Forget which backend types were noted (tests)."""
    with _noted_lock:
        _noted_backend_types.clear()

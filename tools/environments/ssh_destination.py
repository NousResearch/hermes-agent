"""Safe OpenSSH destination formatting.

OpenSSH treats a destination that begins with ``-`` as an option (classic
ProxyCommand / argv smuggling, CWE-88). Callers that build ``ssh``/``scp``
argv from user config must use :func:`format_ssh_destination` and place
``--`` before the destination.
"""

from __future__ import annotations

import os


def format_ssh_destination(host: str, user: str | None = None) -> str:
    """Return ``user@host`` or ``host`` for an OpenSSH destination argument.

    Raises:
        ValueError: if host is empty or host/user starts with ``-``.
    """
    host_s = str(host or "").strip()
    if not host_s:
        raise ValueError("SSH host is empty")
    if host_s.startswith("-"):
        raise ValueError(
            "SSH host must not start with '-' "
            "(OpenSSH would parse it as an option)"
        )
    _reject_leading_dash_ssh_user(user)
    user_s = str(user or "").strip()
    if user_s:
        return f"{user_s}@{host_s}"
    return host_s


def _reject_leading_dash_ssh_user(user: str | None) -> None:
    user_s = str(user or "").strip()
    if user_s.startswith("-"):
        raise ValueError(
            "SSH user must not start with '-' "
            "(OpenSSH would parse it as an option)"
        )


def validate_ssh_identity(host: str | None, user: str | None = None) -> str | None:
    """Validate SSH host/user from config.

    Returns the OpenSSH destination when host is non-empty, else ``None``.
    Always rejects a leading-dash user (even when host is blank) so setup
    cannot persist ``TERMINAL_SSH_USER=-o...`` under ``TERMINAL_ENV=ssh``.
    """
    _reject_leading_dash_ssh_user(user)
    host_s = str(host or "").strip()
    if not host_s:
        return None
    return format_ssh_destination(host_s, user)


def build_ssh_probe_command(
    host: str,
    user: str | None = None,
    *,
    port: str | None = None,
    key: str | None = None,
    connect_timeout: int = 5,
) -> list[str]:
    """Build argv for a BatchMode SSH connectivity probe (``echo ok``).

    Raises ValueError when host/user would be parsed as OpenSSH options.
    Always inserts ``--`` before the destination.
    """
    target = format_ssh_destination(host, user)
    cmd = [
        "ssh",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "BatchMode=yes",
    ]
    if port:
        cmd += ["-p", str(port)]
    if key:
        cmd += ["-i", os.path.expanduser(key)]
    cmd += ["--", target, "echo ok"]
    return cmd

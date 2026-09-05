"""Dependency-free Host and Origin allowlist validation for the CUA bridge."""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from urllib.parse import urlsplit


def _valid_hostname(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    if len(hostname) > 253:
        return False
    labels = hostname.split(".")
    return all(
        label
        and len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )


def _valid_host_entry(value: str) -> bool:
    if not value or any(character.isspace() for character in value) or "*" in value:
        return False
    try:
        parsed = urlsplit(f"//{value}")
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and parsed.hostname is not None
        and _valid_hostname(parsed.hostname)
    )


def _valid_origin_entry(value: str) -> bool:
    if not value or any(character.isspace() for character in value) or "*" in value:
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.username is None
        and parsed.password is None
        and parsed.hostname is not None
        and _valid_hostname(parsed.hostname)
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def validate_security_allowlists(
    allowed_hosts: Sequence[str],
    allowed_origins: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return validated copies or raise before bridge setup can have side effects."""
    if isinstance(allowed_hosts, str) or isinstance(allowed_origins, str):
        raise TypeError("allowed_hosts and allowed_origins must be sequences of strings")

    hosts = list(allowed_hosts)
    origins = list(allowed_origins)
    if not hosts:
        raise ValueError("allowed_hosts must contain at least one host")
    if not all(isinstance(value, str) and _valid_host_entry(value) for value in hosts):
        raise ValueError("allowed_hosts must contain valid host names with optional ports and no wildcards")
    if not all(isinstance(value, str) and _valid_origin_entry(value) for value in origins):
        raise ValueError("allowed_origins must contain valid HTTP(S) origins and no wildcards")
    return hosts, origins

"""Secure-input gates and deferred reference consumption for terminal execution."""
from __future__ import annotations

import re


def validate_secret_refs(stdin_ref, env_refs, *, env_type: str, background: bool) -> None:
    """Reject unsupported transports before allocating a backend or consuming a ref.

    SDK payload modes can stage stdin on disk. Only the existing pipe backends
    are admitted; third-party backends need an explicit audited transport first.
    Background processes have no secure-output path yet, so refs are foreground-only.
    """
    if stdin_ref and (background or env_type not in {"local", "docker", "ssh"}):
        raise ValueError("stdin_secret_ref requires a foreground pipe backend; other backends may embed stdin in command text")
    if env_refs is not None and not isinstance(env_refs, dict):
        raise ValueError("env_secret_refs must be an object")
    if env_refs:
        if any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in env_refs):
            raise ValueError("Invalid env_secret_refs: invalid env var name")
        if background:
            raise ValueError("env_secret_refs require a foreground command; background processes are not supported")
        if env_type not in {"local", "docker"}:
            raise ValueError("env_secret_refs: no out-of-band env channel for this backend")


def consume_secret_refs(stdin_ref, env_refs) -> dict:
    """Called only after command, capability, approval and cwd validation."""
    from agent.secure_input_broker import consume_secret, describe_secret
    kwargs = {}
    refs = ([stdin_ref] if stdin_ref else []) + list((env_refs or {}).values())
    if refs:
        kwargs['secure_output_deadline'] = min(
            (describe_secret(ref) or {}).get('expires_at', 0) for ref in refs)
    if stdin_ref:
        kwargs["stdin_data"] = consume_secret(stdin_ref, consumer="terminal")
    if env_refs:
        kwargs["transient_env"] = {key: consume_secret(ref, consumer="terminal") for key, ref in env_refs.items()}
    return kwargs

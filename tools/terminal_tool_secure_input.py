"""Secure-input gates and deferred reference consumption for terminal execution."""
from __future__ import annotations

import re


def validate_background_secret_output() -> None:
    from agent.runtime_session_store import is_worker_process
    if is_worker_process():
        raise ValueError('secure input background cannot use file-backed worker output; use a foreground pipe')


def validate_secret_refs(stdin_ref, env_refs, *, env_type: str, background: bool) -> None:
    """Reject unsupported transports before allocating a backend or consuming a ref.

    SDK payload modes can stage stdin on disk. Only the existing pipe backends
    are admitted; third-party backends need an explicit audited transport first.
    """
    if stdin_ref and (background or env_type not in {"local", "docker", "ssh"}):
        raise ValueError("stdin_secret_ref requires a foreground pipe backend; other backends may embed stdin in command text")
    if env_refs is not None and not isinstance(env_refs, dict):
        raise ValueError("env_secret_refs must be an object")
    if env_refs:
        if any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in env_refs):
            raise ValueError("Invalid env_secret_refs: invalid env var name")
        if background and env_type != "local":
            raise ValueError("env_secret_refs background commands require the local backend")
        if env_type not in {"local", "docker"}:
            raise ValueError("env_secret_refs: no out-of-band env channel for this backend")
        if background:
            validate_background_secret_output()


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


def redact_notification(event: dict) -> dict:
    """Last egress check, including raw events assembled by gateway watchers."""
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from agent.secure_input_broker import redact_known_secrets
    result = dict(event)
    command = str(result.get("command") or "")
    for key in ("output", "output_preview", "command", "pattern", "message", "handoff_note"):
        value = result.get(key)
        if isinstance(value, str):
            generic = (redact_terminal_output(value, command) if key in {"output", "output_preview"}
                       else redact_sensitive_text(value, code_file=True))
            result[key] = redact_known_secrets(generic)
    return result

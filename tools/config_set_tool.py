"""Opt-in tool for narrowly bounded Hermes configuration changes."""

from __future__ import annotations

import hashlib
import json
from itertools import islice
import math
import re
import threading
from datetime import datetime, timezone
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry


# Keep this list leaf-exact: adding a sibling to config.yaml must never make it
# agent-writable without a separate security review.
WRITABLE_CONFIG_KEYS = frozenset({
    "compression.enabled",
    "compression.threshold",
    "display.show_reasoning",
    "display.skin",
    "display.tool_progress",
    "stt.local.model",
    "tts.deepinfra.voice",
    "tts.edge.voice",
    "tts.elevenlabs.voice_id",
    "tts.gemini.voice",
    "tts.kittentts.voice",
    "tts.minimax.voice_id",
    "tts.mistral.voice_id",
    "tts.openai.voice",
    "tts.xai.voice_id",
})

# Defense in depth. Authorization still requires exact membership above.
DENIED_CONFIG_PREFIXES = (
    "approvals",
    "auxiliary",
    "context_engine",
    "custom_providers",
    "delegation",
    "extra_headers",
    "mcp_servers",
    "model",
    "platform_toolsets",
    "providers",
    "security",
    "skills",
    "terminal",
    "webhook",
)

_BOOL_KEYS = {
    "compression.enabled",
    "display.show_reasoning",
}
# Tokens that the canonical writer (hermes_cli/config.py set_config_value)
# coerces to YAML booleans for keys without a string default in DEFAULT_CONFIG.
# Voice/string leaves have no business accepting these: rejecting them here
# (fail-closed) keeps persistence type-stable even if a future DEFAULT_CONFIG
# drops a voice key's string default or a new writable leaf without a string
# default is added to WRITABLE_CONFIG_KEYS.
_BOOL_LIKE_TOKENS = frozenset({"true", "false", "yes", "no", "on", "off"})
_ENUM_VALUES = {
    "display.tool_progress": frozenset({"off", "new", "all", "verbose", "log"}),
    "stt.local.model": frozenset({"tiny", "base", "small", "medium", "large-v3"}),
}
_VOICE_KEYS = (
    WRITABLE_CONFIG_KEYS
    - _BOOL_KEYS
    - set(_ENUM_VALUES)
    - {
        "compression.threshold",
        "display.skin",
    }
)

_REDACTION_SENTINEL_RE = re.compile(
    r"(?i)(?:\[redacted\]|«redacted(?::[^»]*)?»|\*{3,}|"
    r"(?:sk-|ghp_)[A-Za-z0-9_-]{0,12}\.\.\.[A-Za-z0-9_-]{0,12}|"
    r"sk-(?:<[^>]+>|\*+|placeholder|redacted|changeme|todo|fixme|your[-_ ]?key))"
)
_CREDENTIAL_RE = re.compile(
    r"(?i)^(?:"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{12,}|"
    r"gh[opusr]_[A-Za-z0-9]{20,}|"
    r"(?:xox[baprs]-)\S+|"
    r"bearer\s+\S+|"
    r"bot\d{6,}:[A-Za-z0-9_-]{20,}|"
    r"\d{6,}:[A-Za-z0-9_-]{20,}"
    r")$"
)

_APPLIES = {
    "display.skin": "new_session",
    "display.show_reasoning": "new_session",
    "display.tool_progress": "new_session",
    "compression.enabled": "new_session",
    "compression.threshold": "new_session",
}
_AUDIT_MAX_BYTES = 1_000_000
_AUDIT_MAX_ITEMS = 20
_AUDIT_MAX_STRING = 256
# Maximum length of a primitive (non-string) audit key representation before
# it is replaced by a bounded fingerprint.
_AUDIT_PRIMITIVE_MAX = 64
_CONFIG_MUTATION_LOCK = threading.RLock()


def _is_blacklisted(key: str) -> bool:
    lowered = key.lower()
    return any(
        lowered == prefix or lowered.startswith(prefix + ".")
        for prefix in DENIED_CONFIG_PREFIXES
    )


def _is_whitelisted(key: str) -> bool:
    return key in WRITABLE_CONFIG_KEYS and not _is_blacklisted(key)


def _is_credential_shaped(value: str) -> bool:
    from agent.redact import redact_sensitive_text

    stripped = value.strip()
    return bool(
        stripped
        and (
            _REDACTION_SENTINEL_RE.search(stripped)
            or _CREDENTIAL_RE.fullmatch(stripped)
            or redact_sensitive_text(stripped, force=True, file_read=True) != stripped
        )
    )


def _validate_bool(value: str) -> tuple[str, bool]:
    if value not in {"true", "false"}:
        raise ValueError("expected 'true' or 'false'")
    return value, value == "true"


def _validate_enum(value: str, choices: frozenset[str]) -> tuple[str, str]:
    if value not in choices:
        raise ValueError(f"expected one of: {', '.join(sorted(choices))}")
    return value, value


def _validate_threshold(value: str) -> tuple[str, float]:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("expected a number from 0.50 through 0.95") from exc
    if not math.isfinite(parsed) or not 0.50 <= parsed <= 0.95:
        raise ValueError("expected a number from 0.50 through 0.95")
    return str(parsed), parsed


def _validate_bounded_string(value: str) -> tuple[str, str]:
    if not value or len(value) > 256 or not value.isprintable():
        raise ValueError("expected 1-256 printable characters")
    if value.lower() in _BOOL_LIKE_TOKENS:
        raise ValueError(
            "boolean-like tokens ('true'/'false'/'yes'/'no'/'on'/'off', "
            "case-insensitive) are not accepted for string settings"
        )
    return value, value


def _validate_skin(value: str) -> tuple[str, str]:
    from hermes_cli.skin_engine import list_skins

    names = {entry["name"] for entry in list_skins()}
    if value not in names:
        raise ValueError(f"expected an installed skin: {', '.join(sorted(names))}")
    return value, value


def _validate_value(key: str, value: Any) -> tuple[str, Any]:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    if _is_credential_shaped(value):
        raise ValueError("credential or redaction placeholder values are not accepted")
    if key in _BOOL_KEYS:
        return _validate_bool(value)
    if key in _ENUM_VALUES:
        return _validate_enum(value, _ENUM_VALUES[key])
    if key == "compression.threshold":
        return _validate_threshold(value)
    if key == "display.skin":
        return _validate_skin(value)
    if key in _VOICE_KEYS:
        return _validate_bounded_string(value)
    raise ValueError("key has no value validator")


def _lookup(config: dict[str, Any], key: str) -> Any:
    node: Any = config
    for segment in key.split("."):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def _audit_key(key: Any) -> str:
    """Bounded, deterministic audit representation of a dict key.

    String keys keep the module's redaction + truncation behavior; a truncated
    string key additionally carries a short fingerprint, which reduces
    accidental collisions between distinct truncated keys. Non-string keys are
    explicitly type-tagged so an integer/boolean/tuple key can never
    masquerade as a string key or silently merge with another key. Every key
    representation is bounded to a small maximum length.
    """
    from agent.redact import redact_sensitive_text

    if isinstance(key, str):
        if _is_credential_shaped(key):
            return "«redacted-secret»"
        redacted = redact_sensitive_text(key, force=True, file_read=True)
        if len(redacted) > _AUDIT_MAX_STRING:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
            return f"{redacted[:_AUDIT_MAX_STRING]}…«{digest}»"
        return redacted
    if key is None:
        return "«none»"
    if isinstance(key, bool):
        return f"«bool:{str(key).lower()}»"
    if isinstance(key, (int, float)):
        # Bounded primitive representation. Huge integers must never produce a
        # huge audit string — and str() of a >4300-digit int raises in
        # CPython 3.11+, while unsigned to_bytes() overflows for negatives, so
        # the fingerprint is computed from sign + magnitude bytes.
        if isinstance(key, int):
            magnitude = abs(key)
            if magnitude.bit_length() > 512:
                raw = magnitude.to_bytes(
                    max(1, (magnitude.bit_length() + 7) // 8), "big"
                )
                digest = hashlib.sha256((b"-" if key < 0 else b"+") + raw).hexdigest()[
                    :8
                ]
                return f"«int:<{digest}>»"
        text = f"{key}"
        if len(text) > _AUDIT_PRIMITIVE_MAX:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
            return f"«{type(key).__name__}:<{digest}>»"
        return f"«{type(key).__name__}:{text}»"
    try:
        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:8]
    except Exception:
        digest = "unhashable"
    return f"«{type(key).__name__}:{digest}»"


def _safe_audit_value(value: Any, *, _depth: int = 0) -> Any:
    from agent.redact import redact_sensitive_text

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _is_credential_shaped(value):
            return "«redacted-secret»"
        redacted = redact_sensitive_text(value, force=True, file_read=True)
        if len(redacted) > _AUDIT_MAX_STRING:
            return redacted[:_AUDIT_MAX_STRING] + "…"
        return redacted
    if _depth >= 3:
        return "«truncated»"
    if isinstance(value, dict):
        result = {
            _audit_key(key): _safe_audit_value(item, _depth=_depth + 1)
            for key, item in islice(value.items(), _AUDIT_MAX_ITEMS)
        }
        if len(value) > _AUDIT_MAX_ITEMS:
            result["«truncated»"] = len(value) - _AUDIT_MAX_ITEMS
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _safe_audit_value(item, _depth=_depth + 1)
            for item in value[:_AUDIT_MAX_ITEMS]
        ]
        if len(value) > _AUDIT_MAX_ITEMS:
            result.append(f"«{len(value) - _AUDIT_MAX_ITEMS} more»")
        return result
    return f"<{type(value).__name__}>"


def _audit_log(
    *,
    key: str,
    status: str,
    old_value: Any = None,
    new_value: Any = None,
    applies: str | None = None,
    session_id: str | None = None,
    rollback: str | None = None,
) -> bool:
    with _CONFIG_MUTATION_LOCK:
        try:
            log_dir = get_hermes_home() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": _safe_audit_value(session_id or "unknown"),
                "key": _safe_audit_value(key),
                "status": status,
                "old_value": _safe_audit_value(old_value),
                "new_value": _safe_audit_value(new_value),
                "applies": applies,
            }
            if rollback is not None:
                entry["rollback"] = rollback
            path = log_dir / "config_changes.jsonl"
            if path.exists() and path.stat().st_size >= _AUDIT_MAX_BYTES:
                rotated = log_dir / "config_changes.jsonl.1"
                if rotated.exists():
                    rotated.unlink()
                path.replace(rotated)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False


def _result(**fields: Any) -> str:
    return json.dumps(fields, ensure_ascii=False)


def _contains_key(config: dict[str, Any], key: str) -> bool:
    """True when *key* exists in the raw config (even if its value is null)."""
    node: Any = config
    for segment in key.split("."):
        if not isinstance(node, dict) or segment not in node:
            return False
        node = node[segment]
    return True


def _restore_target_leaf(
    *,
    key: str,
    old_present: bool,
    old_value: Any,
    failed_persisted_present: bool,
    failed_persisted_value: Any,
    config_file_existed_before: bool,
) -> tuple[str, str, str]:
    """Restore exactly the leaf this tool wrote, preserving all other state.

    This targeted rollback re-reads the latest config and only mutates the
    target leaf, avoiding clobbering unrelated changes already visible when
    rollback state is prepared.

    CAS ownership check: the leaf is only touched when the current target state
    still matches the state observed during verification, compared as
    ``(presence, value)`` — ``value`` is only meaningful when ``presence`` is
    True. If another writer changed the same key (or its presence) in between,
    we refuse to overwrite it and fail closed. Note the CAS narrows the window
    for overwriting a same-key update observed before rollback persistence, but
    cannot provide cross-process transactional isolation without a shared
    inter-process lock.

    Returns ``(rollback_flag, status, error)``:
      * ``("restored", "verification_failed", ...)`` — leaf restored, re-verified
      * ``("failed", "rollback_skipped", ...)``       — concurrent state change
        detected (presence or value), nothing overwritten (CAS mismatch)
      * ``("failed", "rollback_failed", ...)``        — restore write raised or
        re-verification failed
    """
    from hermes_cli.config import (
        _set_nested,
        _unset_nested,
        get_config_path,
        read_raw_config,
    )
    from utils import atomic_yaml_write

    def _failed(extra: str = "") -> tuple[str, str, str]:
        return (
            "failed",
            "rollback_failed",
            extra
            or (
                "Configuration persistence could not be verified, and the "
                "original value could not be restored. The configuration file "
                "may be in an inconsistent state; re-check the setting before "
                "retrying."
            ),
        )

    def _cas_skipped() -> tuple[str, str, str]:
        return (
            "failed",
            "rollback_skipped",
            "Configuration persistence could not be verified, and the target "
            "state changed again before rollback could run; the current target "
            "was not overwritten. Manual verification of the current "
            "configuration state is required.",
        )

    try:
        latest = read_raw_config()
    except Exception:
        return _failed()

    latest_present = _contains_key(latest, key)
    if latest_present != failed_persisted_present:
        # Presence changed (key appeared or disappeared) after verification —
        # we cannot claim ownership of the current state.
        return _cas_skipped()
    if failed_persisted_present and _lookup(latest, key) != failed_persisted_value:
        # Same presence, but the value changed after verification.
        return _cas_skipped()

    if old_present:
        _set_nested(latest, key, old_value)
    elif _contains_key(latest, key) and not _unset_nested(latest, key):
        return _failed(
            "Configuration persistence could not be verified, and the target "
            "leaf could not be removed during rollback. The configuration file "
            "may be in an inconsistent state."
        )

    config_path = get_config_path()
    if not config_file_existed_before and latest == {}:
        # The mutation created the file; after removing the leaf nothing
        # remains, so the file should not exist (no leftover "{}").
        try:
            if config_path.exists():
                config_path.unlink()
        except Exception:
            return _failed(
                "Configuration persistence could not be verified, and the "
                "configuration file could not be removed during rollback. The "
                "configuration file may be in an inconsistent state."
            )
    else:
        try:
            atomic_yaml_write(config_path, latest, sort_keys=False)
        except Exception:
            return _failed()

    try:
        restored = read_raw_config()
    except Exception:
        return _failed()
    if old_present:
        ok = _contains_key(restored, key) and _lookup(restored, key) == old_value
    else:
        ok = not _contains_key(restored, key)
    if ok:
        # File-existence re-verification mirrors the rollback branch that ran:
        # an unlinked (originally absent, now empty) file must stay absent;
        # a written file must exist even if its content is empty.
        if not config_file_existed_before and latest == {}:
            ok = not config_path.exists()
        else:
            ok = config_path.exists()
    if not ok:
        return _failed()
    return (
        "restored",
        "verification_failed",
        "Configuration persistence could not be verified; the original value "
        "was restored.",
    )


def _config_set_value_locked(
    key: str, value: Any, *, session_id: str | None = None
) -> str:
    from hermes_cli.config import get_config_path, read_raw_config, set_config_value

    if not isinstance(key, str) or not _is_whitelisted(key):
        audit_logged = _audit_log(
            key=key if isinstance(key, str) else "<non-string>",
            status="denied",
            new_value=value,
            session_id=session_id,
        )
        return _result(
            success=False,
            blocked=True,
            error="This configuration key is not agent-writable.",
            audit_logged=audit_logged,
        )

    try:
        writer_value, expected_value = _validate_value(key, value)
    except ValueError as exc:
        audit_logged = _audit_log(
            key=key,
            status="invalid_value",
            new_value=value,
            session_id=session_id,
        )
        return _result(
            success=False,
            blocked=True,
            error=str(exc),
            audit_logged=audit_logged,
        )

    # Capture only what a failed verification needs to roll back: whether the
    # leaf existed before, its old value, and whether the config file itself
    # existed. We deliberately do NOT snapshot the whole file — a full-file
    # restore could silently drop unrelated concurrent writes.
    original_config = read_raw_config()
    old_present = _contains_key(original_config, key)
    old_value = _lookup(original_config, key)
    config_file_existed_before = get_config_path().exists()
    try:
        set_config_value(key, writer_value)
    except (SystemExit, Exception):
        audit_logged = _audit_log(
            key=key,
            status="error",
            old_value=old_value,
            new_value=value,
            session_id=session_id,
        )
        return _result(
            success=False,
            error="Configuration write failed.",
            audit_logged=audit_logged,
        )

    persisted_config = read_raw_config()
    failed_persisted_present = _contains_key(persisted_config, key)
    failed_persisted_value = _lookup(persisted_config, key)
    if failed_persisted_value != expected_value:
        rollback_flag, status, error = _restore_target_leaf(
            key=key,
            old_present=old_present,
            old_value=old_value,
            failed_persisted_present=failed_persisted_present,
            failed_persisted_value=failed_persisted_value,
            config_file_existed_before=config_file_existed_before,
        )
        audit_logged = _audit_log(
            key=key,
            status=status,
            old_value=old_value,
            new_value=failed_persisted_value,
            session_id=session_id,
            rollback=rollback_flag,
        )
        return _result(
            success=False,
            error=error,
            rollback=rollback_flag,
            audit_logged=audit_logged,
        )

    applies = _APPLIES.get(key, "next_invocation")
    audit_logged = _audit_log(
        key=key,
        status="success",
        old_value=old_value,
        new_value=failed_persisted_value,
        applies=applies,
        session_id=session_id,
    )
    return _result(
        success=True,
        key=key,
        applies=applies,
        requires_process_restart=False,
        audit_logged=audit_logged,
    )


def config_set_value(key: str, value: Any, *, session_id: str | None = None) -> str:
    """Validate and persist one explicitly authorized configuration leaf."""
    with _CONFIG_MUTATION_LOCK:
        return _config_set_value_locked(key, value, session_id=session_id)


CONFIG_SET_TOOL_SCHEMA: dict[str, Any] = {
    "name": "hermes_config_set",
    "description": (
        "Change one explicitly approved, non-secret Hermes configuration leaf. "
        "The tool is operator-enabled and records each attempt in the config audit log."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "enum": sorted(WRITABLE_CONFIG_KEYS),
                "description": "The exact approved configuration leaf to change.",
            },
            "value": {
                "type": "string",
                "description": "The new value; types and allowed domains are validated per key.",
            },
        },
        "required": ["key", "value"],
        "additionalProperties": False,
    },
}


def _config_set_handler(args: dict[str, Any], **kwargs: Any) -> str:
    return config_set_value(
        args.get("key", ""),
        args.get("value", ""),
        session_id=kwargs.get("session_id"),
    )


registry.register(
    name="hermes_config_set",
    toolset="config",
    schema=CONFIG_SET_TOOL_SCHEMA,
    handler=_config_set_handler,
    description="Change an explicitly approved Hermes configuration leaf",
    emoji="⚙️",
)

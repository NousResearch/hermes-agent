"""Vault root resolution, path-safety containment, config validation.

Profile-scoped plugin uses ``get_hermes_home()`` (from hermes_constants) to
locate its state / cache directories. The Vault root is operator-supplied and
must be an existing directory; traversal, symlink escapes, and binary files
are refused. Path containment uses ``Path.resolve()`` and rejects any resolved
path that does not live under the configured Vault root.

Configuration is validated at load: a hard compile cap of 24,000 characters
on the hard ceiling is enforced (the spec says "hard ceiling … not
configurable above this compiled cap"). ``log_raw_query_terms`` must remain
``false`` (the spec says it must remain false; we refuse at load time if an
operator tries to enable it).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


HARD_CEILING_COMPILED_CAP = 24_000
DEFAULT_HARD_CEILING = 24_000
DEFAULT_BUDGET = 12_000
LARGE_FILE_CHARS = 20_000
CANDIDATE_LIMIT = 20
MAX_PRIMARY_EXTRACTS = 3
MAX_EXPANSION_EXTRACTS = 2
MAX_RANGE_LINES = 120
MAX_RANGE_CHARS = 8_000

_VALID_MODES = ("enforce", "audit", "off")


class VaultConfigError(ValueError):
    """Invalid configuration at plugin load time."""


class VaultPathError(ValueError):
    """A path failed containment checks (../, symlink escape, binary, etc.)."""


def resolve_vault_root(raw: str) -> Path:
    """Resolve a Vault root string to an absolute Path.

    Rejects empty strings, non-existent paths, and file (non-directory) paths.
    The result is resolved (symlinks followed) so containment checks downstream
    can compare against the canonical root.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise VaultConfigError("vault_root must be a non-empty string")
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise VaultConfigError(f"vault_root does not exist: {raw!r}")
    if not p.is_dir():
        raise VaultConfigError(f"vault_root is not a directory: {raw!r}")
    return p


def _is_probably_binary(path: Path) -> bool:
    """A small, fast binary sniff — read first 4096 bytes, look for NUL bytes.

    This is sufficient for our use case (refuse to extract evidence from
    clearly binary files). We deliberately do not import chardet.
    """
    try:
        with path.open("rb") as f:
            sample = f.read(4096)
    except OSError:
        # If we can't even read it, refuse.
        return True
    return b"\x00" in sample


def resolve_safe_path(root: Path, requested: str) -> Path:
    """Resolve a vault-relative path safely, refusing escapes and binary files.

    Raises :class:`VaultPathError` on:
      - ``..`` traversal segments (after normalisation, must remain under root)
      - absolute paths outside root
      - symlink escapes (the resolved path is outside root)
      - binary files
    """
    if not isinstance(requested, str) or not requested.strip():
        raise VaultPathError("path must be a non-empty string")
    if "\x00" in requested:
        raise VaultPathError("path contains a NUL byte")
    # Normalise; use the relative form. We refuse absolute outside-root at the
    # string level too: if the operator gave us /etc/passwd, that's clearly wrong.
    candidate = (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise VaultPathError(f"path escapes Vault root: {requested!r}")
    if not candidate.exists():
        raise VaultPathError(f"path does not exist: {requested!r}")
    if candidate.is_dir():
        raise VaultPathError(f"path is a directory, not a file: {requested!r}")
    if _is_probably_binary(candidate):
        raise VaultPathError(f"path appears to be a binary file: {requested!r}")
    return candidate


def _require_int(cfg: Dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VaultConfigError(f"{key} must be a non-negative integer")
    if value < minimum:
        raise VaultConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


def _require_bool(cfg: Dict[str, Any], key: str) -> bool:
    value = cfg.get(key)
    if not isinstance(value, bool):
        raise VaultConfigError(f"{key} must be a boolean")
    return value


def _require_str(cfg: Dict[str, Any], key: str, allowed: tuple) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise VaultConfigError(f"{key} must be one of {allowed}, got {value!r}")
    return value


def _normalize_mode_value(value: Any) -> Any:
    """Normalize YAML 1.1 boolean ``False`` (and its ``Off``/``No`` siblings)
    to the documented ``"off"`` mode value, but only for this single field.

    PyYAML's ``safe_load`` defaults to YAML 1.1, so an operator who follows
    the documented rollback literally — ``mode: off`` — gets Python
    ``bool(False)``. Without normalization, the validator rejects ``False``
    as not-a-string and ``register()`` fails closed, which is exactly the
    opposite of the intended rollback (built-in file tools stay blocked AND
    the ``vault_context`` surface disappears). Normalizing ``False`` to
    ``"off"`` keeps the rollback a single keystroke from the operator.

    Boolean ``True`` is NOT accepted as an undocumented mode — the spec
    enumerates exactly three valid string values; any other value still
    raises :class:`VaultConfigError`. This preserves the strict allow-list
    the architecture memo mandates.
    """
    if value is False:
        return "off"
    if value is True:
        raise VaultConfigError(
            "mode: boolean True is not a documented mode; use one of "
            "('enforce', 'audit', 'off')"
        )
    return value


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the plugin configuration dict.

    Returns a normalised config (defaults filled in). Raises :class:`VaultConfigError`
    on invalid / unsafe values. The hard ceiling is capped at
    :data:`HARD_CEILING_COMPILED_CAP` regardless of operator input.
    """
    out: Dict[str, Any] = dict(cfg)
    # vault_root is mandatory
    if "vault_root" not in cfg:
        raise VaultConfigError("vault_root is required")
    out["vault_root"] = str(resolve_vault_root(cfg["vault_root"]))

    # Limits
    out["default_budget_chars"] = _require_int(cfg, "default_budget_chars", minimum=1)
    out["hard_ceiling_chars"] = _require_int(cfg, "hard_ceiling_chars", minimum=1)
    out["large_file_chars"] = _require_int(cfg, "large_file_chars", minimum=1)
    out["candidate_limit"] = _require_int(cfg, "candidate_limit", minimum=1)
    out["max_primary_extracts"] = _require_int(cfg, "max_primary_extracts", minimum=1)
    out["max_expansion_extracts"] = _require_int(cfg, "max_expansion_extracts", minimum=1)
    out["max_range_lines"] = _require_int(cfg, "max_range_lines", minimum=1)
    out["max_range_chars"] = _require_int(cfg, "max_range_chars", minimum=1)

    # Hard-ceiling compiled cap
    if out["hard_ceiling_chars"] > HARD_CEILING_COMPILED_CAP:
        raise VaultConfigError(
            f"hard_ceiling_chars ({out['hard_ceiling_chars']}) exceeds compiled cap "
            f"({HARD_CEILING_COMPILED_CAP})"
        )

    # default <= hard ceiling
    if out["default_budget_chars"] > out["hard_ceiling_chars"]:
        raise VaultConfigError(
            f"default_budget_chars ({out['default_budget_chars']}) cannot exceed "
            f"hard_ceiling_chars ({out['hard_ceiling_chars']})"
        )

    # Booleans
    out["query_log_enabled"] = _require_bool(cfg, "query_log_enabled")
    out["log_raw_query_terms"] = _require_bool(cfg, "log_raw_query_terms")
    out["snapshots_enabled"] = _require_bool(cfg, "snapshots_enabled")
    out["fts5_enabled"] = _require_bool(cfg, "fts5_enabled")
    out["block_direct_file_reads"] = _require_bool(cfg, "block_direct_file_reads")

    # log_raw_query_terms must remain false (spec)
    if out["log_raw_query_terms"]:
        raise VaultConfigError("log_raw_query_terms must remain false (spec)")

    # Mode — YAML 1.1 turns the documented unquoted ``mode: off`` into
    # Python ``bool(False)``; normalize before the allow-list check so the
    # literal operator rollback works. Boolean ``True`` is still refused
    # by _normalize_mode_value itself.
    out["mode"] = _normalize_mode_value(cfg.get("mode"))
    out["mode"] = _require_str(out, "mode", _VALID_MODES)

    return out


def default_config() -> Dict[str, Any]:
    """The locked spec defaults (used when operator config is missing)."""
    return {
        "vault_root": "/root/Documents/Obsidian Vault",
        "mode": "enforce",
        "default_budget_chars": DEFAULT_BUDGET,
        "hard_ceiling_chars": DEFAULT_HARD_CEILING,
        "large_file_chars": LARGE_FILE_CHARS,
        "candidate_limit": CANDIDATE_LIMIT,
        "max_primary_extracts": MAX_PRIMARY_EXTRACTS,
        "max_expansion_extracts": MAX_EXPANSION_EXTRACTS,
        "max_range_lines": MAX_RANGE_LINES,
        "max_range_chars": MAX_RANGE_CHARS,
        "query_log_enabled": True,
        "log_raw_query_terms": False,
        "snapshots_enabled": False,
        "fts5_enabled": False,
        "block_direct_file_reads": True,
    }

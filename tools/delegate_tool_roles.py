"""Named role definitions for delegate_task (issue #112369, layer 1).

A role is a small YAML file resolved at spawn time:

    ~/.hermes/roles/<name>.yaml             user-level, the ``~/.hermes/skills/`` analogue
    <workspace>/.hermes/roles/<name>.yaml   workspace-level, AGENTS.md-style discovery

Discovery reuses the skills convention (``agent/skill_utils``): the workspace root is the nearest
``.git`` ancestor (``find_project_root``) and project dirs come FIRST, so a workspace role overrides a
same-named user role.

Everything a role can express NARROWS a spawn — it forces the leaf tool surface, intersects its toolset
allowlist with what the parent actually holds, pins a model route, prepends a charter to the child's
system prompt and can skip project-context-file injection. Nothing here widens authority: the parent's
own toolsets and the depth budget still bound every child, and an allowlist that resolves to nothing is
refused instead of degrading to "inherit everything".

Deliberately NOT in this layer (see the issue's phases): ``output_schema`` enforcement on the verdict
(#35688 owns the doer/reviewer harness), the ``delegated_role`` session field plus its usage/browsing
attribution (#41554), persona overlays (#80995) and any desktop/GUI surface (phase D).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

ROLE_SCHEMA = "hermes.role/v1"

_BUILTIN_ROLES = frozenset({"leaf", "orchestrator"})
_ROLE_SUFFIXES = (".yaml", ".yml")
# A role name is a file stem: anything path-like (``..``, ``/``, drive letters) must never reach the
# filesystem join below.
_ROLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_TOOL_MODES = frozenset({"allowlist", "denylist", "inherit"})
_KNOWN_KEYS = frozenset(
    {"schema", "name", "description", "prompt", "spawn", "model", "tools", "context", "output_schema"}
)


class RoleDefinitionError(ValueError):
    """A role that cannot be resolved or parsed. The message is user-facing (tool_error text)."""


@dataclass(frozen=True)
class RoleDefinition:
    """One parsed ``hermes.role/v1`` file. Unset fields keep today's spawn behavior."""

    name: str
    path: Path
    prompt: str = ""
    description: str = ""
    can_delegate: Optional[bool] = None
    model: Optional[Dict[str, str]] = None
    tools_mode: str = "inherit"
    toolsets: Tuple[str, ...] = ()
    context_files: Optional[bool] = None

    def charter(self) -> str:
        """Standing instructions with the ``{name}`` placeholder filled in (never ``str.format``: a
        charter is prose/code and may legitimately contain other braces)."""
        return self.prompt.replace("{name}", self.name)

    def injects_context_files(self) -> bool:
        """False only on an explicit ``context.context_files: false`` (unset = today's behavior)."""
        return self.context_files is not False


@dataclass(frozen=True)
class RoleSpec:
    """A resolved ``delegate_task(role=...)`` value: a built-in label or a named definition."""

    name: str
    definition: Optional[RoleDefinition] = None


# ── Discovery ──────────────────────────────────────────────────────────────


def _workspace_roles_dir() -> Optional[Path]:
    """``<workspace>/.hermes/roles`` for the nearest ``.git`` ancestor, or None outside a repo."""
    try:
        from agent.skill_utils import find_project_root

        root = find_project_root()
    except Exception as exc:  # discovery is best-effort: a broken cwd must not break spawning
        logger.debug("roles: workspace root discovery failed: %s", exc)
        return None
    return None if root is None else Path(root) / ".hermes" / "roles"


def role_dirs() -> List[Path]:
    """Role dirs in precedence order: workspace, then user. Resolved at call time so each served
    profile reads its own home."""
    from hermes_constants import get_hermes_home

    workspace = _workspace_roles_dir()
    dirs = [workspace] if workspace is not None else []
    dirs.append(Path(get_hermes_home()) / "roles")
    return dirs


def role_file(name: Any) -> Optional[Path]:
    """The first existing role file for *name* (workspace wins), or None."""
    name = str(name or "").strip()
    if not _ROLE_NAME_RE.match(name):
        return None
    for directory in role_dirs():
        for suffix in _ROLE_SUFFIXES:
            candidate = directory / f"{name}{suffix}"
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
    return None


def available_role_names() -> List[str]:
    """Every discoverable role name, workspace first then user (first-wins, like skills)."""
    names: List[str] = []
    for directory in role_dirs():
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for path in entries:
            stem, suffix = os.path.splitext(path.name)
            if suffix in _ROLE_SUFFIXES and stem not in names and path.is_file():
                names.append(stem)
    return names


def _unknown_role_message(name: str) -> str:
    searched = ", ".join(str(d) for d in role_dirs())
    known = available_role_names()
    known_txt = f" Roles found: {', '.join(known)}." if known else ""
    return (
        f"Unknown delegate_task role {name!r}. Use the built-in 'leaf' or 'orchestrator', or add a role "
        f"definition ({name}.yaml, schema: {ROLE_SCHEMA}) under one of: {searched}.{known_txt}"
    )


# ── Parsing ────────────────────────────────────────────────────────────────


def _section(raw: Dict[str, Any], key: str, path: Path) -> Dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RoleDefinitionError(f"Role file {path}: '{key}' must be a mapping.")
    return value


def _warn_unknown(unknown: Any, path: Path, prefix: str = "") -> None:
    """A typo'd key (``tool:`` for ``tools:``) would otherwise silently do nothing."""
    if unknown:
        logger.warning("Role file %s: ignoring unknown key(s): %s", path, ", ".join(f"{prefix}{k}" for k in sorted(unknown)))


def _bool_field(section: Dict[str, Any], key: str, path: Path, label: str) -> Optional[bool]:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RoleDefinitionError(f"Role file {path}: {label}.{key} must be true or false, got {value!r}.")
    return value


def _name_list(value: Any, path: Path, label: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RoleDefinitionError(f"Role file {path}: {label} must be a list of names, got {value!r}.")
    names = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise RoleDefinitionError(f"Role file {path}: {label} entries must be non-empty names, got {entry!r}.")
        names.append(entry.strip())
    return tuple(dict.fromkeys(names))


def load_role_definition(path: Any) -> RoleDefinition:
    """Parse and validate one role file. Raises ``RoleDefinitionError`` naming the file and the problem."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoleDefinitionError(f"Role file {path} cannot be read: {exc}") from exc
    from agent.skill_utils import yaml_load

    try:
        raw = yaml_load(text.removeprefix("\ufeff"))
    except Exception as exc:
        raise RoleDefinitionError(f"Role file {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise RoleDefinitionError(f"Role file {path} must hold a YAML mapping (schema: {ROLE_SCHEMA}).")

    schema = str(raw.get("schema") or "").strip()
    if schema != ROLE_SCHEMA:
        detail = "'schema' is missing" if not schema else f"found {schema!r}"
        raise RoleDefinitionError(f"Role file {path}: expected schema: {ROLE_SCHEMA}, {detail}.")

    stem = os.path.splitext(path.name)[0]
    name = str(raw.get("name") or stem).strip()
    if not _ROLE_NAME_RE.match(name):
        raise RoleDefinitionError(
            f"Role file {path}: name {name!r} is not a valid role name (letters, digits, '.', '_', '-')."
        )
    if name != stem:
        raise RoleDefinitionError(
            f"Role file {path}: name {name!r} does not match the file name {stem!r} — delegate_task(role=...) "
            f"resolves the file name."
        )

    _warn_unknown(set(raw) - _KNOWN_KEYS, path)
    if "output_schema" in raw:
        logger.warning(
            "Role file %s declares output_schema; role-enforced output schemas are not part of this release "
            "(see #35688) — the key is ignored. Use delegate_task(output_schema=...) per task instead.",
            path,
        )

    description = raw.get("description") or ""
    if not isinstance(description, str):
        raise RoleDefinitionError(f"Role file {path}: 'description' must be text.")
    prompt = raw.get("prompt") or ""
    if not isinstance(prompt, str):
        raise RoleDefinitionError(f"Role file {path}: 'prompt' must be text.")

    spawn = _section(raw, "spawn", path)
    _warn_unknown(set(spawn) - {"can_delegate"}, path, "spawn.")
    model_section = _section(raw, "model", path)
    _warn_unknown(set(model_section) - {"provider", "model"}, path, "model.")
    model: Dict[str, str] = {}
    for key in ("provider", "model"):
        value = model_section.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise RoleDefinitionError(f"Role file {path}: model.{key} must be a non-empty name.")
        model[key] = value.strip()

    tools = _section(raw, "tools", path)
    _warn_unknown(set(tools) - {"mode", "toolsets"}, path, "tools.")
    mode = tools.get("mode") or "inherit"
    if not isinstance(mode, str) or mode.strip().lower() not in _TOOL_MODES:
        raise RoleDefinitionError(
            f"Role file {path}: tools.mode must be one of {sorted(_TOOL_MODES)}, got {mode!r}."
        )
    mode = mode.strip().lower()
    toolsets = _name_list(tools.get("toolsets"), path, "tools.toolsets")
    if mode in {"allowlist", "denylist"} and not toolsets:
        raise RoleDefinitionError(f"Role file {path}: tools.mode {mode!r} requires a non-empty tools.toolsets.")

    context = _section(raw, "context", path)
    _warn_unknown(set(context) - {"context_files"}, path, "context.")

    return RoleDefinition(
        name=name, path=path, prompt=prompt, description=description.strip(),
        can_delegate=_bool_field(spawn, "can_delegate", path, "spawn"),
        model=model or None, tools_mode=mode, toolsets=toolsets,
        context_files=_bool_field(context, "context_files", path, "context"),
    )


def resolve_role(raw: Any) -> RoleSpec:
    """``RoleSpec`` for a delegate_task ``role`` value.

    ``leaf``/``orchestrator`` (case-insensitive) and an absent value are the built-ins. Any other name
    must resolve to a definition file: an unknown or unparsable role is a typed error, never a silent
    leaf — silently dropping it would run the child without the charter and narrowing it was asked for.
    """
    name = str(raw).strip() if isinstance(raw, str) else ""
    if not name:
        return RoleSpec(name="leaf")
    if name.lower() in _BUILTIN_ROLES:
        return RoleSpec(name=name.lower())
    path = role_file(name)
    if path is None:
        raise RoleDefinitionError(_unknown_role_message(name))
    return RoleSpec(name=name, definition=load_role_definition(path))

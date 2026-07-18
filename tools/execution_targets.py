"""Resolve static named terminal execution targets.

The resolver is deliberately small and lazy: importing tool schemas never reads
user configuration, and configured names never enter a schema.  Legacy flat
terminal configuration stays on the existing environment-variable path.
"""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import os
import threading
from typing import Any, Hashable, Iterable, Mapping, Optional


class ExecutionTargetError(ValueError):
    """Raised when terminal target configuration or selection is invalid."""


_effective_config_override: ContextVar[dict[str, Any] | None] = ContextVar(
    "hermes_execution_target_config", default=None,
)
_classic_config_lock = threading.RLock()
_classic_config_override: dict[str, Any] | None = None


def set_execution_target_config_source(config: Mapping[str, Any] | None) -> None:
    """Use an entry point's already-effective config for target resolution.

    The classic CLI has project-fallback and ``--ignore-user-config`` rules
    that differ from the shared loader. Registering its merged result here
    keeps tool routing on the same authority boundary.
    """
    global _classic_config_override
    value = deepcopy(dict(config)) if isinstance(config, Mapping) else None
    _effective_config_override.set(value)
    with _classic_config_lock:
        _classic_config_override = deepcopy(value)


def _active_profile_scope() -> str:
    """Return a stable multiplex-profile scope, empty in legacy mode."""
    try:
        from agent.secret_scope import is_multiplex_active
        if not is_multiplex_active():
            return ""
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home().resolve())
        return hashlib.sha256(home.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return ""


@dataclass(frozen=True)
class ExecutionTargetResolution:
    """A resolved execution target and its inherited terminal configuration."""

    target: str
    backend: str
    config: Mapping[str, Any]
    named: bool
    is_default: bool = True
    profile_scope: str = ""

    def scope_task_key(self, task_key: Hashable) -> Hashable:
        if not self.profile_scope:
            return task_key
        return f"profile-{self.profile_scope}:{task_key}"

    def environment_key(self, task_key: Hashable) -> Hashable:
        """Scope an already-collapsed environment key to this target."""
        scoped = self.scope_task_key(task_key)
        return (scoped, self.target) if self.named else scoped

    def session_key(self, raw_session_key: Optional[str]) -> Hashable:
        """Scope per-session state without changing legacy string keys."""
        key = str(self.scope_task_key(str(raw_session_key or "default")))
        return (key, self.target) if self.named else key

    def backend_task_id(self, task_key: Hashable) -> str:
        """Return a backend-safe isolation id without exposing target syntax."""
        base = str(self.scope_task_key(task_key))
        if not self.named:
            return base
        digest = hashlib.sha256(self.target.encode("utf-8")).hexdigest()[:12]
        return f"{base}-target-{digest}"

    def metadata(self, *, cwd: Optional[str] = None) -> dict[str, Any]:
        data: dict[str, Any] = {"target": self.target, "backend": self.backend}
        if cwd:
            data["cwd"] = cwd
        return data


def _load_merged_config() -> dict[str, Any]:
    """Load merged config lazily so importing fixed tool schemas stays cheap."""
    override = _effective_config_override.get()
    if override is None:
        try:
            from agent.secret_scope import is_multiplex_active

            multiplex = is_multiplex_active()
        except Exception:
            multiplex = False
        if not multiplex:
            with _classic_config_lock:
                override = deepcopy(_classic_config_override)
    if override is not None:
        return deepcopy(override)
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    return config if isinstance(config, dict) else {}


def _available(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "(none)"


def resolve_execution_target(
    target: Optional[str] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> ExecutionTargetResolution:
    """Resolve *target* against merged ``terminal`` configuration.

    With no non-empty ``terminal.targets`` mapping, only omitted target and
    explicit ``"default"`` are valid and the returned configuration marks the
    legacy env-driven path.  With named targets, top-level terminal settings
    are inherited and the selected target mapping overrides them.
    """
    if target is not None and (not isinstance(target, str) or not target):
        raise ExecutionTargetError("Execution target must be a non-empty string.")

    root = config if config is not None else _load_merged_config()
    profile_scope = _active_profile_scope()
    if not isinstance(root, Mapping):
        raise ExecutionTargetError("Terminal configuration must be a mapping.")
    terminal = root.get("terminal", {})
    if terminal is None:
        terminal = {}
    if not isinstance(terminal, Mapping):
        raise ExecutionTargetError("terminal must be a mapping.")

    raw_targets = terminal.get("targets")
    if raw_targets is None or raw_targets == {}:
        if target not in (None, "default"):
            raise ExecutionTargetError(
                f"Unknown execution target {target!r}. Named targets are not configured. "
                f"Available targets: {_available(['default'])}."
            )
        flat = dict(terminal)
        flat.pop("targets", None)
        flat.pop("default_target", None)
        # Legacy flat mode preserves the historical TERMINAL_ENV precedence.
        # Metadata must describe the backend that will actually execute, not a
        # stale config.yaml value hidden by an explicit launcher/.env override.
        backend = str(
            os.getenv("TERMINAL_ENV")
            or flat.get("backend")
            or flat.get("env_type")
            or "local"
        ).strip().lower() or "local"
        return ExecutionTargetResolution(
            target="default", backend=backend, config=flat, named=False,
            is_default=True, profile_scope=profile_scope,
        )

    if not isinstance(raw_targets, Mapping):
        raise ExecutionTargetError("terminal.targets must be a mapping.")

    invalid_names = [
        name for name in raw_targets
        if not isinstance(name, str) or not name
    ]
    if invalid_names:
        raise ExecutionTargetError("terminal.targets names must be non-empty strings.")

    names = sorted(raw_targets)
    for name in names:
        if not isinstance(raw_targets[name], Mapping):
            raise ExecutionTargetError(
                f"terminal.targets[{name!r}] must be a mapping. "
                f"Available targets: {_available(names)}."
            )

    selected = target if target is not None else terminal.get("default_target")
    if not isinstance(selected, str) or not selected or selected not in raw_targets:
        if target is not None:
            prefix = f"Unknown execution target {target!r}."
        else:
            prefix = (
                "terminal.default_target must be a non-empty name present in "
                "terminal.targets."
            )
        raise ExecutionTargetError(
            f"{prefix} Available targets: {_available(names)}."
        )

    merged = {
        key: value
        for key, value in terminal.items()
        if key not in {"targets", "default_target"}
    }
    merged.update(dict(raw_targets[selected]))
    backend = str(
        merged.get("backend") or merged.get("env_type") or "local"
    ).strip().lower() or "local"
    return ExecutionTargetResolution(
        target=selected, backend=backend, config=merged, named=True,
        is_default=selected == terminal.get("default_target"),
        profile_scope=profile_scope,
    )


def list_execution_targets(
    *, config: Optional[Mapping[str, Any]] = None,
) -> tuple[ExecutionTargetResolution, ...]:
    """Return configured targets in deterministic order for runtime guidance.

    Tool schemas remain static for prompt-cache stability; callers such as the
    system-prompt environment hint can use this lightweight runtime inventory
    to tell the model which fixed-string target values are available.
    """
    root = config if config is not None else _load_merged_config()
    terminal = root.get("terminal", {}) if isinstance(root, Mapping) else {}
    targets = terminal.get("targets") if isinstance(terminal, Mapping) else None
    if not isinstance(targets, Mapping) or not targets:
        return (resolve_execution_target(config=root),)
    # Resolve the default first so name/default validation uses the same clear
    # ExecutionTargetError shape as an ordinary omitted-selector tool call.
    resolve_execution_target(config=root)
    return tuple(
        resolve_execution_target(name, config=root)
        for name in sorted(targets)
    )
